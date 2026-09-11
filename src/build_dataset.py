"""
STAGE B — owned by Person B.

Reads Stage A's clean per-session files (see data_contract.md), aligns GPS
ground truth onto the IMU timeline, computes both label representations
((dx,dy) and (speed, delta_heading)), builds SESSION-AWARE sliding windows
(never crossing a session boundary — that was the old bug), flags ZUPT
windows, and writes a session-split train/val/test dataset.

Usage:
    python build_dataset.py --sessions data/processed/sessions \
        --out data/processed/dataset --window-len 100 --stride 20
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from common import latlon_to_local_xy, xy_to_speed_heading, detect_zupt

IMU_COLS = ["acc_x", "acc_y", "gyro_yaw", "gyro_pitch", "gyro_roll"]
ACC_COLS = ["acc_x", "acc_y"]
GYRO_COLS = ["gyro_yaw", "gyro_pitch", "gyro_roll"]


def build_session_labels(imu_df: pd.DataFrame, gps_df: pd.DataFrame):
    """
    Interpolate GPS-derived position onto the IMU timeline for one session,
    and compute both (dx,dy) and (speed, delta_heading) per IMU sample.
    Returns a DataFrame aligned 1:1 with imu_df rows (first row has NaN
    displacement since there's nothing to diff against).
    """
    if len(gps_df) < 2:
        return None  # not enough GPS fixes to derive ground truth for this session

    lat0, lon0 = gps_df["lat"].iloc[0], gps_df["lon"].iloc[0]
    gx, gy = latlon_to_local_xy(gps_df["lat"].to_numpy(), gps_df["lon"].to_numpy(),
                                 lat0, lon0)

    # Interpolate GPS-derived x,y onto the (denser) IMU timeline. This assumes
    # gps t_sec is within the imu t_sec range, which it will be by construction
    # from Stage A. Extrapolation at the edges falls back to nearest value.
    imu_t = imu_df["t_sec"].to_numpy()
    x_interp = np.interp(imu_t, gps_df["t_sec"], gx)
    y_interp = np.interp(imu_t, gps_df["t_sec"], gy)

    dx = np.diff(x_interp, prepend=x_interp[0])
    dy = np.diff(y_interp, prepend=y_interp[0])

    speed, delta_heading = xy_to_speed_heading(imu_t, x_interp, y_interp)
    # xy_to_speed_heading returns len-1 arrays aligned to the 2nd point of each
    # pair; prepend a NaN/0 to line up with imu rows.
    speed = np.concatenate([[speed[0]], speed])
    delta_heading = np.concatenate([[delta_heading[0]], delta_heading])

    out = imu_df.copy()
    out["dx"] = dx
    out["dy"] = dy
    out["speed"] = speed
    out["delta_heading"] = delta_heading
    return out


def make_windows(labeled_df: pd.DataFrame, session_id: int, window_len: int, stride: int):
    """
    Slide a fixed-size window over ONE session's rows only (caller guarantees
    no cross-session mixing by calling this per-session). Label for a window
    is the cumulative (dx,dy) across the window and the (mean speed, summed
    delta_heading) — adjust to match whatever the model actually expects.
    """
    n = len(labeled_df)
    X_list, y_dxdy_list, y_sh_list, zupt_list = [], [], [], []

    acc = labeled_df[ACC_COLS].to_numpy()
    gyro = labeled_df[GYRO_COLS].to_numpy()
    zupt_flags = detect_zupt(acc, gyro)

    for start in range(0, n - window_len, stride):
        end = start + window_len
        window = labeled_df.iloc[start:end]

        X_list.append(window[IMU_COLS].to_numpy(dtype=np.float32))
        y_dxdy_list.append([window["dx"].sum(), window["dy"].sum()])
        y_sh_list.append([window["speed"].mean(), window["delta_heading"].sum()])
        zupt_list.append(bool(zupt_flags[start:end].mean() > 0.5))

    if not X_list:
        return None

    return {
        "X": np.stack(X_list),
        "y_dxdy": np.array(y_dxdy_list, dtype=np.float32),
        "y_speed_heading": np.array(y_sh_list, dtype=np.float32),
        "session_id": np.full(len(X_list), session_id, dtype=np.int32),
        "is_zupt": np.array(zupt_list, dtype=bool),
    }


def process(sessions_dir: str, out_dir: str, window_len: int, stride: int,
            val_frac: float, test_frac: float, seed: int):
    os.makedirs(out_dir, exist_ok=True)
    manifest = pd.read_csv(os.path.join(sessions_dir, "manifest.csv"))

    all_parts = []
    for _, row in manifest.iterrows():
        sid = row["session_id"]
        sid_str = f"{sid:04d}"
        imu_path = os.path.join(sessions_dir, f"session_{sid_str}_imu.parquet")
        gps_path = os.path.join(sessions_dir, f"session_{sid_str}_gps.parquet")
        if not (os.path.exists(imu_path) and os.path.exists(gps_path)):
            print(f"WARNING: missing files for session {sid}, skipping")
            continue

        imu_df = pd.read_parquet(imu_path)
        gps_df = pd.read_parquet(gps_path)

        labeled = build_session_labels(imu_df, gps_df)
        if labeled is None:
            print(f"WARNING: session {sid} has too few GPS fixes for ground "
                  f"truth, skipping")
            continue

        windows = make_windows(labeled, sid, window_len, stride)
        if windows is None:
            print(f"WARNING: session {sid} too short for window_len={window_len}, "
                  f"skipping")
            continue

        all_parts.append(windows)

    if not all_parts:
        raise RuntimeError("No usable sessions produced any windows — check "
                            "window_len/stride against your session lengths.")

    # Session-level train/val/test split — NEVER split by window, or windows
    # from the same trip leak across splits (the old pipeline's risk).
    rng = np.random.default_rng(seed)
    session_ids = manifest["session_id"].to_numpy().copy()
    rng.shuffle(session_ids)
    n = len(session_ids)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test_ids = set(session_ids[:n_test])
    val_ids = set(session_ids[n_test:n_test + n_val])
    train_ids = set(session_ids[n_test + n_val:])

    def concat_where(key_set):
        keys = ["X", "y_dxdy", "y_speed_heading", "session_id", "is_zupt"]
        filtered = {k: [] for k in keys}
        for part in all_parts:
            mask = np.isin(part["session_id"], list(key_set))
            if not mask.any():
                continue
            for k in keys:
                filtered[k].append(part[k][mask])
        return {k: (np.concatenate(v) if v else np.array([])) for k, v in filtered.items()}

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    split_rows = []
    for split_name, ids in splits.items():
        data = concat_where(ids)
        out_path = os.path.join(out_dir, f"{split_name}.npz")
        np.savez(out_path, **data)
        n_windows = len(data["X"]) if len(data.get("X", [])) else 0
        print(f"{split_name}: {len(ids)} session(s), {n_windows} window(s) -> {out_path}")
        split_rows.append({"split": split_name, "session_ids": sorted(ids),
                            "n_windows": n_windows})

    pd.DataFrame(split_rows).to_csv(os.path.join(out_dir, "split_manifest.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True, help="Stage A output dir")
    ap.add_argument("--out", default="data/processed/dataset")
    ap.add_argument("--window-len", type=int, default=100)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    process(args.sessions, args.out, args.window_len, args.stride,
             args.val_frac, args.test_frac, args.seed)
