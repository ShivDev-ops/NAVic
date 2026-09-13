"""
STAGE B -- owned by Person B.

Reads Stage A's clean per-session files (see data_contract.md), aligns GPS
ground truth onto the IMU timeline, computes both label representations
((dx,dy) and (speed, delta_heading)), builds SESSION-AWARE sliding windows
(never crossing a session boundary), flags ZUPT windows, and writes a
train/val/test dataset.

2026-09-14 update: with the DATE-column fix, the real data resolves to a
SINGLE continuous session (~2.94hr), not several. The old session-level
split (assign whole sessions to train/val/test) leaves train/val empty
when there's only one session to assign. Split logic now works two ways:

- If there are >= 3 sessions available (e.g. once more driver files are
  added), assigns whole sessions to train/val/test as before -- the
  stronger, more defensible split.
- If there are fewer sessions than splits need (the current situation:
  one ~2.94hr trip), each session is instead cut chronologically into
  contiguous train/val/test time blocks, with a gap of `window_len`
  samples at each boundary so no window straddles two splits.

KNOWN LIMITATION (flag this to judges): in the single-session fallback
path, train/val/test all come from the same trip/driver/road -- it's a
temporal split, not a held-out-driver split, so it's a weaker
generalization test than the session-level split. Add more driver files
and re-run once time allows.

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
    """Interpolate GPS-derived position onto the IMU timeline for one
    session, and compute both (dx,dy) and (speed, delta_heading) per IMU
    sample."""
    if len(gps_df) < 2:
        return None

    lat0, lon0 = gps_df["lat"].iloc[0], gps_df["lon"].iloc[0]
    gx, gy = latlon_to_local_xy(gps_df["lat"].to_numpy(), gps_df["lon"].to_numpy(),
                                 lat0, lon0)

    imu_t = imu_df["t_sec"].to_numpy()
    x_interp = np.interp(imu_t, gps_df["t_sec"], gx)
    y_interp = np.interp(imu_t, gps_df["t_sec"], gy)

    dx = np.diff(x_interp, prepend=x_interp[0])
    dy = np.diff(y_interp, prepend=y_interp[0])

    speed, delta_heading = xy_to_speed_heading(imu_t, x_interp, y_interp)
    speed = np.concatenate([[speed[0]], speed])
    delta_heading = np.concatenate([[delta_heading[0]], delta_heading])

    out = imu_df.copy()
    out["dx"] = dx
    out["dy"] = dy
    out["speed"] = speed
    out["delta_heading"] = delta_heading
    return out


def split_by_time(labeled_df: pd.DataFrame, val_frac: float, test_frac: float,
                   window_len: int):
    """
    Chronologically cut ONE session's rows into (train, val, test) blocks,
    with a `window_len`-sample gap at each boundary so no window built later
    can span across two splits.
    """
    n = len(labeled_df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    test_start = n - n_test
    val_end = test_start - window_len
    val_start = max(0, val_end - n_val)
    train_end = max(0, val_start - window_len)

    train_df = labeled_df.iloc[:train_end].reset_index(drop=True)
    val_df = labeled_df.iloc[val_start:val_end].reset_index(drop=True) if val_end > val_start else labeled_df.iloc[0:0]
    test_df = labeled_df.iloc[test_start:].reset_index(drop=True)
    return train_df, val_df, test_df


def make_windows(labeled_df: pd.DataFrame, session_id: int, window_len: int, stride: int):
    """Slide a fixed-size window over ONE session/split's rows only."""
    n = len(labeled_df)
    X_list, y_dxdy_list, y_sh_list, zupt_list = [], [], [], []

    if n <= window_len:
        return None

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
    n_sessions = len(manifest)

    keys = ["X", "y_dxdy", "y_speed_heading", "session_id", "is_zupt"]
    split_parts = {"train": [], "val": [], "test": []}
    split_session_ids = {"train": set(), "val": set(), "test": set()}

    # Decide split strategy: whole-session assignment needs at least 3
    # sessions (one per split) to avoid an empty split, same as before.
    use_session_level_split = n_sessions >= 3
    if use_session_level_split:
        rng = np.random.default_rng(seed)
        session_ids = manifest["session_id"].to_numpy().copy()
        rng.shuffle(session_ids)
        n = len(session_ids)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        session_split = {}
        for sid in session_ids[:n_test]:
            session_split[sid] = "test"
        for sid in session_ids[n_test:n_test + n_val]:
            session_split[sid] = "val"
        for sid in session_ids[n_test + n_val:]:
            session_split[sid] = "train"
    else:
        print(f"NOTE: only {n_sessions} session(s) available -- falling back "
              f"to a chronological within-session split (train/val/test all "
              f"drawn from the same trip). Add more driver files for a "
              f"proper held-out-session split before final evaluation.")

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

        if use_session_level_split:
            split_name = session_split[sid]
            windows = make_windows(labeled, sid, window_len, stride)
            if windows is None:
                print(f"WARNING: session {sid} too short for window_len={window_len}, skipping")
                continue
            split_parts[split_name].append(windows)
            split_session_ids[split_name].add(int(sid))
        else:
            train_df, val_df, test_df = split_by_time(labeled, val_frac, test_frac, window_len)
            for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
                windows = make_windows(split_df, sid, window_len, stride)
                if windows is None:
                    print(f"WARNING: session {sid}'s {split_name} block too "
                          f"short for window_len={window_len}, skipping")
                    continue
                split_parts[split_name].append(windows)
                split_session_ids[split_name].add(int(sid))

    if not any(split_parts.values()):
        raise RuntimeError("No usable sessions produced any windows -- check "
                            "window_len/stride against your session lengths.")

    def concat_parts(parts):
        filtered = {k: [] for k in keys}
        for part in parts:
            for k in keys:
                filtered[k].append(part[k])
        return {k: (np.concatenate(v) if v else np.array([])) for k, v in filtered.items()}

    split_rows = []
    for split_name, parts in split_parts.items():
        data = concat_parts(parts)
        out_path = os.path.join(out_dir, f"{split_name}.npz")
        np.savez(out_path, **data)
        n_windows = len(data["X"]) if len(data.get("X", [])) else 0
        print(f"{split_name}: {len(split_session_ids[split_name])} session(s), "
              f"{n_windows} window(s) -> {out_path}")
        split_rows.append({"split": split_name,
                            "session_ids": sorted(split_session_ids[split_name]),
                            "n_windows": n_windows,
                            "split_strategy": "session-level" if use_session_level_split else "chronological-within-session"})

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
