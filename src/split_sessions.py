"""
STAGE A — owned by Person A.

Loads raw S-*.csv (IMU) and V-*.csv (GPS), detects timestamp breaks
(concatenated-session corruption — see project status report), and emits
clean, monotonic, single-trip session files per the schema in
data_contract.md.

This is the productionized version of find_timestamp_breaks.py from the
status report: instead of just diagnosing, it actually splits and writes.

Usage:
    python split_sessions.py --imu data/raw/S-M.csv --gps data/raw/V-M.csv \
        --out data/processed/sessions
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from common import (
    RAW_TIME_COL, RAW_IMU_COLS, RAW_GRAVITY_COLS, RAW_GPS_TIME_COL,
    RAW_GPS_LAT_COL, RAW_GPS_LON_COL, RAW_GPS_SPEED_COL, MIN_SESSION_SEC,
    compute_dt, find_break_indices, compensate_gravity, load_raw_csv,
)


def diagnose(t_ms: np.ndarray, label: str):
    """Print the same kind of dt-stats sanity check that found the real bug."""
    dt = compute_dt(t_ms)
    print(f"[{label}] dt stats: mean={np.nanmean(dt):.4f}s std={np.nanstd(dt):.4f}s "
          f"min={np.nanmin(dt):.4f}s max={np.nanmax(dt):.4f}s")
    breaks = find_break_indices(t_ms)
    print(f"[{label}] {len(breaks)} break(s) detected at row indices: "
          f"{breaks[:20].tolist()}{' ...' if len(breaks) > 20 else ''}")
    return breaks


def split_by_breaks(df: pd.DataFrame, t_ms_col: str, breaks: np.ndarray):
    """Yield (start_idx, end_idx) row ranges for each contiguous clean segment."""
    bounds = [0] + list(breaks) + [len(df)]
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        if end - start < 2:
            continue
        yield start, end


def process(imu_path: str, gps_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    imu_raw = load_raw_csv(imu_path)   # handles real-data encoding (cp1252) + strips
    gps_raw = load_raw_csv(gps_path)   # whitespace from headers -- plain read_csv()
                                        # crashes on real S-M.csv with UnicodeDecodeError

    imu_t_ms = imu_raw[RAW_TIME_COL].to_numpy()
    gps_t_ms = gps_raw[RAW_GPS_TIME_COL].to_numpy()

    print("=== Diagnostics before splitting ===")
    imu_breaks = diagnose(imu_t_ms, "IMU")
    gps_breaks = diagnose(gps_t_ms, "GPS")

    # IMPORTANT: we do NOT slice GPS by absolute-time overlap with each IMU
    # segment. If both raw files reset their clock at each session boundary
    # (the same corruption pattern found in the real data), every session's
    # absolute time range looks like [0, ~60s] and overlaps ALL sessions,
    # silently pulling in the wrong GPS rows. Instead we split IMU and GPS
    # independently on their own breaks and pair the resulting segments by
    # ORDER — valid as long as both files come from the same synchronized
    # recording (same number of sessions in each, same order). If your GPS
    # and IMU files are NOT independently reset (i.e. share one continuous
    # absolute clock), switch this back to time-based slicing.
    imu_segments = list(split_by_breaks(imu_raw, RAW_TIME_COL, imu_breaks))
    gps_segments = list(split_by_breaks(gps_raw, RAW_GPS_TIME_COL, gps_breaks))

    if len(imu_segments) != len(gps_segments):
        print(f"WARNING: {len(imu_segments)} IMU segments vs {len(gps_segments)} "
              f"GPS segments — counts don't match, order-based pairing may be "
              f"wrong. Inspect manually before trusting the output.")

    manifest_rows = []
    session_id = 0

    for (imu_start, imu_end), (gps_start, gps_end) in zip(imu_segments, gps_segments):
        seg = imu_raw.iloc[imu_start:imu_end].copy()
        t0_ms = seg[RAW_TIME_COL].iloc[0]
        t_sec = (seg[RAW_TIME_COL].to_numpy() - t0_ms) / 1000.0
        duration = t_sec[-1] - t_sec[0]

        if duration < MIN_SESSION_SEC:
            continue

        # Re-verify monotonicity post-split — this should always pass now,
        # but check anyway so a silent bug upstream can't slip through.
        assert np.all(np.diff(t_sec) > 0), (
            f"Session {session_id} still non-monotonic after split — "
            f"break detection missed something, investigate before proceeding."
        )

        # Gravity compensation: raw ACCELEROMETER X/Y includes gravity
        # (confirmed empirically -- see common.py's RAW_GRAVITY_COLS comment).
        # Subtract it here, at Stage A, since data_contract.md's session
        # schema has no raw-gravity column for Stage B to do this itself.
        acc_x_raw = seg[RAW_IMU_COLS["acc_x"]].to_numpy()
        acc_y_raw = seg[RAW_IMU_COLS["acc_y"]].to_numpy()
        grav_x = seg[RAW_GRAVITY_COLS["grav_x"]].to_numpy()
        grav_y = seg[RAW_GRAVITY_COLS["grav_y"]].to_numpy()
        acc_x, acc_y = compensate_gravity(acc_x_raw, acc_y_raw, grav_x, grav_y)

        imu_out = pd.DataFrame({
            "t_sec": t_sec,
            "acc_x": acc_x,
            "acc_y": acc_y,
            "gyro_yaw": seg[RAW_IMU_COLS["gyro_yaw"]].to_numpy(),
            "gyro_pitch": seg[RAW_IMU_COLS["gyro_pitch"]].to_numpy(),
            "gyro_roll": seg[RAW_IMU_COLS["gyro_roll"]].to_numpy(),
        })

        gps_seg = gps_raw.iloc[gps_start:gps_end].copy()
        gps_t0_ms = gps_seg[RAW_GPS_TIME_COL].iloc[0]
        gps_t_sec = (gps_seg[RAW_GPS_TIME_COL].to_numpy() - gps_t0_ms) / 1000.0

        gps_out = pd.DataFrame({
            "t_sec": gps_t_sec,
            "lat": gps_seg[RAW_GPS_LAT_COL].to_numpy(),
            "lon": gps_seg[RAW_GPS_LON_COL].to_numpy(),
            "speed": (gps_seg[RAW_GPS_SPEED_COL].to_numpy()
                      if RAW_GPS_SPEED_COL and RAW_GPS_SPEED_COL in gps_seg.columns
                      else np.full(len(gps_seg), np.nan)),
        })

        sid_str = f"{session_id:04d}"
        imu_out.to_parquet(os.path.join(out_dir, f"session_{sid_str}_imu.parquet"))
        gps_out.to_parquet(os.path.join(out_dir, f"session_{sid_str}_gps.parquet"))

        dt = np.diff(t_sec)
        manifest_rows.append({
            "session_id": session_id,
            "source_file": os.path.basename(imu_path),
            "start_idx": imu_start,
            "end_idx": imu_end,
            "n_imu_samples": len(imu_out),
            "n_gps_fixes": len(gps_out),
            "duration_sec": duration,
            "dt_mean": dt.mean(),
            "dt_std": dt.std(),
            "dt_max": dt.max(),
        })
        session_id += 1

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)

    print(f"\n=== Done ===")
    print(f"{session_id} clean session(s) written to {out_dir}")
    if len(manifest):
        print(manifest[["session_id", "n_imu_samples", "n_gps_fixes", "duration_sec"]]
              .to_string(index=False))
    else:
        print("WARNING: no sessions survived MIN_SESSION_SEC filtering — "
              "check thresholds in common.py or your raw data.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--imu", required=True, help="path to raw S-*.csv")
    ap.add_argument("--gps", required=True, help="path to raw V-*.csv")
    ap.add_argument("--out", default="data/processed/sessions")
    args = ap.parse_args()
    process(args.imu, args.gps, args.out)
