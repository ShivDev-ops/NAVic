"""
STAGE A -- owned by Person A.

Loads raw S-*.csv (IMU) and V-*.csv (GPS), detects timestamp breaks using
the device's WALL CLOCK (DATE column) -- not the buggy TIME SINCE START
(ms) counter -- and emits clean, monotonic, single-trip session files per
the schema in data_contract.md.

2026-09-14 update (per Person A's Stage A finding):
- TIME SINCE START (ms) is a confirmed-buggy internal counter (a -4426s
  jump was found that corresponds to ~1.16s of real elapsed time). Break
  detection now uses s_time_of_day_sec() (the DATE column) instead. On the
  real M-category file this finds ZERO real splices -- one continuous
  ~2.94hr trip. Don't be alarmed if manifest.csv ends up with a single row;
  that's expected now, not a bug.
- IMU and GPS no longer get split independently and paired by ORDER. V's
  clock is a different reference (seconds-since-midnight, confirmed ~3600s
  offset from S's DATE column) and doesn't share S's session structure, so
  order-based pairing is unsafe. GPS rows are now matched to each IMU
  segment by actual overlapping wall-clock time, using find_time_offset().
- Per-sample t_sec (written into the output parquet) is now also derived
  from the wall clock, not the buggy ms counter, so mid-segment timing is
  trustworthy too, not just the break points.
- V-M.csv's GPS-relevant columns can repeat the same timestamp on
  consecutive rows (looks like a full-rate CAN log where GPS itself only
  updates periodically) -- deduped here (keep last) before use, since
  downstream np.interp requires strictly increasing x-values.

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
    RAW_IMU_COLS, RAW_GRAVITY_COLS, RAW_GPS_TIME_COL,
    RAW_GPS_LAT_COL, RAW_GPS_LON_COL, RAW_GPS_SPEED_COL, MIN_SESSION_SEC,
    find_break_indices_sec, compensate_gravity, load_raw_csv,
    s_time_of_day_sec, find_time_offset,
)


def diagnose_sec(t_sec: np.ndarray, label: str):
    """Print dt-stats on a wall-clock-derived seconds array and return break
    indices. Replaces the old ms-counter diagnose() for real data."""
    dt = np.empty_like(t_sec, dtype=float)
    dt[0] = np.nan
    dt[1:] = np.diff(t_sec)
    print(f"[{label}] dt stats: mean={np.nanmean(dt):.4f}s std={np.nanstd(dt):.4f}s "
          f"min={np.nanmin(dt):.4f}s max={np.nanmax(dt):.4f}s")
    breaks = find_break_indices_sec(t_sec)
    print(f"[{label}] {len(breaks)} break(s) detected at row indices: "
          f"{breaks[:20].tolist()}{' ...' if len(breaks) > 20 else ''}")
    return breaks


def split_by_breaks(n_rows: int, breaks: np.ndarray):
    """Yield (start_idx, end_idx) row ranges for each contiguous clean segment."""
    bounds = [0] + list(breaks) + [n_rows]
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        if end - start < 2:
            continue
        yield start, end


def process(imu_path: str, gps_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    imu_raw = load_raw_csv(imu_path)  # handles real-data encoding (cp1252) + strips
    gps_raw = load_raw_csv(gps_path)  # whitespace from headers

    # --- Wall-clock time basis (replaces the buggy ms counter) ---
    imu_tod_sec = s_time_of_day_sec(imu_raw)
    gps_tod_sec = gps_raw[RAW_GPS_TIME_COL].to_numpy(dtype=float)

    # V-M.csv can repeat the same timestamp on consecutive rows (Person A's
    # flagged open issue). Dedupe (keep last) before any interpolation/join
    # so downstream np.interp never sees non-increasing x-values.
    gps_order = np.argsort(gps_tod_sec, kind="stable")
    gps_tod_sec_sorted = gps_tod_sec[gps_order]
    gps_raw_sorted = gps_raw.iloc[gps_order].reset_index(drop=True)
    keep_last_mask = np.r_[np.diff(gps_tod_sec_sorted) != 0, True]
    gps_tod_sec = gps_tod_sec_sorted[keep_last_mask]
    gps_raw = gps_raw_sorted.iloc[keep_last_mask].reset_index(drop=True)

    print("=== Diagnostics before splitting (wall-clock basis) ===")
    imu_breaks = diagnose_sec(imu_tod_sec, "IMU")

    # Confirmed offset to ADD to V's time-of-day so it overlaps S's range.
    offset, overlap = find_time_offset(imu_tod_sec, gps_tod_sec)
    print(f"[GPS] confirmed clock offset: +{offset}s (overlap={overlap:.1f}s)")
    gps_aligned_sec = gps_tod_sec + offset

    imu_segments = list(split_by_breaks(len(imu_raw), imu_breaks))

    manifest_rows = []
    session_id = 0

    for imu_start, imu_end in imu_segments:
        seg = imu_raw.iloc[imu_start:imu_end].copy()
        seg_tod = imu_tod_sec[imu_start:imu_end]
        t_sec = seg_tod - seg_tod[0]
        duration = t_sec[-1] - t_sec[0]

        if duration < MIN_SESSION_SEC:
            continue

        # Re-verify monotonicity post-split -- should always pass now, but
        # check anyway so a silent bug upstream can't slip through.
        assert np.all(np.diff(t_sec) > 0), (
            f"Session {session_id} still non-monotonic after split -- "
            f"break detection missed something, investigate before proceeding."
        )

        # Gravity compensation: raw ACCELEROMETER X/Y includes gravity.
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

        # --- Time-window GPS pairing (replaces order-based pairing) ---
        seg_start_time, seg_end_time = seg_tod[0], seg_tod[-1]
        gps_mask = (gps_aligned_sec >= seg_start_time) & (gps_aligned_sec <= seg_end_time)
        n_gps_in_window = int(gps_mask.sum())

        if n_gps_in_window < 2:
            print(f"WARNING: session {session_id} has only {n_gps_in_window} "
                  f"GPS fix(es) in its time window -- skipping (need >= 2 for "
                  f"ground truth).")
            continue

        gps_seg = gps_raw.iloc[gps_mask].copy()
        gps_seg_t_sec = gps_aligned_sec[gps_mask] - seg_start_time

        gps_out = pd.DataFrame({
            "t_sec": gps_seg_t_sec,
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
            "gps_clock_offset_sec": offset,
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
        print("WARNING: no sessions survived filtering -- check thresholds in "
              "common.py or your raw data.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--imu", required=True, help="path to raw S-*.csv")
    ap.add_argument("--gps", required=True, help="path to raw V-*.csv")
    ap.add_argument("--out", default="data/processed/sessions")
    args = ap.parse_args()
    process(args.imu, args.gps, args.out)
