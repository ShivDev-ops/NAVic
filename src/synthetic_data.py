"""
Generates fake S-*.csv / V-*.csv files with the SAME shape and SAME kind of
corruption (concatenated-session timestamp breaks) as the real IO-VNBD data.

Use this so Person A and Person B can build and test their halves of the
pipeline against each other's interface *before* wiring up real data, and so
either of you can sanity-check your own stage in isolation.

Usage:
    python synthetic_data.py --out data/raw --n-sessions 4
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def make_one_session(rng, session_len_sec, hz_imu=50, hz_gps=1, lat0=28.6, lon0=77.2):
    n_imu = int(session_len_sec * hz_imu)
    t_imu = np.arange(n_imu) / hz_imu

    # fake a car driving in a gentle curve with speed noise
    speed = 8 + 2 * np.sin(t_imu / 20) + rng.normal(0, 0.3, n_imu)
    heading = np.cumsum(rng.normal(0, 0.01, n_imu))
    acc_x = np.gradient(speed, t_imu) * np.cos(heading) + rng.normal(0, 0.2, n_imu)
    acc_y = np.gradient(speed, t_imu) * np.sin(heading) + rng.normal(0, 0.2, n_imu)
    gyro_yaw = np.gradient(heading, t_imu) + rng.normal(0, 0.02, n_imu)
    gyro_pitch = rng.normal(0, 0.05, n_imu)
    gyro_roll = rng.normal(0, 0.05, n_imu)

    imu_df = pd.DataFrame({
        "TIME SINCE START (ms)": t_imu * 1000,
        "ACCELEROMETER X (m/s²)": acc_x, "ACCELEROMETER Y (m/s²)": acc_y,
        # Synthetic acc_x/acc_y are already "linear" (no gravity mixed in),
        # so GRAVITY X/Y are 0 here -- compensate_gravity() becomes a no-op
        # on synthetic data, matching how it always worked before this was
        # added for real-data gravity removal.
        "GRAVITY X (m/s²)": np.zeros(n_imu), "GRAVITY Y (m/s²)": np.zeros(n_imu),
        "GYROSCOPE Yaw (rad/s)": gyro_yaw, "GYROSCOPE Pitch (rad/s)": gyro_pitch,
        "GYROSCOPE Roll (rad/s)": gyro_roll,
    })

    n_gps = int(session_len_sec * hz_gps)
    t_gps = np.arange(n_gps) / hz_gps
    x = np.interp(t_gps, t_imu, np.cumsum(speed * np.cos(heading)) / hz_imu)
    y = np.interp(t_gps, t_imu, np.cumsum(speed * np.sin(heading)) / hz_imu)
    R = 6371000.0
    lat = lat0 + (y / R) * (180 / np.pi)
    lon = lon0 + (x / (R * np.cos(np.radians(lat0)))) * (180 / np.pi)

    gps_df = pd.DataFrame({
        "Time Since Start of Day (seconds)": t_gps,
        "Latitude (degrees)": lat, "Longitude (degrees)": lon,
        "Velocity (km/hr)": speed[:: hz_imu // hz_gps][:n_gps] * 3.6,
    })

    return imu_df, gps_df


def make_corrupted_file(rng, n_sessions=4, session_len_sec=60):
    """Concatenate several sessions with NO reset between them (the real bug)."""
    imu_parts, gps_parts = [], []
    for _ in range(n_sessions):
        imu_df, gps_df = make_one_session(rng, session_len_sec)
        imu_parts.append(imu_df)
        gps_parts.append(gps_df)
    # This is the corruption: raw concat, timestamps restart from 0 each time
    # instead of continuing, exactly like the real dt=-4426s symptom.
    imu_all = pd.concat(imu_parts, ignore_index=True)
    gps_all = pd.concat(gps_parts, ignore_index=True)
    return imu_all, gps_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--n-sessions", type=int, default=4)
    ap.add_argument("--session-len-sec", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    imu_df, gps_df = make_corrupted_file(rng, args.n_sessions, args.session_len_sec)

    imu_path = os.path.join(args.out, "S-M.csv")
    gps_path = os.path.join(args.out, "V-M.csv")
    # Write with cp1252 to match how load_raw_csv() reads real IO-VNBD files --
    # otherwise the "²" in column headers round-trips as mojibake and key
    # lookups in split_sessions.py fail.
    imu_df.to_csv(imu_path, index=False, encoding="cp1252")
    gps_df.to_csv(gps_path, index=False, encoding="cp1252")
    print(f"Wrote {len(imu_df)} IMU rows -> {imu_path}")
    print(f"Wrote {len(gps_df)} GPS rows -> {gps_path}")
    print(f"Contains {args.n_sessions} concatenated sessions with no timestamp reset "
          f"(simulates the real corruption).")


if __name__ == "__main__":
    main()
