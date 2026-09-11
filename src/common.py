"""
Shared utilities for the IDR preprocessing pipeline.

IMPORTANT: The column-name constants below are placeholders based on the
project status report's description of the raw IO-VNBD files. Update them
to match the ACTUAL headers in your S-*.csv / V-*.csv files before running
anything against real data. Both preprocessing stages import from here so
you only need to fix column names in one place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Raw column names (IO-VNBD "M (Driver B)" category) — VERIFY against your
# actual CSV headers, these are best-guess based on the status report.
# ---------------------------------------------------------------------------
RAW_TIME_COL = "TIME SINCE START (ms)"
RAW_IMU_COLS = {
    "acc_x": "acc_x",
    "acc_y": "acc_y",
    "gyro_yaw": "gyro_yaw",
    "gyro_pitch": "gyro_pitch",
    "gyro_roll": "gyro_roll",
}
RAW_GPS_TIME_COL = "TIME SINCE START (ms)"
RAW_GPS_LAT_COL = "lat"
RAW_GPS_LON_COL = "lon"
RAW_GPS_SPEED_COL = "speed"  # set to None if not present in your V-*.csv

# ---------------------------------------------------------------------------
# Session-splitting thresholds — tune once you see find_timestamp_breaks
# style diagnostics on your real data.
# ---------------------------------------------------------------------------
MIN_SESSION_SEC = 5.0        # drop sessions shorter than this
BREAK_NEGATIVE_DT = 0.0      # any dt <= this is a hard break (time went backwards)
BREAK_GAP_SECONDS = 2.0      # a forward gap bigger than this also starts a new session


def compute_dt(t_ms: np.ndarray) -> np.ndarray:
    """dt in seconds between consecutive raw timestamps (ms). First entry is NaN."""
    t_sec = t_ms / 1000.0
    dt = np.empty_like(t_sec)
    dt[0] = np.nan
    dt[1:] = np.diff(t_sec)
    return dt


def find_break_indices(t_ms: np.ndarray) -> np.ndarray:
    """
    Returns the row indices where a NEW session should start (i.e. the break
    is between index-1 and index). Index 0 is always implicitly a session start.
    A break is either:
      - dt <= BREAK_NEGATIVE_DT (time went backwards / stalled — session splice)
      - dt >  BREAK_GAP_SECONDS (suspiciously large forward gap)
    """
    dt = compute_dt(t_ms)
    is_break = (dt <= BREAK_NEGATIVE_DT) | (dt > BREAK_GAP_SECONDS)
    is_break[0] = False  # dt[0] is NaN, not a real break
    return np.nonzero(is_break)[0]


def latlon_to_local_xy(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float):
    """
    Simple equirectangular projection to local ENU-ish meters, good enough for
    short sessions (a few km). Swap for pyproj if you need better accuracy
    over longer distances.
    """
    R = 6371000.0
    lat0_rad = np.radians(lat0)
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    x = dlon * R * np.cos(lat0_rad)  # east
    y = dlat * R                     # north
    return x, y


def xy_to_speed_heading(t_sec: np.ndarray, x: np.ndarray, y: np.ndarray):
    """
    Convert a position track to (speed, delta_heading) per step.
    speed: m/s, forward-difference.
    delta_heading: change in heading (radians) between consecutive steps,
    wrapped to [-pi, pi].
    Returns arrays of length len(t)-1, aligned to the SECOND point of each pair.
    """
    dt = np.diff(t_sec)
    dx = np.diff(x)
    dy = np.diff(y)
    dt_safe = np.where(dt <= 0, np.nan, dt)

    speed = np.sqrt(dx**2 + dy**2) / dt_safe
    heading = np.arctan2(dy, dx)
    delta_heading = np.diff(heading, prepend=heading[0])
    delta_heading = (delta_heading + np.pi) % (2 * np.pi) - np.pi

    return speed, delta_heading


def reconstruct_xy(speed: np.ndarray, delta_heading: np.ndarray, dt: np.ndarray,
                    start_x: float = 0.0, start_y: float = 0.0,
                    start_heading: float | None = None):
    """
    Inverse of xy_to_speed_heading: integrate (speed, delta_heading) back into
    a trajectory. start_heading is REQUIRED (no silent default) — the old
    pipeline had a bug where start_heading defaulted to 0.0 and silently
    rotated reconstructed trajectories. Don't repeat that here.
    """
    if start_heading is None:
        raise ValueError(
            "start_heading must be provided explicitly. A silent 0.0 default "
            "previously caused silently-rotated reconstructed trajectories."
        )

    n = len(speed)
    x = np.empty(n + 1)
    y = np.empty(n + 1)
    x[0], y[0] = start_x, start_y
    heading = start_heading

    for i in range(n):
        heading += delta_heading[i]
        step = speed[i] * dt[i]
        x[i + 1] = x[i] + step * np.cos(heading)
        y[i + 1] = y[i] + step * np.sin(heading)

    return x, y


def detect_zupt(acc: np.ndarray, gyro: np.ndarray, acc_thresh: float = 0.3,
                 gyro_thresh: float = 0.05) -> np.ndarray:
    """
    Simple zero-velocity-update detector: flags samples where both
    acceleration and gyro magnitude are below threshold (vehicle likely
    stationary — stopped at a light, in traffic, etc).
    `acc` and `gyro` should each be (N, k) arrays of the relevant channels.
    Thresholds are placeholders — tune against real stationary segments.
    """
    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    return (acc_mag < acc_thresh) & (gyro_mag < gyro_thresh)
