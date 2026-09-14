"""
Shared utilities for the IDR preprocessing pipeline.

Column-name constants were VERIFIED against real IO-VNBD headers (S-M.csv /
V-M.csv) on 2026-09-12 -- see the comments above each block. If you're
running against a different driver/category file, spot-check the headers
still match before trusting the constants below.

2026-09-14 update: TIME SINCE START (ms) is a confirmed-buggy internal
counter (found a -4426s jump corresponding to ~1.16s of real elapsed time).
Break detection AND per-sample relative time (`t_sec`) should be derived
from the wall-clock DATE column instead -- see find_break_indices_sec()
and s_time_of_day_sec(). RAW_TIME_COL / compute_dt() / find_break_indices()
are kept below for backward compatibility (e.g. synthetic-data testing that
still exercises the old corruption model) but split_sessions.py no longer
calls them for real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Raw column names (IO-VNBD "M (Driver B)" category) -- verified against real
# CSV headers on 2026-09-12.
# ---------------------------------------------------------------------------
RAW_ENCODING = "cp1252"  # confirmed: plain utf-8 raises UnicodeDecodeError on 0xB2 (Â°)
RAW_TIME_COL = "TIME SINCE START (ms)"  # BUGGY -- do not use for real breaks/t_sec
RAW_DATE_COL = "DATE (YYYY-MO-DD HH-MI-SS_SSS)"  # absolute wall-clock timestamp per row
RAW_IMU_COLS = {
    "acc_x": "ACCELEROMETER X (m/s²)",
    "acc_y": "ACCELEROMETER Y (m/s²)",
    "gyro_yaw": "GYROSCOPE Yaw (rad/s)",
    "gyro_pitch": "GYROSCOPE Pitch (rad/s)",
    "gyro_roll": "GYROSCOPE Roll (rad/s)",
}

RAW_GRAVITY_COLS = {
    "grav_x": "GRAVITY X (m/s²)",
    "grav_y": "GRAVITY Y (m/s²)",
}

# S also has its OWN embedded GPS (recorded by the same phone, same clock as
# the IMU -- no sync needed). Useful as a fallback / cross-check.
RAW_S_GPS_LAT_COL = "GPS LATITUDE (degrees)"
RAW_S_GPS_LON_COL = "GPS LONGITUDE (degrees)"
RAW_S_GPS_SPEED_COL = "GPS SPEED (Kmh)"

# --- V-M.csv (vehicle telemetry / OBD file) --- V's time column is
# seconds-since-midnight on a DIFFERENT time reference than S's DATE column
# -- empirically ~3600s apart. Use find_time_offset() per file rather than
# trusting a hardcoded constant.
RAW_GPS_TIME_COL = "Time Since Start of Day (seconds)"
RAW_GPS_LAT_COL = "Latitude (degrees)"
RAW_GPS_LON_COL = "Longitude (degrees)"
RAW_GPS_SPEED_COL = "Velocity (km/hr)"

# ---------------------------------------------------------------------------
# Session-splitting thresholds
# ---------------------------------------------------------------------------
MIN_SESSION_SEC = 5.0  # drop sessions shorter than this
BREAK_NEGATIVE_DT = 0.0  # any dt <= this is a hard break (time went backwards)
BREAK_GAP_SECONDS = 2.0  # a forward gap bigger than this also starts a new session


def compute_dt(t_ms: np.ndarray) -> np.ndarray:
    """dt in seconds between consecutive raw timestamps (ms). First entry is NaN.
    LEGACY -- operates on the buggy TIME SINCE START (ms) counter. Kept for
    synthetic-data / backward-compat testing only."""
    t_sec = t_ms / 1000.0
    dt = np.empty_like(t_sec, dtype=float)
    dt[0] = np.nan
    dt[1:] = np.diff(t_sec)
    return dt


def find_break_indices(t_ms: np.ndarray) -> np.ndarray:
    """LEGACY break detector on the buggy ms counter. Kept for synthetic-data
    testing only -- real pipelines should use find_break_indices_sec()."""
    dt = compute_dt(t_ms)
    is_break = (dt <= BREAK_NEGATIVE_DT) | (dt > BREAK_GAP_SECONDS)
    is_break[0] = False
    return np.nonzero(is_break)[0]


def compute_dt_sec(t_sec: np.ndarray) -> np.ndarray:
    """dt in seconds between consecutive timestamps that are ALREADY in
    seconds (e.g. wall-clock time-of-day). First entry is NaN."""
    dt = np.empty_like(t_sec, dtype=float)
    dt[0] = np.nan
    dt[1:] = np.diff(t_sec)
    return dt


def find_break_indices_sec(t_sec: np.ndarray) -> np.ndarray:
    """
    Break detector for wall-clock-derived time (seconds), the confirmed-
    correct basis per Person A's 2026-09-14 finding. Same break rule as
    find_break_indices(): dt <= BREAK_NEGATIVE_DT (time went backwards) or
    dt > BREAK_GAP_SECONDS (suspicious forward gap).
    """
    dt = compute_dt_sec(t_sec)
    is_break = (dt <= BREAK_NEGATIVE_DT) | (dt > BREAK_GAP_SECONDS)
    is_break[0] = False
    return np.nonzero(is_break)[0]


def compensate_gravity(acc_x: np.ndarray, acc_y: np.ndarray,
                        grav_x: np.ndarray, grav_y: np.ndarray):
    """Subtract device-frame gravity from raw accelerometer readings to get
    linear (driving) acceleration."""
    return acc_x - grav_x, acc_y - grav_y


def load_raw_csv(path: str, encoding: str = RAW_ENCODING) -> pd.DataFrame:
    """Read a raw S-*/V-*.csv and strip whitespace from column names."""
    df = pd.read_csv(path, encoding=encoding)
    df.columns = df.columns.str.strip()
    return df


def s_time_of_day_sec(s_df: pd.DataFrame) -> np.ndarray:
    """Convert S-M.csv's DATE column into seconds-since-midnight, local time
    (as recorded by the phone). This is the confirmed-correct time basis for
    both break detection and per-sample t_sec on real data."""
    dt = pd.to_datetime(s_df[RAW_DATE_COL], format="%Y-%m-%d %H:%M:%S:%f")
    return (dt - dt.dt.normalize()).dt.total_seconds().to_numpy()


def find_time_offset(s_tod_sec: np.ndarray, v_tod_sec: np.ndarray,
                      candidate_range=(-2 * 3600, 2 * 3600), step: int = 1):
    """
    Find the offset (seconds) to ADD to V's time-of-day so its range best
    overlaps S's. Empirically S and V differ by a fixed offset, not
    per-sample drift, so matching overall time-of-day RANGES is enough.
    Run per raw file pair; don't assume +3600 always holds.
    """
    s_min, s_max = float(np.min(s_tod_sec)), float(np.max(s_tod_sec))
    v_min, v_max = float(np.min(v_tod_sec)), float(np.max(v_tod_sec))
    best_offset, best_overlap = 0, -1.0
    for offset in range(candidate_range[0], candidate_range[1] + 1, step):
        start = max(s_min, v_min + offset)
        end = min(s_max, v_max + offset)
        overlap = max(0.0, end - start)
        if overlap > best_overlap:
            best_overlap, best_offset = overlap, offset
    return best_offset, best_overlap


def latlon_to_local_xy(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float):
    """Simple equirectangular projection to local meters, good enough for
    short sessions (a few km)."""
    R = 6371000.0
    lat0_rad = np.radians(lat0)
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    x = dlon * R * np.cos(lat0_rad)  # east
    y = dlat * R  # north
    return x, y


def xy_to_speed_heading(t_sec: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Convert a position track to (speed, delta_heading) per step.
    Returns arrays of length len(t)-1, aligned to the SECOND point of each pair."""
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
    """Inverse of xy_to_speed_heading. start_heading is REQUIRED (no silent
    default) -- a silent 0.0 default previously caused silently-rotated
    reconstructed trajectories."""
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
    """Zero-velocity-update detector: flags samples where both acceleration
    and gyro magnitude are below threshold (vehicle likely stationary)."""
    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    return (acc_mag < acc_thresh) & (gyro_mag < gyro_thresh)
 