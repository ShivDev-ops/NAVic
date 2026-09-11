"""
Run this from inside your NAVic repo (adjust the paths if needed):
    python check_sync.py

Checks:
1. Exact (stripped) column names in S-M.csv and V-M.csv, for updating common.py
2. Whether S's DATE-derived time-of-day and V's "Time Since Start of Day"
   are on the same clock, offset by a constant, or unrelated
3. Whether S's own embedded GPS columns look usable as ground truth on their own
"""
import pandas as pd
import numpy as np

S_PATH = "data/raw/S-M.csv"
V_PATH = "data/raw/V-M.csv"

s = pd.read_csv(S_PATH, encoding="cp1252")
v = pd.read_csv(V_PATH, encoding="cp1252")
s.columns = s.columns.str.strip()
v.columns = v.columns.str.strip()

print("=== S-M.csv columns (stripped) ===")
for c in s.columns:
    print(" ", repr(c))
print("\n=== V-M.csv columns (stripped) ===")
for c in v.columns:
    print(" ", repr(c))

# --- Clock sync check ---
date_col = "DATE (YYYY-MO-DD HH-MI-SS_SSS)"
s_dt = pd.to_datetime(s[date_col], format="%Y-%m-%d %H:%M:%S:%f")
s_tod_sec = (s_dt - s_dt.dt.normalize()).dt.total_seconds()

v_tod_col = "Time Since Start of Day (seconds)"
v_tod_sec = v[v_tod_col]

print("\n=== Clock comparison ===")
print(f"S time-of-day range: {s_tod_sec.min():.1f}s to {s_tod_sec.max():.1f}s "
      f"({s_tod_sec.min()/3600:.2f}h to {s_tod_sec.max()/3600:.2f}h)")
print(f"V time-of-day range: {v_tod_sec.min():.1f}s to {v_tod_sec.max():.1f}s "
      f"({v_tod_sec.min()/3600:.2f}h to {v_tod_sec.max()/3600:.2f}h)")

overlap_start = max(s_tod_sec.min(), v_tod_sec.min())
overlap_end = min(s_tod_sec.max(), v_tod_sec.max())
print(f"Direct overlap (no offset correction): {max(0, overlap_end - overlap_start):.1f}s")

# Test a few candidate offsets (in seconds) to see if any produces a big overlap
for offset in [0, 3600, -3600, 3600*2, -3600*2]:
    shifted_start = max(s_tod_sec.min(), v_tod_sec.min() + offset)
    shifted_end = min(s_tod_sec.max(), v_tod_sec.max() + offset)
    ov = max(0, shifted_end - shifted_start)
    print(f"  if V is offset by {offset:+d}s: overlap = {ov:.1f}s")

# --- S's own embedded GPS sanity check ---
print("\n=== S-M.csv's own embedded GPS (potential ground truth without needing V at all) ===")
lat_col = "GPS LATITUDE (degrees)"
lon_col = "GPS LONGITUDE (degrees)"
print(f"lat range: {s[lat_col].min():.6f} to {s[lat_col].max():.6f}")
print(f"lon range: {s[lon_col].min():.6f} to {s[lon_col].max():.6f}")
print(f"unique (lat,lon) pairs: {s[[lat_col, lon_col]].drop_duplicates().shape[0]} "
      f"out of {len(s)} rows (low uniqueness = GPS updates slower than IMU, expected)")
