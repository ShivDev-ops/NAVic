"""
Run from your NAVic repo root:
    python check_imu_properties.py

Checks two things that affect Stage B decisions regardless of what
Person A's split_sessions.py ends up doing:

1. Real IMU sample rate (synthetic data assumed 50Hz -- is that right?)
2. Whether ACCELEROMETER X/Y/Z already has gravity removed, or needs
   compensation using the separate GRAVITY X/Y/Z columns before it's
   usable for dead reckoning.
"""
import numpy as np
import pandas as pd

S_PATH = "data/raw/S-M.csv"

s = pd.read_csv(S_PATH, encoding="cp1252")
s.columns = s.columns.str.strip()

# --- 1. Sample rate ---
t_ms = s["TIME SINCE START (ms)"].to_numpy()
t_sec = t_ms / 1000.0
dt = np.diff(t_sec)

# Use the median of "normal" dt values (exclude corruption breaks: negative
# or huge gaps) to avoid the concatenated-session jumps skewing this.
normal_dt = dt[(dt > 0) & (dt < 1.0)]
print("=== Sample rate ===")
print(f"Total rows: {len(s)}, total raw duration: {t_sec[-1] - t_sec[0]:.1f}s "
      f"(includes corruption jumps, not meaningful alone)")
print(f"Median dt (excluding breaks): {np.median(normal_dt)*1000:.2f}ms "
      f"-> ~{1/np.median(normal_dt):.1f} Hz")
print(f"Mean dt (excluding breaks):   {np.mean(normal_dt)*1000:.2f}ms "
      f"-> ~{1/np.mean(normal_dt):.1f} Hz")

# --- 2. Gravity check ---
acc_cols = ["ACCELEROMETER X (m/s²)", "ACCELEROMETER Y (m/s²)", "ACCELEROMETER Z (m/s²)"]
grav_cols = ["GRAVITY X (m/s²)", "GRAVITY Y (m/s²)", "GRAVITY Z (m/s²)"]

acc = s[acc_cols].to_numpy()
grav = s[grav_cols].to_numpy()

acc_mag = np.linalg.norm(acc, axis=1)
grav_mag = np.linalg.norm(grav, axis=1)
linear_acc = acc - grav  # what acc would be if gravity is subtracted
linear_mag = np.linalg.norm(linear_acc, axis=1)

print("\n=== Gravity check ===")
print(f"Mean |accelerometer| magnitude: {acc_mag.mean():.3f} m/s^2 "
      f"(9.81 would suggest gravity IS included / vehicle mostly stationary in Z)")
print(f"Mean |gravity| magnitude:       {grav_mag.mean():.3f} m/s^2 (should be ~9.81)")
print(f"Mean |accelerometer - gravity|: {linear_mag.mean():.3f} m/s^2 "
      f"(this is what 'linear acceleration' would look like)")
print("\nInterpretation:")
print("  If mean |accelerometer| is close to 9.81 and mean |acc - gravity| is much")
print("  smaller (closer to typical driving accel, ~0-3 m/s^2), ACCELEROMETER")
print("  columns include gravity -- Stage B needs to subtract GRAVITY columns")
print("  before using acc_x/acc_y for speed/heading integration or ZUPT.")
print("  If mean |accelerometer| is already small (~0-3), gravity is likely")
print("  already removed and no extra step is needed.")
