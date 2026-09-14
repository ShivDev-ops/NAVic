"""
Evaluates the trained DisplacementNet on the held-out test split,
generates the 5 realistic 60s blackout blocks, computes trajectory errors,
and exports demo_data_clean.json and updates the simulation UI.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import latlon_to_local_xy

class DisplacementNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

def run():
    device = torch.device("cpu")

    # 1. Load trained model
    ckpt_path = os.path.join(os.path.dirname(__file__), "model_displacement.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model = DisplacementNet(input_size=ckpt["input_size"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    X_mean = ckpt["X_mean"]
    X_std = ckpt["X_std"]
    Y_mean = ckpt["Y_mean"]
    Y_std = ckpt["Y_std"]
    WINDOW = ckpt["window"]

    # 2. Load clean session data
    sessions_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "sessions")
    imu_df = pd.read_parquet(os.path.join(sessions_dir, "session_0000_imu.parquet"))
    gps_df = pd.read_parquet(os.path.join(sessions_dir, "session_0000_gps.parquet"))

    lat0, lon0 = gps_df["lat"].iloc[0], gps_df["lon"].iloc[0]
    gx, gy = latlon_to_local_xy(gps_df["lat"].to_numpy(), gps_df["lon"].to_numpy(), lat0, lon0)

    imu_t = imu_df["t_sec"].to_numpy()
    gt_x_all = np.interp(imu_t, gps_df["t_sec"], gx)
    gt_y_all = np.interp(imu_t, gps_df["t_sec"], gy)

    acc_x_all = imu_df["acc_x"].to_numpy()
    acc_y_all = imu_df["acc_y"].to_numpy()
    gyro_yaw_all = imu_df["gyro_yaw"].to_numpy()
    gyro_pitch_all = imu_df["gyro_pitch"].to_numpy()
    gyro_roll_all = imu_df["gyro_roll"].to_numpy()

    features_all = np.stack([acc_x_all, acc_y_all, gyro_yaw_all, gyro_pitch_all, gyro_roll_all], axis=1)

    N = len(imu_df)
    split_idx = int(N * 0.8)
    BLACKOUT_SAMPLES = 600  # 60 seconds at 10Hz

    block_starts = list(range(split_idx, N - BLACKOUT_SAMPLES, BLACKOUT_SAMPLES))[:5]

    block_names = [
        "Block 1: Urban Arterial Outage (60s)",
        "Block 2: High-Speed Curve & Straight (60s)",
        "Block 3: Deceleration & Highway Transition (60s)",
        "Block 4: Multi-Lane Intersection Maneuver (60s)",
        "Block 5: Suburban Cruising & Speed Variation (60s)"
    ]

    demo_blocks = []

    print("\n=== Generating 5 60-second Blackout Blocks on Held-out Data ===")
    for b_idx, b_start in enumerate(block_starts):
        b_end = b_start + BLACKOUT_SAMPLES

        # Ground truth at full 10Hz resolution
        gt_x_b = (gt_x_all[b_start:b_end+1] - gt_x_all[b_start])
        gt_y_b = (gt_y_all[b_start:b_end+1] - gt_y_all[b_start])

        # Model trajectory: 30 consecutive non-overlapping windows of 20 samples (2s each)
        pred_x = [0.0]
        pred_y = [0.0]
        n_windows = BLACKOUT_SAMPLES // WINDOW  # 30 windows

        for w in range(n_windows):
            i = b_start + w * WINDOW
            w_feats = features_all[i:i+WINDOW].flatten()
            w_norm = (w_feats - X_mean) / X_std

            with torch.no_grad():
                pred_norm = model(torch.tensor(w_norm, dtype=torch.float32).unsqueeze(0)).numpy()[0]
            dx, dy = pred_norm * Y_std + Y_mean
            pred_x.append(pred_x[-1] + dx)
            pred_y.append(pred_y[-1] + dy)

        # Naive dead reckoning at full 10Hz resolution
        block_dt = np.diff(imu_t[b_start:b_end], prepend=imu_t[b_start])
        block_dt[0] = 0.1
        block_ax = acc_x_all[b_start:b_end]
        block_ay = acc_y_all[b_start:b_end]

        vel_x = np.cumsum(block_ax * block_dt)
        vel_y = np.cumsum(block_ay * block_dt)
        naive_traj_x = np.concatenate([[0.0], np.cumsum(vel_x * block_dt)])
        naive_traj_y = np.concatenate([[0.0], np.cumsum(vel_y * block_dt)])

        true_dx = gt_x_all[b_end] - gt_x_all[b_start]
        true_dy = gt_y_all[b_end] - gt_y_all[b_start]

        model_err = float(np.sqrt((pred_x[-1] - true_dx)**2 + (pred_y[-1] - true_dy)**2))
        naive_err = float(np.sqrt((naive_traj_x[-1] - true_dx)**2 + (naive_traj_y[-1] - true_dy)**2))
        imp_pct = float((1.0 - model_err / naive_err) * 100.0)

        demo_blocks.append({
            "name": block_names[b_idx],
            "duration_sec": 60.0,
            "gt": {
                "x": [round(float(v), 3) for v in gt_x_b],
                "y": [round(float(v), 3) for v in gt_y_b]
            },
            "model": {
                "x": [round(float(v), 3) for v in pred_x],
                "y": [round(float(v), 3) for v in pred_y]
            },
            "naive": {
                "x": [round(float(v), 3) for v in naive_traj_x],
                "y": [round(float(v), 3) for v in naive_traj_y]
            },
            "modelErr": round(model_err, 1),
            "naiveErr": round(naive_err, 1),
            "improvement_pct": round(imp_pct, 1)
        })

        print(f"{block_names[b_idx]}:")
        print(f"  AI Model Error:  {model_err:.1f} m")
        print(f"  Naive DR Error:  {naive_err:.1f} m")
        print(f"  Improvement:     {imp_pct:.1f}% lower error\n")

    # Save to demo_data_clean.json
    out_json = os.path.join(os.path.dirname(__file__), "..", "..", "demo_data_clean.json")
    with open(out_json, "w") as f:
        json.dump(demo_blocks, f, indent=2)
    print(f"Saved demo_data_clean.json to {out_json}")

    # Now rebuild the HTML files
    from build_simulation_html import build_html
    build_html()
    print("Simulation HTML rebuilt successfully!")

if __name__ == "__main__":
    run()
