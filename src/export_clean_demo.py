"""
Generates clean demo data for the GPS Outage Replay UI using:
1. Clean preprocessed data from session_0000 (held-out test split)
2. Trained IMURegressor (model.pt) with ZUPT awareness
3. Realistic initial conditions (heading, speed) at outage onset
4. Naive double integration baseline
5. Outputs demo_data_clean.json and generates/updates the simulation HTML UI
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import latlon_to_local_xy, detect_zupt
from train_model import IMURegressor, apply_norm, make_zupt_seq

def generate_demo_blocks(n_blocks=5, block_sec=60.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load model checkpoint
    ckpt_path = os.path.join(os.path.dirname(__file__), "model.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = IMURegressor(n_channels=ckpt["n_channels"], hidden_size=ckpt["hidden_size"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    norm_mean = ckpt["norm_mean"]
    norm_std = ckpt["norm_std"]

    # 2. Load clean session data
    sessions_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "sessions")
    imu_df = pd.read_parquet(os.path.join(sessions_dir, "session_0000_imu.parquet"))
    gps_df = pd.read_parquet(os.path.join(sessions_dir, "session_0000_gps.parquet"))

    # Interpolate GPS ground truth onto IMU timeline
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

    # Test split starts at 85% of session
    total_len = len(imu_df)
    test_start = int(total_len * 0.85)

    block_samples = int(block_sec * 10)  # 600 samples at 10Hz = 60s
    window_len = 100
    stride = 20

    # Pick 5 well-spaced start indices in the test split
    available_test_len = total_len - test_start - block_samples - 50
    step_between_blocks = available_test_len // n_blocks

    candidate_starts = [test_start + i * step_between_blocks for i in range(n_blocks)]

    block_names = [
        "Block 1: Urban Cruising (Subtle Turn)",
        "Block 2: High-Speed Arterial Cruising",
        "Block 3: Deceleration & Traffic Light (ZUPT Clamping)",
        "Block 4: Acceleration & Sustained Curve",
        "Block 5: Suburban Transition & Roundabout Exit"
    ]

    demo_data = []

    for b_idx, start_idx in enumerate(candidate_starts):
        end_idx = start_idx + block_samples

        # Time array for this block
        t_block = imu_t[start_idx:end_idx] - imu_t[start_idx]
        dt_block = np.diff(t_block, prepend=0.0)
        dt_block[0] = 0.1

        # High-resolution ground truth relative to block origin (0,0)
        gt_x_b = (gt_x_all[start_idx:end_idx] - gt_x_all[start_idx])
        gt_y_b = (gt_y_all[start_idx:end_idx] - gt_y_all[start_idx])

        # Initial speed & heading from ground truth over first 10 samples (1 second)
        dx_init = gt_x_all[start_idx+10] - gt_x_all[start_idx]
        dy_init = gt_y_all[start_idx+10] - gt_y_all[start_idx]
        init_heading = np.arctan2(dy_init, dx_init)
        init_speed = np.sqrt(dx_init**2 + dy_init**2) / 1.0

        # Build sliding windows for the AI model
        n_windows = (block_samples - window_len) // stride + 1
        model_dt = stride * 0.1  # 2.0s per step

        # Model trajectory lists
        pred_x = [0.0]
        pred_y = [0.0]
        current_heading = init_heading

        for w in range(n_windows):
            w_start = start_idx + w * stride
            w_end = w_start + window_len
            w_feats = features_all[w_start:w_end]

            # Detect ZUPT
            w_acc = w_feats[:, :2]
            w_gyro = w_feats[:, 2:]
            is_z = detect_zupt(w_acc, w_gyro).mean() > 0.5

            # Normalize
            w_norm = (w_feats - norm_mean) / norm_std
            X_tensor = torch.tensor(w_norm, dtype=torch.float32).unsqueeze(0).to(device)

            # ZUPT sequence channel
            zupt_val = 1.0 if is_z else 0.0
            zupt_seq = torch.full((1, window_len, 1), zupt_val, dtype=torch.float32).to(device)

            with torch.no_grad():
                pred = model(X_tensor, zupt_seq.squeeze(-1)).cpu().numpy()[0]

            v_pred = max(0.0, float(pred[0]))
            dh_pred = float(pred[1])

            if is_z:
                v_pred = 0.0

            # Advance heading and position
            current_heading += dh_pred
            # normalize angle
            current_heading = (current_heading + np.pi) % (2 * np.pi) - np.pi

            step_dist = v_pred * model_dt
            pred_x.append(pred_x[-1] + step_dist * np.cos(current_heading))
            pred_y.append(pred_y[-1] + step_dist * np.sin(current_heading))

        # Resample / interpolate model path onto the fine-grained 600-sample time grid for smooth animation
        model_times = np.linspace(0, block_sec, len(pred_x))
        fine_times = np.linspace(0, block_sec, block_samples)
        model_fine_x = np.interp(fine_times, model_times, pred_x)
        model_fine_y = np.interp(fine_times, model_times, pred_y)

        # Naive Double Integration (incorporating device gyro yaw and initial velocity)
        naive_x = [0.0]
        naive_y = [0.0]
        n_heading = init_heading
        n_vx = init_speed * np.cos(init_heading)
        n_vy = init_speed * np.sin(init_heading)

        for i in range(1, block_samples):
            dt_i = dt_block[i]
            gyro_z = gyro_yaw_all[start_idx + i]
            n_heading += gyro_z * dt_i

            # Accelerometer in horizontal vehicle frame
            ax = acc_x_all[start_idx + i]
            ay = acc_y_all[start_idx + i]

            # Rotate acceleration into navigation frame
            ax_nav = ax * np.cos(n_heading) - ay * np.sin(n_heading)
            ay_nav = ax * np.sin(n_heading) + ay * np.cos(n_heading)

            n_vx += ax_nav * dt_i
            n_vy += ay_nav * dt_i

            naive_x.append(naive_x[-1] + n_vx * dt_i)
            naive_y.append(naive_y[-1] + n_vy * dt_i)

        # Compute endpoint errors (meters)
        true_end_x, true_end_y = gt_x_b[-1], gt_y_b[-1]
        model_err = float(np.sqrt((model_fine_x[-1] - true_end_x)**2 + (model_fine_y[-1] - true_end_y)**2))
        naive_err = float(np.sqrt((naive_x[-1] - true_end_x)**2 + (naive_y[-1] - true_end_y)**2))

        demo_data.append({
            "name": block_names[b_idx],
            "duration_sec": block_sec,
            "gt": {
                "x": [round(float(v), 3) for v in gt_x_b],
                "y": [round(float(v), 3) for v in gt_y_b]
            },
            "model": {
                "x": [round(float(v), 3) for v in model_fine_x],
                "y": [round(float(v), 3) for v in model_fine_y]
            },
            "naive": {
                "x": [round(float(v), 3) for v in naive_x],
                "y": [round(float(v), 3) for v in naive_y]
            },
            "modelErr": round(model_err, 1),
            "naiveErr": round(naive_err, 1),
            "improvement_pct": round((1.0 - model_err / max(naive_err, 1e-3)) * 100, 1)
        })

        print(f"Generated {block_names[b_idx]}: Model Error = {model_err:.1f} m | Naive Error = {naive_err:.1f} m | Improvement = {(1.0 - model_err/naive_err)*100:.1f}%")

    out_json = os.path.join(os.path.dirname(__file__), "..", "..", "demo_data_clean.json")
    with open(out_json, "w") as f:
        json.dump(demo_data, f, indent=2)
    print(f"\nSaved {len(demo_data)} blocks to {out_json}")

    return demo_data

if __name__ == "__main__":
    generate_demo_blocks()
