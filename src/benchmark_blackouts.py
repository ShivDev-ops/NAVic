"""
Comprehensive GNSS Outage Benchmark:
Evaluates the trained IDR model against traditional Dead Reckoning baselines
across realistic simulated GNSS outage durations (15s, 30s, 60s).

Consumes:
  - model.pt: Trained IMURegressor checkpoint
  - test.npz: Held-out test split
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import reconstruct_xy, detect_zupt
from train_model import IMURegressor, apply_norm, make_zupt_seq, load_split


def evaluate_blackout_blocks(model, X_test, y_true, is_zupt, mean, std, device,
                             stride_sec: float = 2.0,
                             durations=(15, 30, 60)):
    """
    Evaluates drift across multiple simulated GNSS blackouts.
    stride_sec: time interval between consecutive window steps (stride=20 at 10Hz -> 2.0s).
    """
    model.eval()
    X_norm = apply_norm(X_test, mean, std)
    zupt_seq = make_zupt_seq(X_test.shape, is_zupt)

    with torch.no_grad():
        Xt = torch.from_numpy(X_norm).float().to(device)
        zt = torch.from_numpy(zupt_seq).float().to(device)
        raw_preds = model(Xt, zt).cpu().numpy()

    # Model predictions with ZUPT applied
    preds_zupt = raw_preds.copy()
    preds_zupt[is_zupt, 0] = 0.0  # zero speed when stationary

    # Model predictions without ZUPT (pure raw GRU output)
    preds_no_zupt = raw_preds.copy()

    total_windows = len(X_test)
    benchmark_results = {}

    for duration in durations:
        n_steps = max(1, int(round(duration / stride_sec)))
        dt_seq = np.full(n_steps, stride_sec)

        model_zupt_errors = []
        model_raw_errors = []
        naive_errors = []
        const_vel_errors = []

        # Slide through test set in non-overlapping blackout episodes
        for start_idx in range(0, total_windows - n_steps, n_steps):
            end_idx = start_idx + n_steps

            # Ground truth
            v_true = y_true[start_idx:end_idx, 0]
            dh_true = y_true[start_idx:end_idx, 1]
            gt_x, gt_y = reconstruct_xy(v_true, dh_true, dt_seq, start_x=0.0, start_y=0.0, start_heading=0.0)
            target_x, target_y = gt_x[-1], gt_y[-1]

            # 1. AI Model + ZUPT
            v_model_z = preds_zupt[start_idx:end_idx, 0]
            dh_model_z = preds_zupt[start_idx:end_idx, 1]
            mx_z, my_z = reconstruct_xy(v_model_z, dh_model_z, dt_seq, start_x=0.0, start_y=0.0, start_heading=0.0)
            err_zupt = np.sqrt((mx_z[-1] - target_x)**2 + (my_z[-1] - target_y)**2)
            model_zupt_errors.append(float(err_zupt))

            # 2. AI Model without ZUPT
            v_model_raw = preds_no_zupt[start_idx:end_idx, 0]
            dh_model_raw = preds_no_zupt[start_idx:end_idx, 1]
            mx_r, my_r = reconstruct_xy(v_model_raw, dh_model_raw, dt_seq, start_x=0.0, start_y=0.0, start_heading=0.0)
            err_raw = np.sqrt((mx_r[-1] - target_x)**2 + (my_r[-1] - target_y)**2)
            model_raw_errors.append(float(err_raw))

            # 3. Constant Velocity Baseline (holds speed & heading from pre-outage window)
            v_const = np.full(n_steps, v_true[0])
            dh_const = np.zeros(n_steps)
            cx, cy = reconstruct_xy(v_const, dh_const, dt_seq, start_x=0.0, start_y=0.0, start_heading=0.0)
            err_const = np.sqrt((cx[-1] - target_x)**2 + (cy[-1] - target_y)**2)
            const_vel_errors.append(float(err_const))

            # 4. Naive Double Integration of linear accelerometer & gyro
            # Approximated by integrating raw linear acceleration across windows
            window_raw_acc_x = X_test[start_idx:end_idx, :, 0].mean(axis=1)
            window_raw_acc_y = X_test[start_idx:end_idx, :, 1].mean(axis=1)
            window_raw_yaw = X_test[start_idx:end_idx, :, 2].mean(axis=1)
            
            # Simple dead reckoning step from initial velocity
            naive_vx = np.cumsum(window_raw_acc_x * stride_sec) + v_true[0]
            naive_vy = np.cumsum(window_raw_acc_y * stride_sec)
            naive_x = np.cumsum(naive_vx * stride_sec)[-1]
            naive_y = np.cumsum(naive_vy * stride_sec)[-1]
            err_naive = np.sqrt((naive_x - target_x)**2 + (naive_y - target_y)**2)
            naive_errors.append(float(err_naive))

        m_z = np.array(model_zupt_errors)
        m_r = np.array(model_raw_errors)
        c_v = np.array(const_vel_errors)
        n_v = np.array(naive_errors)

        improvement_vs_naive = (1.0 - m_z.mean() / max(n_v.mean(), 1e-6)) * 100.0
        improvement_vs_const = (1.0 - m_z.mean() / max(c_v.mean(), 1e-6)) * 100.0

        benchmark_results[f"{duration}s_outage"] = {
            "n_episodes_tested": len(model_zupt_errors),
            "ai_model_zupt": {
                "mean_error_m": float(np.mean(m_z)),
                "median_error_m": float(np.median(m_z)),
                "p90_error_m": float(np.percentile(m_z, 90)),
                "std_error_m": float(np.std(m_z)),
            },
            "ai_model_raw": {
                "mean_error_m": float(np.mean(m_r)),
                "median_error_m": float(np.median(m_r)),
                "p90_error_m": float(np.percentile(m_r, 90)),
            },
            "constant_velocity": {
                "mean_error_m": float(np.mean(c_v)),
                "median_error_m": float(np.median(c_v)),
                "p90_error_m": float(np.percentile(c_v, 90)),
            },
            "naive_double_integration": {
                "mean_error_m": float(np.mean(n_v)),
                "median_error_m": float(np.median(n_v)),
                "p90_error_m": float(np.percentile(n_v, 90)),
            },
            "error_reduction_vs_naive_pct": float(improvement_vs_naive),
            "error_reduction_vs_const_pct": float(improvement_vs_const),
        }

    return benchmark_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="../data/processed/dataset")
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--out", default="blackout_benchmark_results.json")
    ap.add_argument("--stride-sec", type=float, default=2.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on device: {device}")

    # Load model checkpoint
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model = IMURegressor(n_channels=ckpt["n_channels"], hidden_size=ckpt["hidden_size"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    # Load test split
    test_data = load_split(args.dataset, "test")
    X_test = test_data["X"]
    y_test = test_data["y_speed_heading"]
    is_zupt = test_data["is_zupt"]

    print(f"Loaded {len(X_test)} test windows. Running outage simulations...")
    results = evaluate_blackout_blocks(
        model, X_test, y_test, is_zupt,
        ckpt["norm_mean"], ckpt["norm_std"], device,
        stride_sec=args.stride_sec,
        durations=(15, 30, 60),
    )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved benchmark results to {args.out}")
    print("\n" + "="*70)
    print(f"{'Outage Duration':<16} | {'AI Model (ZUPT)':<16} | {'Constant Vel':<14} | {'Naive Int.':<12} | {'Improvement':<12}")
    print("="*70)
    for dur_key, res in results.items():
        dur_label = dur_key.replace("_outage", "")
        m_err = f"{res['ai_model_zupt']['mean_error_m']:.2f} m"
        c_err = f"{res['constant_velocity']['mean_error_m']:.2f} m"
        n_err = f"{res['naive_double_integration']['mean_error_m']:.2f} m"
        imp = f"{res['error_reduction_vs_naive_pct']:.1f}%"
        print(f"{dur_label:<16} | {m_err:<16} | {c_err:<14} | {n_err:<12} | {imp:<12}")
    print("="*70)


if __name__ == "__main__":
    main()
