"""
Trains a sequence model to predict (speed, delta_heading) per window from
IMU data (acc_x, acc_y, gyro_yaw, gyro_pitch, gyro_roll), then evaluates it
against a naive baseline by reconstructing full trajectories.

Consumes the train/val/test.npz produced by build_dataset.py:
  X:               (N, window_len, 5)  float32
  y_speed_heading: (N, 2)              float32  [speed, delta_heading]
  y_dxdy:          (N, 2)              float32  (unused here, kept for reference)
  is_zupt:         (N,)                bool
  session_id:      (N,)                int32

Section 5 (evaluation): reconstructs (x, y) trajectories from predicted
(speed, delta_heading) via reconstruct_xy() and compares position error
against a naive "assume last known speed/heading" baseline, per the
preprocessing plan's rationale for this label representation.

Usage:
python train_model.py --dataset ../data/processed/dataset_real \
    --out model.pt --epochs 30
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    sys.exit(
        "PyTorch is required for this script but isn't installed.\n"
        "Install it with:\n"
        "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "(CPU-only build is fine for this dataset size -- no GPU needed.)"
    )

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import reconstruct_xy  # noqa: E402


class IMURegressor(nn.Module):
    """Small GRU regressor: (batch, window_len, 5) -> (batch, 2) [speed, delta_heading]."""

    def __init__(self, n_channels: int = 5, hidden_size: int = 64, n_layers: int = 2,
                 use_zupt_feature: bool = True):
        super().__init__()
        in_size = n_channels + (1 if use_zupt_feature else 0)
        self.use_zupt_feature = use_zupt_feature
        self.gru = nn.GRU(in_size, hidden_size, num_layers=n_layers,
                           batch_first=True, dropout=0.1 if n_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x, zupt_seq=None):
        if self.use_zupt_feature and zupt_seq is not None:
            x = torch.cat([x, zupt_seq.unsqueeze(-1)], dim=-1)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)


def load_split(dataset_dir: str, name: str):
    path = os.path.join(dataset_dir, f"{name}.npz")
    d = np.load(path)
    return {k: d[k] for k in d.files}


def compute_norm_stats(X_train: np.ndarray):
    """z-score stats from TRAINING data only -- reused unchanged on val/test."""
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    std[std < 1e-8] = 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean) / std


def make_zupt_seq(X_shape, is_zupt_scalar):
    """Broadcast the per-window ZUPT flag across the window's time steps as
    an auxiliary input channel (simple, matches the plan's step 6 option 2)."""
    n, window_len, _ = X_shape
    seq = np.zeros((n, window_len), dtype=np.float32)
    seq[is_zupt_scalar] = 1.0
    return seq


def build_loader(X, y, zupt_seq, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
        torch.from_numpy(zupt_seq).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batches = 0
    loss_fn = nn.MSELoss()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X, y, zupt_seq in loader:
            X, y, zupt_seq = X.to(device), y.to(device), zupt_seq.to(device)
            pred = model(X, zupt_seq)
            loss = loss_fn(pred, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(1, n_batches)


def evaluate_trajectory(model, X, y_speed_heading, is_zupt, mean, std, device,
                         session_ids, dt_per_window: float):
    """
    Reconstruct trajectories from predicted (speed, delta_heading), apply
    ZUPT correction (force speed=0 when flagged stationary), and compare
    position error against a naive baseline that just repeats the last
    observed (speed, delta_heading) for every window.
    """
    model.eval()
    Xn = apply_norm(X, mean, std)
    zupt_seq = make_zupt_seq(X.shape, is_zupt)

    with torch.no_grad():
        Xt = torch.from_numpy(Xn).float().to(device)
        zt = torch.from_numpy(zupt_seq).float().to(device)
        pred = model(Xt, zt).cpu().numpy()

    # ZUPT correction at inference: force speed to 0 when flagged stationary.
    pred_corrected = pred.copy()
    pred_corrected[is_zupt, 0] = 0.0

    results = {}
    for sid in np.unique(session_ids):
        mask = session_ids == sid
        speed_pred = pred_corrected[mask, 0]
        dh_pred = pred_corrected[mask, 1]
        speed_true = y_speed_heading[mask, 0]
        dh_true = y_speed_heading[mask, 1]

        dt_seq = np.full(len(speed_pred), dt_per_window)

        x_true, y_true = reconstruct_xy(speed_true, dh_true, dt_seq, start_heading=0.0)
        x_pred, y_pred = reconstruct_xy(speed_pred, dh_pred, dt_seq, start_heading=0.0)

        # Naive baseline: repeat the FIRST window's true speed/heading for
        # every subsequent window (no learning at all).
        speed_naive = np.full_like(speed_true, speed_true[0])
        dh_naive = np.zeros_like(dh_true)
        x_naive, y_naive = reconstruct_xy(speed_naive, dh_naive, dt_seq, start_heading=0.0)

        pos_err_model = np.sqrt((x_true - x_pred) ** 2 + (y_true - y_pred) ** 2)
        pos_err_naive = np.sqrt((x_true - x_naive) ** 2 + (y_true - y_naive) ** 2)

        results[int(sid)] = {
            "speed_mae": float(np.mean(np.abs(speed_pred - speed_true))),
            "heading_mae_rad": float(np.mean(np.abs(dh_pred - dh_true))),
            "final_pos_err_model_m": float(pos_err_model[-1]),
            "final_pos_err_naive_m": float(pos_err_naive[-1]),
            "mean_pos_err_model_m": float(np.mean(pos_err_model)),
            "mean_pos_err_naive_m": float(np.mean(pos_err_naive)),
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dir with train/val/test.npz")
    ap.add_argument("--out", default="model.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--window-sec", type=float, default=10.0,
                     help="approx real-world seconds spanned by one window "
                          "(window_len samples at ~10Hz) -- used only for "
                          "trajectory reconstruction timing")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train = load_split(args.dataset, "train")
    val = load_split(args.dataset, "val")
    test = load_split(args.dataset, "test")

    mean, std = compute_norm_stats(train["X"])
    print(f"Normalization stats (train-only): mean={mean}, std={std}")

    Xtr = apply_norm(train["X"], mean, std)
    Xva = apply_norm(val["X"], mean, std)

    zupt_tr = make_zupt_seq(train["X"].shape, train["is_zupt"])
    zupt_va = make_zupt_seq(val["X"].shape, val["is_zupt"])

    train_loader = build_loader(Xtr, train["y_speed_heading"], zupt_tr,
                                 args.batch_size, shuffle=True)
    val_loader = build_loader(Xva, val["y_speed_heading"], zupt_va,
                               args.batch_size, shuffle=False)

    model = IMURegressor(n_channels=train["X"].shape[-1],
                          hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)
        print(f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.5f}  "
              f"val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({
        "model_state": model.state_dict(),
        "norm_mean": mean,
        "norm_std": std,
        "hidden_size": args.hidden_size,
        "n_channels": train["X"].shape[-1],
    }, args.out)
    print(f"\nSaved best checkpoint (val_loss={best_val_loss:.5f}) -> {args.out}")

    window_len = train["X"].shape[1]
    dt_per_window = args.window_sec

    print("\n=== Test-set evaluation (per session) ===")
    results = evaluate_trajectory(
        model, test["X"], test["y_speed_heading"], test["is_zupt"],
        mean, std, device, test["session_id"], dt_per_window,
    )
    for sid, r in results.items():
        print(f"\nSession {sid}:")
        for k, v in r.items():
            print(f"  {k}: {v:.4f}")
        improvement = (r["mean_pos_err_naive_m"] - r["mean_pos_err_model_m"]) \
            / max(r["mean_pos_err_naive_m"], 1e-8) * 100
        print(f"  model vs naive improvement: {improvement:.1f}%")


if __name__ == "__main__":
    main()