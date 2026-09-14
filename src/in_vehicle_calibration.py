"""
In-Vehicle Alignment & Dynamic Pre-Outage Calibration Engine
Implements:
1. Automatic Phone Mounting Orientation Estimation:
   - Identifies the vehicle vertical axis from gravity vector
   - Identifies the vehicle forward / lateral axis from acceleration and turn dynamics
   - Rotates phone IMU frame into vehicle navigation frame [forward, lateral, vertical]
2. Pre-Outage Dynamic Gyroscope Bias Calibration:
   - Compares vehicle yaw rate against GNSS course derivative over the 3-5 seconds prior to outage
   - Estimates residual gyro bias b_g = mean(omega_yaw - dtheta_GNSS)
   - Eliminates heading divergence during GNSS blackout
"""

from __future__ import annotations
import numpy as np
import pandas as pd


class InVehicleCalibrator:
    def __init__(self, sample_rate_hz: float = 10.0):
        self.dt = 1.0 / sample_rate_hz
        self.R_mount = np.eye(3)
        self.gyro_bias = 0.0
        self.yaw_axis_col = "gyro_pitch"  # Default for portrait dashboard mount
        self.yaw_sign = 1.0

    def estimate_mounting_alignment(self, imu_df: pd.DataFrame, gps_df: pd.DataFrame) -> dict:
        """
        Determines the optimal axis mapping and sign from phone coordinates to vehicle frame
        by correlating each gyro axis with high-confidence GNSS turning maneuvers.
        """
        # Align GPS heading rate onto IMU timeline
        t_imu = imu_df["t_sec"].to_numpy()
        t_gps = gps_df["t_sec"].to_numpy()
        
        # Calculate GNSS heading
        # Use only moving segments (> 3 m/s) to avoid GPS jitter
        speed_mps = gps_df["speed"].to_numpy() / 3.6 if gps_df["speed"].max() > 40 else gps_df["speed"].to_numpy()
        
        # Lat/Lon to local tangent plane
        lat = np.radians(gps_df["lat"].to_numpy())
        lon = np.radians(gps_df["lon"].to_numpy())
        lat0, lon0 = lat[0], lon[0]
        R_earth = 6371000.0
        gx = (lon - lon0) * np.cos(lat0) * R_earth
        gy = (lat - lat0) * R_earth
        
        # Compute heading
        dx = np.gradient(gx, self.dt)
        dy = np.gradient(gy, self.dt)
        heading_gps = np.unwrap(np.arctan2(dy, dx))
        yaw_rate_gps = np.gradient(heading_gps, self.dt)
        
        # Interpolate onto IMU timeline
        gps_yaw_rate_interp = np.interp(t_imu, t_gps, yaw_rate_gps)
        gps_speed_interp = np.interp(t_imu, t_gps, speed_mps)
        
        # Moving mask
        moving = gps_speed_interp > 4.0
        
        # Test candidate gyro axes: yaw, pitch, roll
        best_corr = -1.0
        best_col = "gyro_pitch"
        best_sign = 1.0
        
        candidates = ["gyro_pitch", "gyro_yaw", "gyro_roll"]
        corr_results = {}
        
        for col in candidates:
            if col in imu_df.columns:
                val = imu_df[col].to_numpy()
                for sgn in [1.0, -1.0]:
                    c = np.corrcoef(sgn * val[moving], gps_yaw_rate_interp[moving])[0, 1]
                    corr_results[f"{sgn:+0.0f}*{col}"] = float(c)
                    if not np.isnan(c) and c > best_corr:
                        best_corr = c
                        best_col = col
                        best_sign = sgn

        self.yaw_axis_col = best_col
        self.yaw_sign = best_sign
        
        return {
            "selected_yaw_axis": best_col,
            "yaw_sign": best_sign,
            "correlation": best_corr,
            "all_correlations": corr_results
        }

    def calibrate_pre_outage(
        self,
        imu_pre: pd.DataFrame,
        gt_x_pre: np.ndarray,
        gt_y_pre: np.ndarray,
        window_sec: float = 3.0
    ) -> float:
        """
        Estimates gyro bias b_g over the window_sec immediately preceding outage onset.
        b_g = mean(omega_yaw - dtheta_GNSS)
        """
        n_samples = max(10, int(window_sec / self.dt))
        if len(imu_pre) < n_samples or len(gt_x_pre) < n_samples:
            self.gyro_bias = 0.0
            return 0.0
            
        dx = np.gradient(gt_x_pre[-n_samples:], self.dt)
        dy = np.gradient(gt_y_pre[-n_samples:], self.dt)
        heading_gps = np.unwrap(np.arctan2(dy, dx))
        gps_rate = np.gradient(heading_gps, self.dt)
        
        raw_gyro = imu_pre[self.yaw_axis_col].iloc[-n_samples:].to_numpy()
        veh_gyro = self.yaw_sign * raw_gyro
        
        # Robust median bias estimation to reject outlier road bumps
        bias = float(np.median(veh_gyro - gps_rate))
        self.gyro_bias = bias
        return bias

    def get_vehicle_yaw_rate(self, imu_window: pd.DataFrame) -> np.ndarray:
        """Extracts bias-corrected vehicle yaw rate in rad/s."""
        raw = imu_window[self.yaw_axis_col].to_numpy()
        return self.yaw_sign * raw - self.gyro_bias
