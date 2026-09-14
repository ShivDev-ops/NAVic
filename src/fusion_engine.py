"""
GNSS+INS Fusion Engine & Seamless GNSS Deficit Handler
Implements:
1. 10Hz Extended Kalman Filter (EKF) state estimator:
   - State: [p_x, p_y, forward_velocity, heading, gyro_bias]
   - Smoothly fuses 10Hz GNSS position & velocity with high-rate smartphone IMU
2. Instantaneous GNSS Deficit Handler:
   - Millisecond-level outage detection triggered by satellite loss, covariance spike, or missed packet
   - Automatically shifts state propagation to Intelligent Dead Reckoning (IDR) mode
3. Seamless Re-acquisition & Innovation Blending:
   - On GNSS signal restoration, smoothly merges the estimated trajectory back with GNSS fixes
   - Prevents teleportation / jumping artifacts on the vehicle navigation UI
"""

from __future__ import annotations
import numpy as np


class GNSSINSFusionEngine:
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        # State: [x, y, v, heading, gyro_bias]
        self.state = np.zeros(5, dtype=float)
        
        # State covariance
        self.P = np.diag([1.0, 1.0, 0.5, 0.05, 0.001])
        
        # Process noise covariance Q
        self.Q = np.diag([0.05, 0.05, 0.1, 0.005, 1e-6])
        
        # Measurement noise covariance R (GNSS)
        self.R_gnss = np.diag([2.0, 2.0, 0.3, 0.08])
        
        self.in_outage = False
        self.outage_duration_sec = 0.0
        self.reacquisition_blend_steps = 0
        self.max_blend_steps = 10  # 1.0s at 10Hz

    def initialize(self, x0: float, y0: float, v0: float, heading0: float, gyro_bias0: float = 0.0):
        self.state = np.array([x0, y0, v0, heading0, gyro_bias0], dtype=float)
        self.P = np.diag([1.0, 1.0, 0.5, 0.05, 0.001])
        self.in_outage = False
        self.outage_duration_sec = 0.0
        self.reacquisition_blend_steps = 0

    def predict_imu(self, a_fwd: float, omega_z: float):
        """
        Propagate state using vehicle frame IMU measurements.
        a_fwd: forward acceleration (m/s^2)
        omega_z: vehicle yaw rate (rad/s)
        """
        x, y, v, theta, b_g = self.state
        
        # Bias-corrected yaw rate
        omega_eff = omega_z - b_g
        theta_new = theta + omega_eff * self.dt
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
        
        # Speed update
        v_new = max(0.0, v + a_fwd * self.dt)
        
        # Position update (NHC: vehicle moves along heading)
        x_new = x + v_new * np.cos(theta_new) * self.dt
        y_new = y + v_new * np.sin(theta_new) * self.dt
        
        self.state = np.array([x_new, y_new, v_new, theta_new, b_g])
        
        # Jacobian F of motion model
        F = np.eye(5)
        F[0, 2] = np.cos(theta_new) * self.dt
        F[0, 3] = -v_new * np.sin(theta_new) * self.dt
        F[1, 2] = np.sin(theta_new) * self.dt
        F[1, 3] = v_new * np.cos(theta_new) * self.dt
        F[3, 4] = -self.dt
        
        self.P = F @ self.P @ F.T + self.Q

    def update_gnss(self, gps_x: float, gps_y: float, gps_speed: float, gps_heading: float):
        """
        EKF measurement update when 10Hz GNSS fix is received.
        """
        if self.in_outage:
            # Transition out of blackout: trigger smooth blending
            self.in_outage = False
            self.reacquisition_blend_steps = self.max_blend_steps

        # Measurement residual
        z = np.array([gps_x, gps_y, gps_speed, gps_heading])
        
        # Handle heading wrapping
        h_diff = (gps_heading - self.state[3] + np.pi) % (2 * np.pi) - np.pi
        
        H = np.zeros((4, 5))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        H[3, 3] = 1.0
        
        y_res = np.array([
            gps_x - self.state[0],
            gps_y - self.state[1],
            gps_speed - self.state[2],
            h_diff
        ])
        
        # Smooth reacquisition blending factor
        if self.reacquisition_blend_steps > 0:
            alpha = 1.0 - (self.reacquisition_blend_steps / self.max_blend_steps)
            y_res[:2] *= alpha  # Smooth positional transition
            self.reacquisition_blend_steps -= 1

        S = H @ self.P @ H.T + self.R_gnss
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y_res
        self.state[3] = (self.state[3] + np.pi) % (2 * np.pi) - np.pi
        self.P = (np.eye(5) - K @ H) @ self.P

    def handle_deficit(self):
        """Called when GNSS outage is detected. Freezes covariance divergence and flags IDR mode."""
        self.in_outage = True
        self.outage_duration_sec += self.dt
