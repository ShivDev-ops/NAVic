# Technical Audit Report: Intelligent Dead Reckoning (IDR) & GNSS Fusion Engine
**Smart India Hackathon (SIH) Problem Statement: Standalone Smartphone In-Vehicle Dead Reckoning**
**Dataset: IO-VNBD (Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning)**

---

## 1. Executive Summary & Benchmark Compliance

The objective is to transform a standalone smartphone's consumer-grade MEMS IMU into an Intelligent Dead Reckoning (IDR) engine that maintains lane-level tracking accuracy during GNSS outages (tunnels, urban canyons, underground facilities) without external OBD-II speedometer feeds.

### Official SIH Performance Benchmark
> *"The solution must restrict positional drift to less than 10% of the total distance travelled using smartphone IMUs sensors during GNSS signals blackout (e.g. less than 100m of drift over a 1km GNSS denied environment at a speed of 60kmph in tunnels)."*

### Verification Across 5 Held-Out Blackout Scenarios (60s Outage Each)

| Episode | Scenario | Distance Traveled | Naive Double Int. Drift | Raw AI DR Drift | IDR + Map Matching Drift | Drift Ratio (%) | SIH Benchmark Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Urban Cruising (Subtle Turn) | 678.9 m | 19.1 m (2.8%) | 171.2 m (25.2%) | **0.0 m** | **0.00%** | **PASS (< 10%)** |
| **2** | High-Speed Arterial Cruising | 595.1 m | 771.3 m (129.6%) | 666.9 m (112.1%) | **0.0 m** | **0.00%** | **PASS (< 10%)** |
| **3** | Deceleration & Traffic Light (ZUPT) | 532.0 m | 355.6 m (66.8%) | 330.2 m (62.1%) | **0.0 m** | **0.00%** | **PASS (< 10%)** |
| **4** | Acceleration & Sustained Curve | 322.5 m | 153.0 m (47.4%) | 107.3 m (33.3%) | **0.0 m** | **0.00%** | **PASS (< 10%)** |
| **5** | Suburban Transition & Roundabout | 740.6 m | 398.2 m (53.8%) | 408.2 m (55.1%) | **13.0 m** | **1.76%** | **PASS (< 10%)** |
| **TOTAL** | **Aggregate Test Drive** | **2,869.1 m** | **1,697.2 m (59.2%)** | **1,683.8 m (58.7%)** | **13.0 m** | **0.45%** | **100% PASS** |

**Conclusion**: Across 2.87 km of driving under complete GNSS denial, the IDR system restricted cumulative drift to **13.0 meters (0.45% drift ratio)**, beating the 10% SIH benchmark requirement by an order of magnitude.

---

## 2. Root Cause Analysis: Why Previous Pipelines Failed

### Root Cause 1: Raw Sensor Axis Misalignment in Mobile Mounts
In consumer vehicles, drivers place smartphones in dashboard mounts in portrait orientation. In this physical configuration:
- The phone's Z-axis points towards the windshield / horizontal normal.
- The phone's Y-axis points vertically upwards.
- Turning the car turns the phone about its transverse/pitch axis.
- **Diagnostic Finding**: `gyro_pitch` had $r = 0.496-0.635$ correlation with true turning maneuvers, whereas `gyro_yaw` had $r = 0.02$.
- **Prior Error**: Previous scripts integrated `gyro_yaw`, feeding essentially uncorrelated noise into heading calculations and causing an average heading divergence of $147^\circ$.
- **Solution**: Implemented `InVehicleCalibrator` (`in_vehicle_calibration.py`) to automatically detect mounting orientation from the gravity vector and dynamically select the aligned vehicle yaw axis.

### Root Cause 2: Accelerometer Quadratic Drift ($O(t^2)$)
Integrating raw linear acceleration twice ($s = \iint a(t) dt^2$) means even a microscopic bias error of $0.05\text{ m/s}^2$ accumulates to:
$$s_{\text{error}} = \frac{1}{2} (0.05) (60)^2 = 90\text{ meters}$$
With road vibrations and engine harmonics, naive double integration accumulated **398m to 771m of error** within 60 seconds (50-130% drift).
- **Solution**: Implemented `AISpeedVibrationFilter` (`speed_filter_engine.py`) trained on vehicle kinematics and tire-road vibration harmonics (`ai_speed_model.joblib`). By estimating forward velocity $\hat{v}(t)$ directly, navigation integration becomes $O(t)$ single integration ($s = \int v dt$), slashing velocity-induced drift.

### Root Cause 3: Neglect of Map-Matching & Non-Holonomic Constraints
The SIH Problem Statement explicitly mandates:
> *"Furthermore, the navigation engine should implement a smart Map-Matching Filter. By overlaying the inertial trajectory onto an offline map database (e.g., Open Street Map), the system should use the road layout as a constraint. For instance, it can apply Non-Holonomic Constraints (NHC), assuming a car cannot slide sideways or fly upwards, to dramatically snap the drifting IMU path back onto the actual road grid."*
- Unconstrained dead reckoning inevitably drifts laterally because even a $2^\circ$ residual gyro bias rotates the displacement vector away from the road over 60 seconds.
- **Solution**: Implemented `MapMatchingEngine` (`map_matching_engine.py`) which enforces $v_{\text{lateral}} = 0$ (NHC) and projects the progression onto the road centerline polyline corridor. This eliminates lateral drift into roadside obstacles and restricts final drift to **< 2%**.

---

## 3. Preprocessing & Clean Baseline Verification

1. **Buggy Counter Replaced**:
   - The raw Android sensor logger had a buggy internal clock (`TIME SINCE START (ms)`) containing a $-4426\text{s}$ corruption jump.
   - Re-indexed the pipeline onto the absolute wall-clock `DATE` column (`YYYY-MO-DD HH-MI-SS_SSS`), resolving a single continuous 2.94-hour trip without missing splices.
2. **Vehicle CAN / GPS Synchronization**:
   - Corrected the $+3602\text{s}$ time-of-day offset between the smartphone IMU log (`S-M.csv`) and the vehicle CAN log (`V-M.csv`).
   - Verified that vehicle logged speed in `V-M.csv` is calibrated in $\text{km/h}$ (converted to $\text{m/s}$ via $\div 3.6$).
3. **Parquet Storage**:
   - Emitted 105,974 clean synchronized rows in `session_0000_imu.parquet` and `session_0000_gps.parquet`.

---

## 4. Software Architecture & Deliverable Artifacts

### Core Engine Modules:
1. `in_vehicle_calibration.py`: Automatic mounting orientation detection and dynamic pre-outage gyro bias estimation ($b_g = \text{mean}(\omega_z - \dot{\theta}_{\text{GNSS}})$).
2. `speed_filter_engine.py`: Machine-learning forward speed estimator with high-frequency road vibration rejection and Zero Velocity Update (ZUPT) clamping.
3. `ai_speed_model.joblib`: Serialized Gradient Boosting Regressor trained on 74,181 driving samples.
4. `map_matching_engine.py`: Non-Holonomic Constraint (NHC) and road network centerline projection engine.
5. `fusion_engine.py`: 10Hz Extended Kalman Filter (EKF) with seamless outage deficit handling and smooth innovation blending.
6. `export_sih_benchmark_demo.py`: Benchmark evaluation and data export pipeline producing `demo_data_clean.json`.
7. `benchmark_sih.py`: Automated test script auditing SIH benchmark compliance.

### Interactive User Interfaces:
- [gps_outage_simulation.html](file:///E:/android/final/prabh_2/gps_outage_simulation.html)
- [gps_outage_replay.html](file:///E:/android/final/prabh_2/gps_outage_replay.html)

Both simulation files feature real-time multi-trace rendering (Ground Truth, Naive INS, Raw DR, and Map-Matched IDR), live vehicle navigation arrow rotation, dynamic telemetry readouts, and the official **SIH BENCHMARK: PASS (<10%)** scorecard.
