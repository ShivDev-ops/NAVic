# Data Contract: Cleaning (Person A) → Windowing (Person B)

This is the interface between the two preprocessing stages. Both people should
treat this file as the source of truth — if you need to change a column name
or shape, edit this file in a PR and tag the other person, don't just change
your own script silently.

## Stage A output (produced by `split_sessions.py`)

For each raw `S-*.csv` / `V-*.csv` pair, Stage A emits one clean, monotonic,
single-trip segment per detected session:

```
data/processed/sessions/
  session_0000_imu.parquet
  session_0000_gps.parquet
  session_0001_imu.parquet
  session_0001_gps.parquet
  ...
  manifest.csv
```

**`session_XXXX_imu.parquet`** — one row per IMU sample, sorted, monotonic `t_sec`:
| column | type | notes |
|---|---|---|
| t_sec | float64 | seconds since session start, strictly increasing |
| acc_x | float64 | |
| acc_y | float64 | |
| gyro_yaw | float64 | |
| gyro_pitch | float64 | |
| gyro_roll | float64 | |

**`session_XXXX_gps.parquet`** — one row per GPS fix, sorted, monotonic `t_sec`:
| column | type | notes |
|---|---|---|
| t_sec | float64 | same clock/origin as the IMU file for this session |
| lat | float64 | degrees |
| lon | float64 | degrees |
| speed | float64 | optional, NaN if not present in source |

**`manifest.csv`** — one row per session:
| column | type | notes |
|---|---|---|
| session_id | int | matches the `XXXX` in filenames |
| source_file | str | original raw file this came from |
| start_idx / end_idx | int | row indices in the raw file |
| n_imu_samples | int | |
| n_gps_fixes | int | |
| duration_sec | float | |
| dt_mean / dt_std / dt_max | float | diagnostics, for sanity checking |

**Guarantees Stage A promises to Stage B:**
- Every `t_sec` column is sorted and strictly increasing within a session.
- No session mixes rows from two different original recordings.
- Sessions shorter than `MIN_SESSION_SEC` (default 5s) are dropped, not emitted.
- IMU and GPS files for the same `session_id` share the same time origin
  (`t_sec = 0` at the same real moment).

**What Stage A does NOT promise:**
- Uniform sample rate within a session (IMU sampling can still jitter a bit —
  that's normal MEMS behavior, not corruption. Stage B should handle it, not
  assume a fixed dt).
- Alignment/interpolation between IMU and GPS — that's Stage B's job.

## Stage B output (produced by `build_dataset.py`)

```
data/processed/dataset/
  train.npz
  val.npz
  test.npz
  split_manifest.csv
```

Each `.npz` contains:
- `X`: float32 array, shape `(N, window_len, 5)` — raw IMU channels per window
- `y_speed_heading`: float32 array, shape `(N, 2)` — `(speed, delta_heading)` label per window
- `y_dxdy`: float32 array, shape `(N, 2)` — `(dx, dy)` label per window, for backward compatibility with the old model
- `session_id`: int array, shape `(N,)` — which session each window came from
- `is_zupt`: bool array, shape `(N,)` — zero-velocity flag for the window

**Split guarantee:** splitting is done **by `session_id`**, never by window —
so no session appears in more than one of train/val/test. This is the fix for
the leakage risk in the old pipeline.

## If you need to change this contract
Open a PR against `data_contract.md` only, get the other person's review,
*then* update code. This file changing without a corresponding code change
(or vice versa) is the #1 way this integration breaks.
