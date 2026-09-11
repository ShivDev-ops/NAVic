# IDR Project — Preprocessing-First Restart

This restarts the Intelligent Dead Reckoning (IDR) project with data
cleaning/preprocessing as the foundation, following the discovery that the
raw `S-*.csv` timestamps are corrupted (concatenated recording sessions with
no clock reset — see `docs/status_report.md` if you copy it in).

## Why preprocessing first
Every previous accuracy hypothesis (time-shift misalignment, axis swap,
orientation compensation) was tested on top of data with at least one
physically impossible `dt` value. Fix the data before touching the model
again — map matching, EKF correction, and ensembling all assume the base
trajectory is "roughly right," which nothing built on corrupted windows can
be.

## Pipeline

```
raw CSVs  --[Stage A: split_sessions.py]-->  clean per-session files
          --[Stage B: build_dataset.py]-->  train/val/test .npz
```

See `data_contract.md` for the exact schema between the two stages —
**read this before writing code**, it's the interface the two of you build
against.

## Repo layout

```
src/
  common.py           shared constants + utilities (column names, dt math,
                       lat/lon projection, speed/heading conversion, ZUPT)
  synthetic_data.py    generates fake raw CSVs with the same corruption
                       pattern, for developing/testing without real data
  split_sessions.py   STAGE A (Person A) — timestamp-break detection +
                       session splitting
  build_dataset.py    STAGE B (Person B) — GPS/IMU alignment, label
                       generation, session-aware windowing, session-level
                       train/val/test split
data_contract.md       the schema contract between Stage A and Stage B
data/                   gitignored — raw and processed data live here locally
ISSUES.md               suggested GitHub issue breakdown / milestones
```

## Quickstart (test against synthetic data first)

```bash
pip install pandas numpy pyarrow

cd src
python synthetic_data.py --out ../data/raw --n-sessions 4 --session-len-sec 60
python split_sessions.py --imu ../data/raw/S-M.csv --gps ../data/raw/V-M.csv \
    --out ../data/processed/sessions
python build_dataset.py --sessions ../data/processed/sessions \
    --out ../data/processed/dataset --window-len 100 --stride 20
```

Then swap `../data/raw/S-M.csv` / `V-M.csv` for the real IO-VNBD files once
you've verified the actual column names in `src/common.py` match — the
constants there are best-guess placeholders from the project status report,
**not verified against the real CSV headers yet**.

## Before running on real data
1. Open a real `S-M.csv` / `V-M.csv` and confirm the column names in
   `src/common.py` (`RAW_TIME_COL`, `RAW_IMU_COLS`, `RAW_GPS_LAT_COL`, etc.)
   actually match.
2. Run `split_sessions.py` on the real files and read the diagnostic dt
   stats — confirm the break count and locations make sense before trusting
   the split.
3. Sanity-check a couple of the emitted `session_XXXX_gps.parquet` files by
   eye (plot lat/lon) — corrupted GPS ordering wouldn't necessarily show up
   as a `dt` red flag the same way IMU corruption did.

## Data is not committed
`data/` is gitignored. Each person regenerates their own local copies via
the scripts above, or downloads the real IO-VNBD subset separately — don't
commit CSVs/parquet/npz to the repo.
