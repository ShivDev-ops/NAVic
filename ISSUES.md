# Issue & Milestone Breakdown

Paste these into GitHub Issues (or run the `gh` commands at the bottom if
you have the CLI + a repo already created). Suggested labels: `stage-a`,
`stage-b`, `shared`, `blocking`.

## Milestone: Preprocessing v1

### Shared / setup
- [ ] **#1 — Repo scaffold + branch protection**
  Create the repo, push this scaffold, protect `main` (require PR review),
  create a long-lived `preprocessing` integration branch that both feature
  branches merge into before `main`.
- [ ] **#2 — Confirm real column names against actual IO-VNBD files** `blocking`
  Open a real `S-M.csv`/`V-M.csv`, verify every constant in `src/common.py`
  (`RAW_TIME_COL`, `RAW_IMU_COLS`, GPS lat/lon/speed column names). Update
  `common.py` in a PR both people review. **Nothing else should start on
  real data until this lands** — both stages import from here.
- [ ] **#3 — Review and lock `data_contract.md`**
  Both people read `data_contract.md`, raise any concerns about the
  interface (window shape, label representation, split granularity) *before*
  writing stage-specific code against it.

### Stage A — Ingestion & Cleaning (Person A)
- [ ] **#4 — Run break-detection on real data, characterize corruption** `stage-a`
  Run `split_sessions.py`'s diagnostics against the real files. Record: how
  many breaks, session length distribution, whether MIN_SESSION_SEC /
  BREAK_GAP_SECONDS in `common.py` need tuning for this dataset.
- [ ] **#5 — Validate IMU/GPS session pairing assumption** `stage-a` `blocking`
  The current pairing is order-based (IMU segment *i* ↔ GPS segment *i*).
  Confirm this holds for real data (same number of sessions in both files,
  same order) — if not, needs a different alignment strategy (e.g. matching
  by segment duration or a shared external counter column, if one exists).
- [ ] **#6 — Sanity-check emitted sessions**
  For a handful of `session_XXXX_gps.parquet` files, plot lat/lon and eyeball
  that each one is a single coherent trip, not a discontinuous jump.
- [ ] **#7 — Handle edge cases in real data**
  Duplicate timestamps, NaN GPS rows, sessions with very sparse GPS
  (fewer fixes than expected for their duration). Decide: drop, interpolate,
  or flag in manifest.

### Stage B — Alignment & Windowing (Person B)
- [ ] **#8 — Validate GPS→IMU interpolation on real (not synthetic) sessions** `stage-b`
  Real GPS dropout patterns near tunnel entries may behave differently than
  the synthetic generator. Check `build_session_labels()`'s `np.interp`
  behavior at session edges and during longer GPS gaps.
- [ ] **#9 — Decide real windowing config** `stage-b`
  Pick `window_len` / `stride` based on the actual GNSS-outage benchmark
  (e.g. 60s blackout blocks) rather than the placeholder 100/20 used for
  synthetic testing.
- [ ] **#10 — Tune ZUPT thresholds against real stationary segments** `stage-b`
  `detect_zupt()`'s `acc_thresh`/`gyro_thresh` are placeholders. Find a few
  real stopped-at-a-light segments and tune against those.
- [ ] **#11 — Re-run `diagnose_real_data.py`-style correlation check on clean data**
  Confirm the alignment correlation (was -0.22 on corrupted data) recovers
  substantially once Stage A + Stage B are both wired to real data. This is
  the actual validation that the fix worked.

### Integration
- [ ] **#12 — End-to-end run on real IO-VNBD subset**
  Both stages wired together against real data, output dataset produced,
  spot-checked.
- [ ] **#13 — Regenerate `gps_outage_replay.html` demo with the new pipeline's output**
  Once retraining happens on the cleaned dataset, refresh the existing demo
  artifact with new `demo_data.json`.

---

## Optional: create these via `gh` CLI

If you have the GitHub CLI authenticated and the repo already created,
something like this creates the issues directly (edit repo name first):

```bash
REPO="your-org/idr-project"

gh issue create -R "$REPO" -t "Repo scaffold + branch protection" -l shared
gh issue create -R "$REPO" -t "Confirm real column names against actual IO-VNBD files" -l shared,blocking
gh issue create -R "$REPO" -t "Review and lock data_contract.md" -l shared
gh issue create -R "$REPO" -t "Run break-detection on real data, characterize corruption" -l stage-a
gh issue create -R "$REPO" -t "Validate IMU/GPS session pairing assumption" -l stage-a,blocking
gh issue create -R "$REPO" -t "Sanity-check emitted sessions" -l stage-a
gh issue create -R "$REPO" -t "Handle edge cases in real data" -l stage-a
gh issue create -R "$REPO" -t "Validate GPS-to-IMU interpolation on real sessions" -l stage-b
gh issue create -R "$REPO" -t "Decide real windowing config" -l stage-b
gh issue create -R "$REPO" -t "Tune ZUPT thresholds against real stationary segments" -l stage-b
gh issue create -R "$REPO" -t "Re-run correlation check on clean data" -l stage-b
gh issue create -R "$REPO" -t "End-to-end run on real IO-VNBD subset" -l shared
gh issue create -R "$REPO" -t "Regenerate demo with new pipeline output" -l shared
```

## Suggested branches
```
main
  preprocessing                  <- integration branch, both merge here first
    preprocessing/stage-a-sessions   <- Person A
    preprocessing/stage-b-windowing  <- Person B
```
PR flow: feature branch → `preprocessing` (reviewed by the other person,
since you're each other's main reviewer on this) → `main` once the
end-to-end run (#12) passes.
