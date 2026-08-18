# MTP refine sequence

After an initial (or partial) fit, force RMS may sit slightly above the validation gate and BFGS may stop with **step limit reached** rather than a clean small-decrement exit. The refine workflow addresses that without throwing away the best intermediate potential.

Entry points:

```bash
./run.sh --only refine          # orchestrator
./run.sh --refine               # after train in the same run
./scripts/refine_mtp.sh         # direct
./scripts/refine_mtp.sh [train.cfg] [start.mtp]
```

`run.sh` also schedules refine automatically when validation fails and `AUTO_REFINE=1` (default). Disable with `--skip-refine` or `AUTO_REFINE=0`.

Related: [CONFIGURATION.md](CONFIGURATION.md) · [HOW_TO_USE.md](HOW_TO_USE.md) · [ACTIVE_LEARNING.md](ACTIVE_LEARNING.md)

---

## When to refine

Typical symptoms (from `active_learning/logs/train.log` / `calc_errors_train.log`):

- `BFGS ... step limit reached` (not fully converged)
- Force RMS slightly above `VAL_FORCE_RMS_MAX` (default 0.08 eV/Å)
- Post-train radial rescaling applied only once after a truncated BFGS run

If force RMS is already within threshold and BFGS did not hit the step limit, refine exits immediately as a no-op.

---

## Stages

Each train stage restarts BFGS from the **best intermediate pot**. `mlp train` always ends with a linear re-fit plus radial rescaling, so continue rounds are not “wasted” Hessian steps on a dead search direction.

| Stage | What | Controls |
|-------|------|----------|
| **0. Filter** | Drop bad / irrelevant DFT from the train set | `FILTER_BAD_DFT=1` (default) |
| **1. Continue BFGS** | Restart from best pot until step limit clears (up to N rounds) | `TRAIN_CONTINUE_MAX_ROUNDS`, `MAX_ITER_CONTINUE` |
| **2. Force-weighted retrain** | Higher force weight from best intermediate | `FORCE_WEIGHT_RETRAIN`, `FORCE_RETRAIN_*` |
| **3. High-error subset** | Force-focused fit on worst force-RMSE configs, then polish on full set | `ENABLE_HIGH_ERROR_RETRAIN`, `HIGH_ERROR_*` |
| **4. Mindist check** | Compare dataset mindist vs `MTP_MIN_DIST` | advisory |
| **5. Validate + next steps** | Re-check thresholds; suggest AL or level escalate | `AUTO_ESCALATE_LEVEL` |

### Stage 0 — filter bad DFT

Uses `scripts/filter_cfg.py` to reject configs that are likely harmful:

- Missing energy or forces  
- Unphysically short min pair distance (`FILTER_MIN_DIST_ABS`, default 0.5 Å)  
- Extreme force magnitudes (`FILTER_MAX_FORCE`, default 50 eV/Å)  
- Energy-per-atom outliers vs median (`FILTER_MAX_EPA_OUTLIER`, default 5 eV)  
- Explicit bad labels (`Feature EFS_by = VASP_not_converged`, etc.)

Filtered set is written under `active_learning/refine/train_filtered.cfg`. When filtering the main `train.cfg`, a one-time backup is saved as `datasets/initial/train_before_filter.cfg`.

Standalone:

```bash
./scripts/filter_cfg.py train.cfg train_clean.cfg \
  --max-force 50 --min-dist 0.5 --max-epa-outlier 5.0 \
  --rejected-cfg rejected.cfg
```

### Stage 1 — continue BFGS

While the last train hit the step limit and `round < TRAIN_CONTINUE_MAX_ROUNDS` (default 3):

```text
TRAIN_START_MTP=<best>  EFFECTIVE_MAX_ITER=MAX_ITER_CONTINUE  train_mtp.sh
→ active_learning/refine/WC_L{level}_continue_r{N}.mtp
```

Already-complete continue stages are skipped on resume (crash-safe).

### Stage 2 — force-weighted retrain

If force RMS still exceeds the gate:

| Variable | Default |
|----------|---------|
| `FORCE_WEIGHT_RETRAIN` | `0.5` |
| `FORCE_RETRAIN_ENERGY_WEIGHT` | same as `ENERGY_WEIGHT` |
| `FORCE_RETRAIN_STRESS_WEIGHT` | same as `STRESS_WEIGHT` |
| `FORCE_RETRAIN_MAX_ITER` | `1500` |

### Stage 3 — high-error subset + polish

If still failing and `ENABLE_HIGH_ERROR_RETRAIN=1` (default):

1. `mlp calc-efs` → predicted EFS on the train set  
2. `extract_high_error_cfg.py` keeps the top force-RMSE fraction (and any above an absolute floor)  
3. Force-heavy train on that subset (`HIGH_ERROR_FORCE_WEIGHT=1.0`, low energy/stress)  
4. Polish on the full train set from the high-error pot  

| Variable | Default |
|----------|---------|
| `HIGH_ERROR_TOP_FRAC` | `0.15` |
| `HIGH_ERROR_MIN_FORCE_RMSE` | `0.15` eV/Å |
| `HIGH_ERROR_MAX_ITER` | `500` |
| `HIGH_ERROR_POLISH_MAX_ITER` | `1000` |

Standalone high-error extract:

```bash
mlp calc-efs pot.mtp train.cfg pred.cfg
./scripts/extract_high_error_cfg.py train.cfg pred.cfg high_err.cfg \
  --top-frac 0.15 --min-force-rmse 0.15
```

### Stage 4 — mindist vs min cutoff

Reports global mindist vs `MTP_MIN_DIST`. If short pairs remain, lower `MTP_MIN_DIST` toward ~0.99×mindist (or exclude those configs) and retrain.

### Stage 5 — next steps if still stuck

Refine prints practical next actions:

1. Active learning on high-force-error / high-grade configs: `./run.sh --al`  
   Leftover AIMD staging frames already carry VASP EFS and are merged automatically, then BFGS is forced on the grown `train.cfg` (`TRAIN_FORCE=1`).  
2. High-error DFT subset (already labeled; for prioritization): `active_learning/refine/high_force_error.cfg`  
3. Escalate level: `MTP_LEVEL=22 ./run.sh --skip-dataset`  

Optional automatic level bump (off by default):

```bash
AUTO_ESCALATE_LEVEL=1 ESCALATE_LEVEL=22 ./scripts/refine_mtp.sh
```

---

## Start-potential resolution

`refine_mtp.sh` / `train_mtp.sh` look for a **fitted** MTP (has `species_coeffs` and `moment_coeffs`), not a bare template:

1. Explicit argument / `TRAIN_START_MTP`  
2. Newest fitted pot among `WC_L{level}_*.mtp`, `curr`, `trained`, and `refine/` stages  

Set `TRAIN_FRESH=1` to ignore existing pots and start from the untrained template.

---

## Crash recovery

Refine is designed to survive multi-day interruptions:

- Each stage pot is promoted to `TRAINED_MTP` as soon as it is fitted  
- Incomplete postproc after BFGS is salvaged from the train log (`TRAIN ERRORS` block)  
- Completed `continue_rN` stages are skipped when re-entering refine  
- `run.sh` auto-resume can re-enter at `refine` via `workflow_state.env`

State / logs of interest:

| Path | Role |
|------|------|
| `active_learning/refine/WC_L*_*.mtp` | Per-stage potentials |
| `active_learning/logs/continue_r*.log` | Continue BFGS logs |
| `active_learning/logs/train_status.env` | Last stage metrics (`FORCE_RMS`, `STEP_LIMIT`, …) |
| `active_learning/logs/calc_errors_*.log` | Per-tag error reports |
| `active_learning/workflow_state.env` | Orchestrator resume point |

---

## Environment cheat sheet

```bash
# More / fewer continue rounds
TRAIN_CONTINUE_MAX_ROUNDS=5 MAX_ITER_CONTINUE=3000 ./scripts/refine_mtp.sh

# Stronger force focus
FORCE_WEIGHT_RETRAIN=1.0 ./scripts/refine_mtp.sh

# Skip DFT filter or high-error stage
FILTER_BAD_DFT=0 ENABLE_HIGH_ERROR_RETRAIN=0 ./scripts/refine_mtp.sh

# Stricter filter
FILTER_MAX_FORCE=30 FILTER_MIN_DIST_ABS=0.8 ./scripts/refine_mtp.sh
```

Full variable list: [CONFIGURATION.md](CONFIGURATION.md).
