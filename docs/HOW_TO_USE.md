# How to use the WC MTP pipelines

Step-by-step setup, the main `run.sh` workflow, resume, refine, active learning, and script reference.

Related docs:

- [CONFIGURATION.md](CONFIGURATION.md) — hyperparameters and environment overrides  
- [REFINE.md](REFINE.md) — post-train refine sequence  
- [ACTIVE_LEARNING.md](ACTIVE_LEARNING.md) — AL loop in depth  
- [DATA_LAYOUT.md](DATA_LAYOUT.md) — directory layout and gitignore  

---

## 1. Install dependencies

### MLIP-2

```bash
export MLIP_ROOT="${HOME}/software/mlip-2"
export PATH="${MLIP_ROOT}/bin:${PATH}"
which mlp
ls "${MLIP_ROOT}/untrained_mtps/"   # level templates 02.mtp … 28.mtp
```

If MLIP lives elsewhere:

```bash
export MLIP_ROOT=/path/to/mlip-2
export MLP="${MLIP_ROOT}/bin/mlp"
```

### MPI

Train and active-learning stages run under `mpirun` by default:

```bash
which mpirun
MPI_NPROCS=8 ./run.sh --skip-dataset   # override rank count
MPI_NPROCS=1 ./run.sh --skip-dataset   # serial
```

On single-node Intel MPI, the config forces `I_MPI_FABRICS=shm` to avoid multi-day TCP `POLLERR` crashes. For multi-node, override `I_MPI_FABRICS_LOCAL` / fabrics as appropriate.

### Kerberos / NFS staging

If `mlp` lives under NFS with Kerberos (`$HOME` on many clusters), multi-day nohup jobs can fail with `execvp ... Permission denied` after tickets expire. With `MLP_STAGE=1` (default), the workflow copies `mlp` to `.bin/mlp` under the project tree (typically a non-krb filesystem such as `/slask`).

### Python

Scripts use **Python 3** with the standard library only:

- `merge_cfg.py`, `subsample_cfg.py`, `validate_mtp.py`, `prepare_mtp_template.py`
- `filter_cfg.py`, `extract_high_error_cfg.py` (refine helpers)
- `cfg_label_status.py` (detect Energy + forces on AL queues)

### Local DFT data (not in git)

Keep large VASP outputs **outside git**. Typical layout next to `run.sh`:

```text
wc-MTP-workflows/
  run.sh
  scripts/
  templates/
  datasets/
    sources.conf
    candidates/          # optional new MD frames for AL
    labeled/             # only for unlabeled AL picks that need new VASP
  vac_W_2300/OUTCAR
  vac_W_2500_ML/OUTCAR
  vac_W_2500_small_ML/OUTCAR
  vac_W_2800_ML/OUTCAR
  vac_W_2800_small_ML/OUTCAR
```

Folder names and temperatures are defined in `scripts/mtp_config.sh` (`AIMD_SOURCES` / `AIMD_SOURCE_TEMPS`).

---

## 2. One-command full pipeline

From the project root:

```bash
chmod +x run.sh scripts/*.sh scripts/*.py
./run.sh
```

This runs:

1. **Template** — ensure `templates/WC_L20.mtp` (or regenerate from MLIP untrained MTPs).  
2. **Dataset** — convert AIMD `OUTCAR`s + `datasets/sources.conf` → `datasets/initial/train.cfg`.  
3. **Train** — linear MTP fit (continues any fitted pot already on disk) → `active_learning/WC_L20_trained.mtp`.  
4. **Validate** — parse `mlp calc-errors` logs against thresholds.  
5. **Refine** (if validation fails and `AUTO_REFINE=1`) — more BFGS, force-weighted retrain, high-error subset.  

Logs go to `logs/run_YYYYMMDD_HHMMSS.log`. Step state: `active_learning/workflow_state.env`.

### Resume after crash (default)

```bash
./run.sh          # continues from the failed / incomplete step
./run.sh --fresh  # ignore resume skips; re-run selected steps
```

| Situation | What resume does |
|-----------|------------------|
| Dataset already built | Skips dataset |
| BFGS finished, calc-errors died | Finishes errors + status only |
| Pot trained + validated | Skips train; re-validates cheaply |
| Mid-refine crash | Re-enters refine; skips completed continue rounds |
| AL paused (`CURRENT_STATUS=paused`) | Reuses `iter_*/dft_queue.cfg`; merges if already labeled |
| AL iter already merged (`merged.ok`) | Skips that iteration only if the pot still matches `train.cfg` |

Concurrent second run while a step is live is blocked (PID recorded in state). Use `--fresh` only if that process is gone.

### Useful flags

| Flag | Effect |
|------|--------|
| `--skip-dataset` | Reuse existing `train.cfg` |
| `--skip-train` | Do not train (validate / refine / al only) |
| `--skip-validate` | Skip error thresholds |
| `--refine` | Force refine sequence after train |
| `--skip-refine` | Do not auto-refine when validation fails |
| `--al` | After train, run AL (auto-merge already-labeled leftover AIMD frames) |
| `--candidates FILE` | Candidate pool for AL (implies `--al`) |
| `--labeled FILE` | Merge new DFT labels into `train.cfg` before retrain |
| `--per-traj` | Also train one MTP per AIMD folder |
| `--only STEP` | `template` \| `dataset` \| `train` \| `validate` \| `refine` \| `al` \| `per-traj` |
| `--fresh` / `--no-resume` | Ignore resume state for step planning |
| `--resume` | Force auto-resume planning (default) |
| `-h` / `--help` | Full help |

Examples:

```bash
./run.sh --only dataset
./run.sh --skip-dataset
./run.sh --only refine
MTP_LEVEL=22 ./run.sh --skip-dataset
TRAIN_FRESH=1 ./run.sh --skip-dataset   # random init; ignore trained pots
./run.sh --al --candidates datasets/candidates/md_frames.cfg
MPI_NPROCS=8 ./run.sh
```

---

## 3. Building the training set

```bash
./scripts/build_initial_dataset.sh
# or
./run.sh --only dataset
```

### What it does

1. Converts each `vac_W_*/OUTCAR` to staging `.cfg` via `mlp convert-cfg --input-format=vasp-outcar` (serial).  
2. Subsamples high-T trajectories more densely (`HIGH_T_STRIDE`, default 25 above 1800 K; else `SUBSAMPLE_STRIDE` 50).  
3. Merges optional entries from `datasets/sources.conf`.  
4. Writes `datasets/initial/train.cfg` (and `train_full.cfg` / manifest under `datasets/initial/`).  

### `datasets/sources.conf` format

```text
# category:kind:path
bulk:outcar:/path/to/bulk/OUTCAR
defect:cfg:datasets/static/defects.cfg
surface:dir:/path/to/surface_cfgs/
```

| Field | Values |
|-------|--------|
| **category** | `bulk`, `defect`, `high_t`, `close_cc`, `surface`, `pathway` |
| **kind** | `outcar` \| `cfg` \| `dir` |
| **path** | Absolute or project-relative |

Target mix (guidance only): ~500–2000 structures total; see comments in `sources.conf`.

### Recommended VASP settings (labels)

- Functional: **PBE**  
- PAW: **W_sv + C**  
- ENCUT: **450–500 eV**  
- k-spacing: **~0.025 Å⁻¹** (Gamma-only for very large cells if converged)  

---

## 4. Training the MTP

```bash
./scripts/train_mtp.sh
# or
./run.sh --skip-dataset
```

Optional arguments:

```bash
./scripts/train_mtp.sh path/to/train.cfg path/to/output.mtp
```

### Continue vs fresh

By default (`TRAIN_RESUME=auto`), if a fitted pot already exists under `active_learning/` (trained, curr, or refine stages), training **continues from the newest** instead of random-init from the template.

```bash
TRAIN_FRESH=1 ./run.sh --skip-dataset          # force template / random init
TRAIN_START_MTP=/path/to.mtp ./scripts/train_mtp.sh
```

Key hyperparameters (from `mtp_config.sh`):

```bash
MTP_LEVEL=20
MTP_MIN_DIST=1.2
MTP_MAX_DIST=6.0
MTP_RADIAL_BASIS_SIZE=11
ENERGY_WEIGHT=1.0
FORCE_WEIGHT=0.1
STRESS_WEIGHT=0.005
MAX_ITER=2000
MPI_NPROCS=19
```

Increase level if defect forces remain bad after active learning:

```bash
MTP_LEVEL=22 ./run.sh --skip-dataset
```

Train writes status to `active_learning/logs/train_status.env` (`FORCE_RMS`, `STEP_LIMIT`, weights, paths) and tags logs as `calc_errors_<tag>.log`. For the default tag `train`, that log *is* `calc_errors_train.log`; postproc no longer `cp`s the file onto itself (GNU cp used to abort there under `set -e`). If BFGS finished but status was not sealed, `TRAIN_RESUME_POSTPROC=1 ./scripts/train_mtp.sh` reuses the existing errors log and writes `FORCE_RMS` / `POSTPROC_PENDING=0`.

---

## 5. Validation

```bash
./run.sh --only validate
# or directly
python3 scripts/validate_mtp.py active_learning/logs/calc_errors_train.log \
  --force-rms-max 0.08 \
  --force-mae-max 0.08 \
  --energy-per-atom-max 0.005 \
  --stress-rms-max 0.5
```

Default pass criteria:

| Metric | Max |
|--------|-----|
| Force RMS | 0.08 eV/Å |
| Force MAE | 0.08 eV/Å |
| Energy/atom RMS | 5 meV/atom |
| Stress RMS | 0.5 GPa |

If validation fails and `AUTO_REFINE=1` (default), `run.sh` schedules the refine sequence next.

If defect-region force RMSE stays above ~0.1 eV/Å after refine + AL, raise `MTP_LEVEL`.

---

## 6. Refine sequence

When BFGS hits the step limit or force RMS is slightly high:

```bash
./run.sh --only refine
# or
./scripts/refine_mtp.sh
```

Stages (details in [REFINE.md](REFINE.md)):

0. Filter bad DFT (`filter_cfg.py`)  
1. Continue BFGS from best pot (up to `TRAIN_CONTINUE_MAX_ROUNDS`)  
2. Force-weighted retrain (`FORCE_WEIGHT_RETRAIN`)  
3. Optional high-error subset fit + full-set polish  
4. Mindist vs min cutoff check  
5. Validate; recommend AL or `MTP_LEVEL=22` if still stuck  

---

## 7. Active-learning loop

Goal: select structures where the MTP extrapolates (high maxvol grade), add their DFT labels, retrain—until the reconstructed C–C dimer is stable (~3–4 eV lowering vs unreconstructed vacancy).

The initial `train.cfg` is a **stride subsample** of the AIMD OUTCARs (every 25th high-T frame by default). The unused frames are already DFT-labeled. If `datasets/candidates/` is empty, AL grades that leftover pool and **reuses those labels**.

Short cycle:

1. Optional: dump new MTP-MD / LAMMPS frames as `.cfg` into `datasets/candidates/`. Otherwise the leftover AIMD staging set is used.  
2. `./run.sh --al` — one command runs **all** AL iterations up to `AL_MAX_ITERATIONS` (default 20). After each verified retrain it starts the next grade/select automatically.  
3. Already-labeled selections (Energy + forces present) → merge into `train.cfg` and **force-retrain** (`TRAIN_FORCE=1`, `AL_RETRAIN_MAX_ITER=400`). Leftover AIMD is not capped at 50 — the full MaxVol set is merged in one pass. No new VASP.  
4. Unlabeled selections only → run VASP, put `.cfg` in `datasets/labeled/`, re-run `./run.sh --al` (the loop then continues remaining iters).  
5. Resume is cheap: existing `iter_NNN/dft_queue.cfg` is reused (no re-grade). After a *verified* retrain, `iter_NNN/merged.ok` skips that iteration. A crash mid-iter is retried (`AL_LOOP_RESTARTS`); completed stamps are skipped. A 0-byte stamp from a skipped BFGS is ignored.

A pause for unlabeled DFT is `CURRENT_STATUS=paused` (exit 10), not a failed workflow.

If refine produced `active_learning/refine/high_force_error.cfg`, that subset is already DFT-labeled and is a useful focus map for high-force local environments.

Full details: [ACTIVE_LEARNING.md](ACTIVE_LEARNING.md).

---

## 8. Per-trajectory / temperature fits

Train a separate MTP on each AIMD folder (useful for diagnostics):

```bash
./run.sh --per-traj
# or
./scripts/run_all_trajectories.sh --wait
```

Default `MAX_PARALLEL=1` so concurrent jobs do not multiply MPI ranks. Individual folder:

```bash
./scripts/train_trajectory.sh vac_W_2500_ML
```

---

## 9. Script reference

| Script | Role |
|--------|------|
| `run.sh` | Unified orchestrator (resume, refine, AL) |
| `scripts/run_workflow.sh` | Thin wrapper (`init`, `train`, `al-loop`, …) |
| `scripts/mtp_config.sh` | Shared paths, hyperparameters, `run_mlp` helpers |
| `scripts/prepare_mtp_template.py` | Build W–C template from MLIP untrained level |
| `scripts/build_initial_dataset.sh` | OUTCAR → subsampled `train.cfg` |
| `scripts/merge_cfg.py` | Concatenate / dedupe `.cfg` files |
| `scripts/subsample_cfg.py` | Keep every Nth config |
| `scripts/filter_cfg.py` | Drop bad / irrelevant DFT configs |
| `scripts/extract_high_error_cfg.py` | High force-RMSE subset for force-focused retrain |
| `scripts/train_mtp.sh` | Fit / continue MTP (MPI) |
| `scripts/refine_mtp.sh` | Continue BFGS + force retrain + high-error stages |
| `scripts/validate_mtp.py` | Threshold check on error logs |
| `scripts/cfg_label_status.py` | Report / extract already-labeled vs unlabeled `.cfg` blocks |
| `scripts/active_learning.sh` | One AL iteration (grade + select-add, MPI) |
| `scripts/run_active_learning.sh` | Multi-iteration AL driver (auto-merge labeled queues; pause if unlabeled) |
| `scripts/select_from_md.sh` | Helper to pull frames from MD |
| `scripts/run_all_trajectories.sh` | Batch per-trajectory training |
| `scripts/train_trajectory.sh` | Single-trajectory training |
| `scripts/submit_train.sh` | Cluster job submission helper |
| `scripts/run_all_temperatures.sh` | Alias → `run_all_trajectories.sh` |
| `scripts/train_temperature.sh` | Alias → `train_trajectory.sh` |

---

## 10. Environment overrides cheat sheet

```bash
export MTP_PROJECT_ROOT=/path/to/this/repo   # usually auto-detected
export MLIP_ROOT=/path/to/mlip-2
export MTP_LEVEL=22
export MTP_MAX_DIST=6.5
export FORCE_WEIGHT=0.15
export MPI_NPROCS=8
export SUBSAMPLE_STRIDE=40
export AL_SELECT_THRESHOLD=2.5
export TRAIN_FRESH=1
export AUTO_REFINE=0
export TRAIN_CFG=/custom/path/train.cfg
export TRAINED_MTP=/custom/path/potential.mtp
```

Full list: [CONFIGURATION.md](CONFIGURATION.md). Source of truth: `scripts/mtp_config.sh`.

---

## 11. Troubleshooting

| Problem | What to try |
|---------|-------------|
| `MLIP not found` | Set `MLIP_ROOT` / `MLP`; ensure `mlp` is executable |
| `execvp ... Permission denied` after multi-day run | Keep `MLP_STAGE=1`; renew `kinit` if staging disabled; check `.bin/mlp` |
| MPI TCP `POLLERR` mid-train | Default fabric is `shm`; avoid `shm:tcp` for long single-node jobs |
| Empty / missing `train.cfg` | Check `vac_W_*/OUTCAR` exist; read `datasets/initial/logs/convert.log` |
| Conversion failed | OUTCAR incomplete or not VASP format; convert manually with `mlp convert-cfg` |
| BFGS step limit / force RMS slightly high | `./run.sh --only refine` |
| Validation fails after refine | Add bulk + defect diversity; run AL; try `MTP_LEVEL=22` |
| No AL candidates | Put `.cfg` files in `datasets/candidates/` or ensure staging AIMD cfgs exist |
| AL paused for DFT on leftover AIMD | Queue should already be labeled; rerun `./run.sh --al` with the updated driver |
| AL ran BFGS instead of grade/select | After a merge the pot must be refit before the next grade. Wait for train, then AL continues |
| AL skipped BFGS after merge (`Already complete`) | Driver now sets `TRAIN_FORCE=1`; `train.cfg` newer/larger than the pot also blocks the skip. Remove a stale 0-byte `iter_*/merged.ok` if needed |
| AL re-grades a huge pool | Existing `iter_*/dft_queue.cfg` should be reused; do not delete it |
| Template regenerate loop | Ensure `MLIP_ROOT/untrained_mtps/` has the matching level file |
| “Another workflow appears to be running” | Wait for PID, or kill stale process, or `./run.sh --fresh` if state is stale |

---

## 12. What never goes to GitHub

These are ignored by `.gitignore` on purpose:

- All `OUTCAR` and VASP restart/charge files  
- `vac_W_*` trajectory directories  
- Generated `train.cfg`, staging pools, AL state, refine pots, logs  
- Staged `.bin/mlp`  

Commit only scripts, docs, `sources.conf`, and small templates so collaborators rebuild data on their machines. See [DATA_LAYOUT.md](DATA_LAYOUT.md).
