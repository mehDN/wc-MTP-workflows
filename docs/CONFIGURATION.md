# Configuration reference

All defaults live in **`scripts/mtp_config.sh`**. Scripts source that file; every variable below can be overridden in the environment **before** running `./run.sh` or any step script.

```bash
MTP_LEVEL=22 FORCE_WEIGHT=0.15 MPI_NPROCS=8 ./run.sh --skip-dataset
```

---

## MLIP installation and execution

| Variable | Default | Description |
|----------|---------|-------------|
| `MLIP_ROOT` | `$HOME/software/mlip-2` | MLIP-2 install root |
| `MLP` / `MLP_SOURCE` | `$MLIP_ROOT/bin/mlp` | Path to `mlp` executable (source before staging) |
| `MLIP_UNTRAINED_MTPS` | `$MLIP_ROOT/untrained_mtps` | Untrained level templates (e.g. `20.mtp`) |
| `MLP_STAGE` | `1` | Copy `mlp` onto the project filesystem (`.bin/mlp`) so multi-day MPI jobs still exec after Kerberos tickets for `$HOME` expire |
| `MLP_STAGED_BIN` | `$MTP_PROJECT_ROOT/.bin/mlp` | Destination of staged binary |

Helpers used by all scripts:

- `ensure_mlp` — stage / refresh the binary  
- `run_mlp` — `nice` + `mpirun -np MPI_NPROCS` with local fabric settings  
- `run_mlp_serial` — serial `mlp` under `nice` (convert-cfg, mindist, …)  
- `run_mlp_or_serial` — MPI first, fall back to serial on `Permission denied` / `execvp`  

---

## Parallel execution (MPI)

| Variable | Default | Description |
|----------|---------|-------------|
| `MPI_NPROCS` | `19` | Ranks for train / calc-grade / select-add / calc-errors |
| `MPIRUN` | `mpirun` | MPI launcher |
| `NICE_N` | `19` | `nice -n` priority |
| `MPI_EXTRA_ARGS` | *(empty)* | Extra args passed to `mpirun` |
| `I_MPI_FABRICS_LOCAL` | `shm` | Intel MPI fabric for single-node jobs (avoids long-run TCP `POLLERR` crashes) |

Serial train / AL:

```bash
MPI_NPROCS=1 ./run.sh --skip-dataset
```

Per-trajectory batch defaults `MAX_PARALLEL=1` so total ranks ≈ `MPI_NPROCS` (not `MAX_PARALLEL × MPI_NPROCS`).

---

## MTP complexity and radial basis

| Variable | Default | Description |
|----------|---------|-------------|
| `MTP_LEVEL` | `20` | Moment level; raise to `22` if defect force RMSE stays > 0.1 eV/Å after AL |
| `MTP_SPECIES_COUNT` | `2` | W and C as distinct species |
| `MTP_MIN_DIST` | `1.2` Å | Min pair distance (C–C dimer well ~1.3–1.5 Å) |
| `MTP_MAX_DIST` | `6.0` Å | Cutoff (try 5.0–6.5; covers 1st–3rd shells for WC) |
| `MTP_RADIAL_BASIS_SIZE` | `11` | Chebyshev radial functions (typical 10–12) |

Physical notes encoded in comments:

- W–C nearest neighbor ~2.1–2.2 Å  
- Lattice *a* ~2.906 Å, *c* ~2.837 Å  

---

## Fitting weights and training control

| Variable | Default | Description |
|----------|---------|-------------|
| `ENERGY_WEIGHT` | `1.0` | Energy term weight |
| `FORCE_WEIGHT` | `0.1` | Force term (typical range 0.05–0.2) |
| `STRESS_WEIGHT` | `0.005` | Stress term (typical 0.001–0.01) |
| `SCALE_BY_FORCE` | `0` | If non-zero, passed as `--scale-by-force` |
| `MAX_ITER` | `2000` | Max training iterations (initial fit) |
| `BFGS_CONV_TOL` | `1e-4` | BFGS convergence tolerance |
| `WEIGHTING` | `vibrations` | MLIP weighting scheme |
| `INIT_PARAMS` | `random` | Parameter initialization |
| `TRAIN_FRESH` | `0` | `1` = ignore fitted pots; start from untrained template |
| `TRAIN_RESUME` | `auto` | `auto` = continue from newest fitted pot; `1` = prefer curr; `0` = no auto-pick |
| `TRAIN_START_MTP` | *(unset)* | Explicit start potential path |

Training uses a **linear fit on a fixed basis** by default (nonlinear only if you change the workflow). Fitted pots are auto-discovered under `active_learning/` (`WC_L{level}_*.mtp`, `curr`, `trained`, refine stages).

---

## Refine / continue sequence

See [REFINE.md](REFINE.md) for stage details.

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTO_REFINE` | `1` | After validate fail in `run.sh`, schedule refine |
| `MAX_ITER_CONTINUE` | `2000` | BFGS iters per continue round |
| `TRAIN_CONTINUE_MAX_ROUNDS` | `3` | Max continue rounds after step limit |
| `FORCE_WEIGHT_RETRAIN` | `0.5` | Force weight for force-weighted retrain stage |
| `FORCE_RETRAIN_ENERGY_WEIGHT` | `ENERGY_WEIGHT` | Energy weight in force retrain |
| `FORCE_RETRAIN_STRESS_WEIGHT` | `STRESS_WEIGHT` | Stress weight in force retrain |
| `FORCE_RETRAIN_MAX_ITER` | `1500` | Max iters for force retrain |
| `ENABLE_HIGH_ERROR_RETRAIN` | `1` | High force-error subset + polish stage |
| `HIGH_ERROR_TOP_FRAC` | `0.15` | Top fraction by per-config force RMSE |
| `HIGH_ERROR_MIN_FORCE_RMSE` | `0.15` | Absolute force RMSE floor (eV/Å) |
| `HIGH_ERROR_ENERGY_WEIGHT` | `0.1` | Weights on high-error subset fit |
| `HIGH_ERROR_FORCE_WEIGHT` | `1.0` | |
| `HIGH_ERROR_STRESS_WEIGHT` | `0.0` | |
| `HIGH_ERROR_MAX_ITER` | `500` | Subset fit iters |
| `HIGH_ERROR_POLISH_MAX_ITER` | `1000` | Full-set polish iters |
| `FILTER_BAD_DFT` | `1` | Filter train set before refine |
| `FILTER_MAX_FORCE` | `50.0` | Drop configs with larger \|F\| (eV/Å) |
| `FILTER_MIN_DIST_ABS` | `0.50` | Drop short mindist (Å) |
| `FILTER_MAX_EPA_OUTLIER` | `5.0` | Drop \|E/atom − median\| outliers (eV) |
| `AUTO_ESCALATE_LEVEL` | `0` | If refine still fails, retrain at `ESCALATE_LEVEL` |
| `ESCALATE_LEVEL` | `22` | Target level for auto-escalate |

---

## Workflow resume

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTO_RESUME` | `1` | Skip steps already completed on disk |
| `WORKFLOW_STATE_FILE` | `$MTP_AL_DIR/workflow_state.env` | Step status written by `run.sh` |

CLI: `--fresh` / `--no-resume` disable auto-skip; `--resume` forces it on.

---

## Dataset subsampling

| Variable | Default | Description |
|----------|---------|-------------|
| `SUBSAMPLE_STRIDE` | `50` | Keep every Nth AIMD frame (T ≤ threshold) |
| `HIGH_T_STRIDE` | `25` | Denser sampling for high-T trajectories |
| `HIGH_T_THRESHOLD_K` | `1800` | Temperatures above this use `HIGH_T_STRIDE` |

AIMD folders and temperatures (`AIMD_SOURCES` / `AIMD_SOURCE_TEMPS`) are set inside `mtp_config.sh`:

| Folder | T (K) |
|--------|------:|
| `vac_W_2300` | 2300 |
| `vac_W_2500_ML` | 2500 |
| `vac_W_2500_small_ML` | 2500 |
| `vac_W_2800_ML` | 2800 |
| `vac_W_2800_small_ML` | 2800 |
| `vac_W_2800_betaWC` | 2800 (temp map only; not in default `AIMD_SOURCES`) |

Edit the arrays in `mtp_config.sh` to add or remove trajectories.

External sources: `SOURCES_CONF` → default `datasets/sources.conf`.

---

## Active learning

If the candidate pool is leftover AIMD staging frames, selected configs usually already have VASP `Energy` + forces. The driver merges those into `train.cfg` and retrains. New DFT is requested only when a selection has no Energy + forces.

A cfg is **labeled** when it has a numeric `Energy` and `AtomData` force columns (`fx`/`fy`/`fz`). Detection: `scripts/cfg_label_status.py` (also `al_cfg_label_status` in `mtp_config.sh`).

| Variable | Default | Description |
|----------|---------|-------------|
| `AL_INIT_THRESHOLD` | `1e-5` | Init threshold for grade / select-add |
| `AL_SELECT_THRESHOLD` | `3.0` | Add configs with grade above this (try 2.0–4.0) |
| `AL_SWAP_THRESHOLD` | `1.0000001` | Swap threshold for maxvol |
| `AL_GRADE_THRESHOLD` | same as select | Alias for grade threshold |
| `AL_SELECTION_LIMIT` | `50` | Max structures per select-add (`0` = unlimited) |
| `AL_MAX_ITERATIONS` | `20` | Cap for `run_active_learning.sh` |
| `AL_PREFER_HIGH_FORCE_ERROR` | `1` | Announce refine high-error subset as AL focus material |
| `AL_PAUSE_EXIT` | `10` | Exit code when unlabeled selections still need VASP (`run.sh` records `CURRENT_STATUS=paused`) |
| `AL_CANDIDATE_CFG` | *(unset)* | Explicit candidate pool path |
| `AL_LABELED_CFG` | *(unset)* | Explicit labeled cfg path |
| `AL_CANDIDATES_DIR` | `datasets/candidates` | Auto-discover `*.cfg` here |
| `AL_LABELED_DIR` | `datasets/labeled` | Auto-discover newly labeled `*.cfg` after a pause |
| `AL_MERGED_CANDIDATES` | `.../candidates/_merged_pool.cfg` | Merge of multiple pools |

Resume artifacts under `active_learning/iter_NNN/`:

| File | Role |
|------|------|
| `dft_queue.cfg` | Selected configs; reused on resume (grade/select skipped) |
| `unlabeled_queue.cfg` | Split remainder that still needs DFT (mixed queues) |
| `merged.ok` | Stamp written after a successful merge + retrain |

If no `datasets/candidates/*.cfg` exists, the fallback pool is the full AIMD staging set (`datasets/initial/staging/aimd_*.cfg`, not the stride-subsampled train set). That pool is cached as `datasets/candidates/_aimd_staging_pool.cfg` and rebuilt only when staging files are newer.

`workflow_infer_resume_point` treats `CURRENT_STATUS=paused` like `running`/`failed` so `./run.sh --al` continues the AL step.

---

## Validation thresholds

| Variable | Default | Unit |
|----------|---------|------|
| `VAL_FORCE_RMS_MAX` | `0.08` | eV/Å |
| `VAL_FORCE_MAE_MAX` | `0.08` | eV/Å |
| `VAL_ENERGY_PER_ATOM_MAX` | `0.005` | eV/atom |
| `VAL_STRESS_RMS_MAX` | `0.5` | GPa |
| `VAL_DEFECT_FORCE_RMS_MAX` | `0.1` | eV/Å (suggest raising MTP level) |

`validate_mtp.py` can take the same limits as CLI flags (`--force-rms-max`, etc.).

---

## Paths

| Variable | Default |
|----------|---------|
| `MTP_PROJECT_ROOT` | Parent of `scripts/` (auto-detected) |
| `MTP_TEMPLATES_DIR` | `$MTP_PROJECT_ROOT/templates` |
| `MTP_DATASETS_DIR` | `$MTP_PROJECT_ROOT/datasets` |
| `MTP_AL_DIR` | `$MTP_PROJECT_ROOT/active_learning` |
| `MTP_LOGS_DIR` | `$MTP_PROJECT_ROOT/logs` |
| `MTP_TEMPLATE` | `$MTP_TEMPLATES_DIR/WC_L${MTP_LEVEL}.mtp` |
| `TRAIN_CFG` | `$MTP_DATASETS_DIR/initial/train.cfg` |
| `VALID_CFG` | `$MTP_DATASETS_DIR/validation/holdout.cfg` |
| `TRAINED_MTP` | `$MTP_AL_DIR/WC_L${MTP_LEVEL}_trained.mtp` |
| `ALS_FILE` | `$MTP_AL_DIR/state.als` |
| `SOURCES_CONF` | `$MTP_DATASETS_DIR/sources.conf` |
| `WORKFLOW_STATE_FILE` | `$MTP_AL_DIR/workflow_state.env` |

### Per-trajectory batch

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_PARALLEL` | `1` | Concurrent jobs in `run_all_trajectories.sh` (keeps total MPI ranks ≈ `MPI_NPROCS`) |

---

## Recommended tuning order

1. Build a diverse initial set (bulk + defect + high-T AIMD).  
2. Train level **20** (MPI), check force RMS.  
3. If BFGS hits step limit or force RMS fails → let **auto-refine** run (or `./run.sh --only refine`).  
4. Run AL. Leftover AIMD frames already in staging are labeled and will be merged automatically; dump new MTP-MD frames to `datasets/candidates/` when you need extra unlabeled exploration.  
5. If defect forces stay > ~0.1 eV/Å after several AL rounds → `MTP_LEVEL=22` and/or slightly larger cutoff / force weight.  
6. Only then explore `FORCE_WEIGHT`, `MTP_MAX_DIST`, and radial basis size.
