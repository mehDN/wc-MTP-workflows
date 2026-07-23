# Configuration reference

All defaults live in **`scripts/mtp_config.sh`**. Scripts source that file; every variable below can be overridden in the environment **before** running `./run.sh` or any step script.

```bash
MTP_LEVEL=22 FORCE_WEIGHT=0.15 ./run.sh --skip-dataset
```

---

## MLIP installation

| Variable | Default | Description |
|----------|---------|-------------|
| `MLIP_ROOT` | `$HOME/software/mlip-2` | MLIP-2 install root |
| `MLP` | `$MLIP_ROOT/bin/mlp` | Path to `mlp` executable |
| `MLIP_UNTRAINED_MTPS` | `$MLIP_ROOT/untrained_mtps` | Untrained level templates (e.g. `20.mtp`) |

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
| `MAX_ITER` | `2000` | Max training iterations |
| `BFGS_CONV_TOL` | `1e-4` | BFGS convergence tolerance |
| `WEIGHTING` | `vibrations` | MLIP weighting scheme |
| `INIT_PARAMS` | `random` | Parameter initialization |

Training uses a **linear fit on a fixed basis** by default (nonlinear only if you change the workflow).

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

| Variable | Default | Description |
|----------|---------|-------------|
| `AL_INIT_THRESHOLD` | `1e-5` | Init threshold for grade / select-add |
| `AL_SELECT_THRESHOLD` | `3.0` | Add configs with grade above this (try 2.0–4.0) |
| `AL_SWAP_THRESHOLD` | `1.0000001` | Swap threshold for maxvol |
| `AL_GRADE_THRESHOLD` | same as select | Alias for grade threshold |
| `AL_SELECTION_LIMIT` | `50` | Max structures per select-add (`0` = unlimited) |
| `AL_MAX_ITERATIONS` | `20` | Cap for `run_active_learning.sh` |
| `AL_CANDIDATE_CFG` | *(unset)* | Explicit candidate pool path |
| `AL_LABELED_CFG` | *(unset)* | Explicit labeled cfg path |
| `AL_CANDIDATES_DIR` | `datasets/candidates` | Auto-discover `*.cfg` here |
| `AL_LABELED_DIR` | `datasets/labeled` | Auto-discover labeled `*.cfg` |
| `AL_MERGED_CANDIDATES` | `.../candidates/_merged_pool.cfg` | Merge of multiple pools |

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

### Per-trajectory batch

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_PARALLEL` | `3` | Concurrent jobs in `run_all_trajectories.sh` |

---

## Recommended tuning order

1. Build a diverse initial set (bulk + defect + high-T AIMD).  
2. Train level **20**, check force RMS on train (and holdout if present).  
3. Run AL around the unreconstructed vacancy / dimer pathway.  
4. If defect forces stay > ~0.1 eV/Å after several AL rounds → `MTP_LEVEL=22` and/or slightly larger cutoff / force weight.  
5. Only then explore `FORCE_WEIGHT`, `MTP_MAX_DIST`, and radial basis size.
