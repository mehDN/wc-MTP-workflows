# How to use the WC MTP pipelines

Step-by-step setup, the main `run.sh` workflow, active learning, and script reference.

Related docs:

- [CONFIGURATION.md](CONFIGURATION.md) — hyperparameters and environment overrides  
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

### Python

Scripts use **Python 3** with the standard library only:

- `merge_cfg.py`, `subsample_cfg.py`, `validate_mtp.py`, `prepare_mtp_template.py`

### Local DFT data (not in git)

Keep large VASP outputs **outside git**. Typical layout next to `run.sh`:

```text
wc-MTP-workflows/
  run.sh
  scripts/
  templates/
  datasets/
    sources.conf
    candidates/          # MD frames for AL (you create)
    labeled/             # DFT-labeled .cfg from AL (you create)
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
3. **Train** — linear MTP fit → `active_learning/WC_L20_trained.mtp`.  
4. **Validate** — parse `mlp calc-errors` logs against thresholds.  

Logs go to `logs/run_YYYYMMDD_HHMMSS.log`.

### Useful flags

| Flag | Effect |
|------|--------|
| `--skip-dataset` | Reuse existing `train.cfg` |
| `--skip-train` | Build dataset only (no fit) |
| `--skip-validate` | Skip error thresholds |
| `--al` | After train, run active-learning selection |
| `--candidates FILE` | Candidate pool for AL (implies `--al`) |
| `--labeled FILE` | Merge new DFT labels into `train.cfg` before retrain |
| `--per-traj` | Also train one MTP per AIMD folder |
| `--only STEP` | `template` \| `dataset` \| `train` \| `validate` \| `al` \| `per-traj` |
| `-h` / `--help` | Full help |

Examples:

```bash
./run.sh --only dataset
./run.sh --skip-dataset
MTP_LEVEL=22 ./run.sh --skip-dataset
./run.sh --al --candidates datasets/candidates/md_frames.cfg
```

---

## 3. Building the training set

```bash
./scripts/build_initial_dataset.sh
# or
./run.sh --only dataset
```

### What it does

1. Converts each `vac_W_*/OUTCAR` to staging `.cfg` via `mlp convert-cfg --input-format=vasp-outcar`.  
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
```

Increase level if defect forces remain bad after active learning:

```bash
MTP_LEVEL=22 ./run.sh --skip-dataset
```

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

If defect-region force RMSE stays above ~0.1 eV/Å, add AL data or raise `MTP_LEVEL`.

---

## 6. Active-learning loop

Goal: select structures where the MTP extrapolates (high maxvol grade), label with DFT, merge, retrain—until the reconstructed C–C dimer is stable (~3–4 eV lowering vs unreconstructed vacancy).

Short cycle:

1. Dump candidate frames as MLIP `.cfg` into `datasets/candidates/`.  
2. `./run.sh --skip-dataset --al`  
3. Label selected configs with VASP → put `.cfg` in `datasets/labeled/`.  
4. `./run.sh --skip-dataset --labeled datasets/labeled/my_new_labels.cfg` then AL again.  

Full details: [ACTIVE_LEARNING.md](ACTIVE_LEARNING.md).

---

## 7. Per-trajectory / temperature fits

Train a separate MTP on each AIMD folder (useful for diagnostics):

```bash
./run.sh --per-traj
# or
./scripts/run_all_trajectories.sh --wait
```

Individual folder:

```bash
./scripts/train_trajectory.sh vac_W_2500_ML
```

---

## 8. Script reference

| Script | Role |
|--------|------|
| `run.sh` | Unified orchestrator |
| `scripts/run_workflow.sh` | Thin wrapper (`init`, `train`, `al-loop`, …) |
| `scripts/mtp_config.sh` | Shared paths and hyperparameters |
| `scripts/prepare_mtp_template.py` | Build W–C template from MLIP untrained level |
| `scripts/build_initial_dataset.sh` | OUTCAR → subsampled `train.cfg` |
| `scripts/merge_cfg.py` | Concatenate / dedupe `.cfg` files |
| `scripts/subsample_cfg.py` | Keep every Nth config |
| `scripts/train_mtp.sh` | Fit MTP |
| `scripts/validate_mtp.py` | Threshold check on error logs |
| `scripts/active_learning.sh` | One AL iteration (grade + select-add) |
| `scripts/run_active_learning.sh` | Multi-iteration AL driver |
| `scripts/select_from_md.sh` | Helper to pull frames from MD |
| `scripts/run_all_trajectories.sh` | Batch per-trajectory training |
| `scripts/train_trajectory.sh` | Single-trajectory training |
| `scripts/submit_train.sh` | Cluster job submission helper |
| `scripts/run_all_temperatures.sh` | Alias → `run_all_trajectories.sh` |
| `scripts/train_temperature.sh` | Alias → `train_trajectory.sh` |

---

## 9. Environment overrides cheat sheet

```bash
export MTP_PROJECT_ROOT=/path/to/this/repo   # usually auto-detected
export MLIP_ROOT=/path/to/mlip-2
export MTP_LEVEL=22
export MTP_MAX_DIST=6.5
export FORCE_WEIGHT=0.15
export SUBSAMPLE_STRIDE=40
export AL_SELECT_THRESHOLD=2.5
export TRAIN_CFG=/custom/path/train.cfg
export TRAINED_MTP=/custom/path/potential.mtp
```

Full list: [CONFIGURATION.md](CONFIGURATION.md). Source of truth: `scripts/mtp_config.sh`.

---

## 10. Troubleshooting

| Problem | What to try |
|---------|-------------|
| `MLIP not found` | Set `MLIP_ROOT` / `MLP`; ensure `mlp` is executable |
| Empty / missing `train.cfg` | Check `vac_W_*/OUTCAR` exist; read `datasets/initial/logs/convert.log` |
| Conversion failed | OUTCAR incomplete or not VASP format; convert manually with `mlp convert-cfg` |
| Validation fails | Add bulk + defect diversity; run AL; try `MTP_LEVEL=22` |
| No AL candidates | Put `.cfg` files in `datasets/candidates/` or ensure staging AIMD cfgs exist |
| Template regenerate loop | Ensure `MLIP_ROOT/untrained_mtps/` has the matching level file |

---

## 11. What never goes to GitHub

These are ignored by `.gitignore` on purpose:

- All `OUTCAR` and VASP restart/charge files  
- `vac_W_*` trajectory directories  
- Generated `train.cfg`, staging pools, AL state, logs  

Commit only scripts, docs, `sources.conf`, and small templates so collaborators rebuild data on their machines. See [DATA_LAYOUT.md](DATA_LAYOUT.md).
