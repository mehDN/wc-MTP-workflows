# WC W-Vacancy MTP Workflows

Orchestration scripts for **Moment Tensor Potential (MTP)** training and active learning on tungsten carbide (α-WC) **W-vacancy reconstruction** (planar C–C dimer pathway).

Repository: [github.com/mehDN/wc-MTP-workflows](https://github.com/mehDN/wc-MTP-workflows)

This repo ships **pipelines, config, and small MTP templates**. Large VASP trajectories, built training sets, and trained potentials stay local (see [What is not committed](#what-is-not-committed)).

## Features

- One-command pipeline: template → dataset → train → validate
- Optional **active learning** (maxvol / extrapolation grade → DFT label → retrain)
- Optional **per-trajectory** MTP fits for diagnostics
- Shared hyperparameters in `scripts/mtp_config.sh` (all overridable via env)
- External DFT sources via `datasets/sources.conf` with composition categories

## Repository layout

```text
wc-MTP-workflows/
├── run.sh                      # Main entry point
├── README.md
├── docs/
│   ├── HOW_TO_USE.md           # Step-by-step usage
│   ├── CONFIGURATION.md        # Hyperparameters & env overrides
│   ├── ACTIVE_LEARNING.md      # AL loop details
│   └── DATA_LAYOUT.md          # Paths, gitignore, local data
├── scripts/
│   ├── mtp_config.sh           # Shared config (source of truth)
│   ├── build_initial_dataset.sh
│   ├── train_mtp.sh
│   ├── validate_mtp.py
│   ├── active_learning.sh
│   ├── run_active_learning.sh
│   └── ...
├── templates/
│   ├── WC_L20.mtp              # Level-20 W–C template
│   └── WC_L22.mtp              # Level-22 (if defect forces stay high)
├── datasets/
│   ├── sources.conf            # Extra DFT sources (committed)
│   ├── candidates/             # AL candidate pools (local)
│   └── labeled/                # DFT-labeled AL configs (local)
└── active_learning/            # Trained MTPs & AL state (local)
```

## Prerequisites

| Dependency | Role |
|------------|------|
| [MLIP-2](https://gitlab.com/ashapeev/mlip-2) | `mlp` binary on `PATH` (or set `MLIP_ROOT`) |
| Python 3 | Stdlib-only helpers (`merge_cfg.py`, `validate_mtp.py`, …) |
| VASP (or equivalent DFT) | Label AL selections; produce AIMD `OUTCAR`s |

```bash
export MLIP_ROOT="${HOME}/software/mlip-2"
export MLP="${MLIP_ROOT}/bin/mlp"
export PATH="${MLIP_ROOT}/bin:${PATH}"
which mlp
```

## Quick start

```bash
git clone https://github.com/mehDN/wc-MTP-workflows.git
cd wc-MTP-workflows
chmod +x run.sh scripts/*.sh scripts/*.py

# Place AIMD OUTCARs next to run.sh (names match mtp_config.sh), e.g.:
#   vac_W_2300/OUTCAR
#   vac_W_2500_ML/OUTCAR
# Or list extra paths in datasets/sources.conf

./run.sh                 # dataset → train → validate
./run.sh --al            # same + active-learning selection
```

## Pipeline overview

```text
  Template (L20/L22)  →  Dataset (OUTCAR→cfg)  →  Train (mlp)  →  Validate
                                                              │
                                                              ▼
                                                    Active learning
                                                    (select → DFT → merge → retrain)
```

| Step | What runs | Main outputs |
|------|-----------|--------------|
| 1. Template | Ensure `templates/WC_L{level}.mtp` | Template file |
| 2. Dataset | Convert AIMD + `sources.conf` | `datasets/initial/train.cfg` |
| 3. Train | Linear MTP fit | `active_learning/WC_L20_trained.mtp` |
| 4. Validate | Parse `calc-errors` vs thresholds | Pass/fail gate |
| 5. AL (opt.) | Grade + select-add | `active_learning/iter_*/dft_queue.cfg` |
| 6. Per-traj (opt.) | One MTP per AIMD folder | Per-folder potentials |

## Common commands

```bash
./run.sh --help
./run.sh                          # full: dataset + train + validate
./run.sh --skip-dataset           # retrain existing train.cfg
./run.sh --only dataset           # build dataset only
./run.sh --al                     # train + AL
./run.sh --al --candidates path/to/pool.cfg
./run.sh --labeled datasets/labeled/new.cfg   # merge labels before retrain
./run.sh --per-traj               # also fit per-AIMD-trajectory MTPs

# Hyperparameter overrides
MTP_LEVEL=22 MTP_MAX_DIST=6.5 FORCE_WEIGHT=0.15 ./run.sh
```

Legacy wrapper:

```bash
./scripts/run_workflow.sh init
./scripts/run_workflow.sh train
./scripts/run_workflow.sh al-loop datasets/candidates/md_frames.cfg
```

## Default hyperparameters

From `scripts/mtp_config.sh` (see [docs/CONFIGURATION.md](docs/CONFIGURATION.md)):

| Parameter | Default | Notes |
|-----------|---------|--------|
| `MTP_LEVEL` | 20 | Use 22 if defect force RMSE stays > 0.1 eV/Å |
| `MTP_MAX_DIST` | 6.0 Å | Cutoff (try 5.0–6.5) |
| `MTP_RADIAL_BASIS_SIZE` | 11 | Chebyshev radial functions |
| `FORCE_WEIGHT` | 0.1 | Energy / force / stress weights |
| `AL_SELECT_THRESHOLD` | 3.0 | Maxvol grade for select-add |
| Force RMS target | ≤ 0.08 eV/Å | Validation gate |

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/HOW_TO_USE.md](docs/HOW_TO_USE.md) | Full walkthrough: setup, train, AL, scripts |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | All env vars and defaults |
| [docs/ACTIVE_LEARNING.md](docs/ACTIVE_LEARNING.md) | Select → label → merge → retrain loop |
| [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md) | Local data paths and what Git ignores |

## What is not committed

Ignored by `.gitignore` on purpose:

- VASP `OUTCAR` / charge / wavefunction files and `vac_W_*` trajectory folders  
- Generated `train.cfg`, staging pools, logs  
- Trained potentials and active-learning iteration state  

Collaborators rebuild datasets and potentials on their machines from local DFT data.

## Citation

Research workflow scripts. When publishing results, cite **MLIP/MTP** (Shapeev et al.) and your DFT settings (PBE, ENCUT 450–500 eV recommended in `datasets/sources.conf`).
