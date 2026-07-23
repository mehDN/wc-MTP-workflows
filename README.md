# WC W-Vacancy MTP Workflows

Unified **Moment Tensor Potential (MTP)** training and active-learning pipelines for tungsten carbide (α-WC) **W-vacancy reconstruction** (planar C–C dimer pathway).

This repository contains orchestration scripts, configuration, and lightweight templates. It does **not** include large VASP `OUTCAR` trajectories or generated training sets (those stay local).

## What is included

| Path | Description |
|------|-------------|
| `run.sh` | Main entry point: template → dataset → train → validate → optional AL |
| `scripts/` | Pipeline steps (dataset build, train, AL, per-trajectory fits) |
| `templates/` | Prebuilt W–C MTP templates (levels 20 and 22) |
| `datasets/sources.conf` | External DFT source list and composition targets |
| `docs/HOW_TO_USE.md` | Step-by-step usage guide |

## What is **not** included (by design)

- `OUTCAR` / VASP run directories (`vac_W_*`, etc.)
- Built `train.cfg` / staging `.cfg` files
- Trained potentials and active-learning iteration state
- Runtime logs

Place your own DFT data next to the scripts (or list paths in `datasets/sources.conf`) and rebuild locally.

## Prerequisites

1. **[MLIP-2](https://gitlab.com/ashapeev/mlip-2)** installed with `mlp` on your path (or set `MLIP_ROOT`).
2. **Python 3** (stdlib only for helper scripts).
3. **VASP** (or another DFT code) for labeling active-learning candidates.
4. AIMD / static DFT data for WC and W-vacancies (local only).

Default MLIP location:

```bash
export MLIP_ROOT="${HOME}/software/mlip-2"
export MLP="${MLIP_ROOT}/bin/mlp"
```

## Quick start

```bash
# Clone
git clone https://github.com/<YOUR_USER>/wc-mtp-workflows.git
cd wc-mtp-workflows

# Point at your MLIP install if needed
export MLIP_ROOT=/path/to/mlip-2

# Put AIMD OUTCARs in folders matching mtp_config.sh, e.g.:
#   vac_W_2300/OUTCAR
#   vac_W_2500_ML/OUTCAR
# or list extra sources in datasets/sources.conf

# Full pipeline: dataset → train → validate
./run.sh

# With active learning (after dropping candidates into datasets/candidates/)
./run.sh --al
```

## Pipeline overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────┐    ┌──────────┐
│ 1. Template │ -> │ 2. Dataset   │ -> │ 3. Train│ -> │ 4. Valid │
│  L20 / L22  │    │  OUTCAR→cfg  │    │  mlp    │    │  errors  │
└─────────────┘    └──────────────┘    └─────────┘    └──────────┘
                                              │
                     ┌────────────────────────┘
                     v
              ┌──────────────┐    DFT labels     ┌────────────┐
              │ 5. Active    │ ----------------> │ retrain    │
              │    learning  │ <---------------- │ merge cfg  │
              └──────────────┘                   └────────────┘
```

Optional step 6: per-AIMD-trajectory MTP fits (`--per-traj`).

## Common commands

```bash
./run.sh --help
./run.sh                          # dataset + train + validate
./run.sh --skip-dataset           # train existing train.cfg
./run.sh --only dataset           # build dataset only
./run.sh --al                     # train + active learning
./run.sh --al --candidates path/to/pool.cfg
./run.sh --per-traj               # also fit per-trajectory MTPs

# Environment overrides
MTP_LEVEL=22 MTP_MAX_DIST=6.5 FORCE_WEIGHT=0.15 ./run.sh
```

Legacy wrapper (same options via short commands):

```bash
./scripts/run_workflow.sh init
./scripts/run_workflow.sh train
./scripts/run_workflow.sh al-loop datasets/candidates/md_frames.cfg
```

## Default hyperparameters

Defined in `scripts/mtp_config.sh` (all overridable via environment):

| Parameter | Default | Notes |
|-----------|---------|--------|
| `MTP_LEVEL` | 20 | Use 22 if defect force RMSE stays > 0.1 eV/Å |
| `MTP_MAX_DIST` | 6.0 Å | Cutoff (try 5.0–6.5) |
| `MTP_RADIAL_BASIS_SIZE` | 11 | Chebyshev radial functions |
| `FORCE_WEIGHT` | 0.1 | Energy / force / stress weights |
| `AL_SELECT_THRESHOLD` | 3.0 | Maxvol grade for select-add |
| Force RMS target | ≤ 0.08 eV/Å | Validation gate |

## Documentation

See **[docs/HOW_TO_USE.md](docs/HOW_TO_USE.md)** for:

- Dataset layout and `sources.conf` format  
- Active-learning loop (select → DFT label → merge → retrain)  
- Per-trajectory training  
- Troubleshooting  

## License / citation

Research workflow scripts. Cite MLIP/MTP (Shapeev et al.) and your DFT settings (PBE, ENCUT 450–500 eV recommended in `sources.conf`) when publishing results.
