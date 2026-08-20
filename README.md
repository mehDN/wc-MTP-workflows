# WC W-Vacancy MTP Workflows

Orchestration scripts for **Moment Tensor Potential (MTP)** training and active learning on tungsten carbide (α-WC) **W-vacancy reconstruction** (planar C–C dimer pathway).

Repository: [github.com/mehDN/wc-MTP-workflows](https://github.com/mehDN/wc-MTP-workflows)

This repo ships **pipelines, config, and small MTP templates**. Large VASP trajectories, built training sets, and trained potentials stay local (see [What is not committed](#what-is-not-committed)).

## Features

- One-command pipeline: template → dataset → train → validate
- **Auto-resume** after crashes (`workflow_state.env`; skip completed steps)
- **MPI-parallel** train / grade / select-add (`MPI_NPROCS`, default 19)
- **MLP staging** onto project filesystem (survives multi-day jobs when `$HOME` NFS+krb5 tickets expire)
- **Auto-refine** when validation fails: more BFGS + rescale, force-weighted retrain, high-error subset, mindist check
- Optional **active learning** based on the MLIP **MaxVol / extrapolation-grade** strategy (D-optimality)
  - Selects configurations that most expand the training coverage in descriptor space
  - If a selection already has VASP `Energy` + forces (leftover AIMD/OUTCAR frames), it is **merged and retrained** — no new DFT
  - Retrain after a merge always runs BFGS (`TRAIN_FORCE=1`, `AL_RETRAIN_MAX_ITER=400`); an existing pot is not treated as done, and `merged.ok` is written only after the pot matches the new `train.cfg`
  - Already-labeled leftover AIMD is not capped at 50 (`AL_LABELED_SELECTION_LIMIT=0`) so one select-add can merge the full MaxVol set
  - New VASP is requested only for unlabeled MD/exploratory frames; that pause is recorded as `paused`, not a crash
  - Resume reuses `iter_NNN/dft_queue.cfg` so grade/select is not rerun
  - See [docs/ACTIVE_LEARNING.md](docs/ACTIVE_LEARNING.md) for the loop, MaxVol rationale, and when DFT is actually needed
- Optional **per-trajectory** MTP fits for diagnostics
- Shared hyperparameters in `scripts/mtp_config.sh` (all overridable via env)
- External DFT sources via `datasets/sources.conf` with composition categories

## Repository layout

```text
wc-MTP-workflows/
├── run.sh                      # Main entry point (resume + refine aware)
├── README.md
├── docs/
│   ├── HOW_TO_USE.md           # Step-by-step usage
│   ├── CONFIGURATION.md        # Hyperparameters & env overrides
│   ├── ACTIVE_LEARNING.md      # AL loop details
│   ├── REFINE.md               # Post-train refine sequence
│   └── DATA_LAYOUT.md          # Paths, gitignore, local data
├── scripts/
│   ├── mtp_config.sh           # Shared config (source of truth)
│   ├── build_initial_dataset.sh
│   ├── train_mtp.sh            # Fit / continue from fitted pot
│   ├── refine_mtp.sh           # BFGS continue + force retrain + high-error
│   ├── filter_cfg.py           # Drop bad / irrelevant DFT configs
│   ├── extract_high_error_cfg.py
│   ├── validate_mtp.py
│   ├── active_learning.sh
│   ├── run_active_learning.sh
│   ├── cfg_label_status.py     # Detect Energy+forces on AL queues
│   └── ...
├── templates/
│   ├── WC_L20.mtp              # Level-20 W–C template
│   └── WC_L22.mtp              # Level-22 (if defect forces stay high)
├── datasets/
│   ├── sources.conf            # Extra DFT sources (committed)
│   ├── candidates/             # optional new MD pools (local)
│   └── labeled/                # new DFT labels after an unlabeled AL pause
└── active_learning/            # Trained MTPs, refine/, state (local)
```

## Prerequisites

| Dependency | Role |
|------------|------|
| [MLIP-2](https://gitlab.com/ashapeev/mlip-2) | `mlp` binary (set `MLIP_ROOT` / `MLP`) |
| MPI (`mpirun`) | Parallel train / AL (optional serial with `MPI_NPROCS=1`) |
| Python 3 | Stdlib-only helpers (`merge_cfg.py`, `filter_cfg.py`, `validate_mtp.py`, `cfg_label_status.py`, …) |
| VASP (or equivalent DFT) | Produce AIMD `OUTCAR`s; label **unlabeled** AL selections only |

```bash
export MLIP_ROOT="${HOME}/software/mlip-2"
export MLP="${MLIP_ROOT}/bin/mlp"
export PATH="${MLIP_ROOT}/bin:${PATH}"
which mlp
which mpirun
```

On clusters where `$HOME` is NFS with Kerberos, the workflow **stages** `mlp` into `.bin/mlp` under the project tree by default (`MLP_STAGE=1`) so multi-day `mpirun` jobs still exec after tickets expire.

## Quick start

```bash
git clone https://github.com/mehDN/wc-MTP-workflows.git
cd wc-MTP-workflows
chmod +x run.sh scripts/*.sh scripts/*.py

# Place AIMD OUTCARs next to run.sh (names match mtp_config.sh), e.g.:
#   vac_W_2300/OUTCAR
#   vac_W_2500_ML/OUTCAR
# Or list extra paths in datasets/sources.conf

./run.sh                 # dataset → train → validate (+ auto-refine if needed)
./run.sh --al            # same + AL loop (all iters until cap or pool exhausted)
./run.sh                 # re-run after crash or AL pause: auto-resumes
```

## Pipeline overview

```text
  Template (L20/L22)  →  Dataset (OUTCAR→cfg)  →  Train (MPI mlp)  →  Validate
                                                              │
                         ┌────────────────────────────────────┤ fail / step limit
                         ▼                                    │
                    Refine (filter → continue BFGS →          │
                     force retrain → high-error → mindist)    │
                                                              ▼
                                                    Active learning
                                                    (select → merge if already labeled
                                                     else DFT → merge → retrain)
```

| Step | What runs | Main outputs |
|------|-----------|--------------|
| 1. Template | Ensure `templates/WC_L{level}.mtp` | Template file |
| 2. Dataset | Convert AIMD + `sources.conf` | `datasets/initial/train.cfg` |
| 3. Train | Linear MTP fit (continues fitted pot if found) | `active_learning/WC_L20_trained.mtp` |
| 4. Validate | Parse `calc-errors` vs thresholds | Pass/fail gate |
| 4b. Refine | More BFGS, force retrain, high-error subset | `active_learning/refine/*.mtp` |
| 5. AL (opt.) | Grade + select-add; merge already-labeled queues and **force-retrain** | `iter_*/dft_queue.cfg`, updated `train.cfg` + refit pot |
| 6. Per-traj (opt.) | One MTP per AIMD folder | Per-folder potentials |

### Resume behavior

By default (`AUTO_RESUME=1`), re-running `./run.sh` after a crash:

- Skips steps already complete (dataset present, pot trained + errors logged, …)
- Continues train post-processing if BFGS finished but `calc-errors` died
- Restores planned `--al` / `--per-traj` / refine intents from `active_learning/workflow_state.env`
- If AL paused for unlabeled DFT (`CURRENT_STATUS=paused`), re-running `--al` continues from the existing queue
- Existing `iter_NNN/dft_queue.cfg` is reused (no re-grade); `iter_NNN/merged.ok` skips a finished iteration only if the pot still matches the current `train.cfg` (stale stamps from a skipped BFGS are ignored)

Force a full re-plan: `./run.sh --fresh`. Force random-init train (ignore fitted pots): `TRAIN_FRESH=1 ./run.sh --skip-dataset`.

## Common commands

```bash
./run.sh --help
./run.sh                          # full: dataset + train + validate (+ auto-refine)
./run.sh                          # after crash: resume incomplete step
./run.sh --fresh                  # ignore resume skips; re-run selected steps
./run.sh --skip-dataset           # retrain / continue existing train.cfg
./run.sh --only dataset           # build dataset only
./run.sh --only refine            # continue BFGS / force retrain from existing pot
./run.sh --refine                 # force refine after train
./run.sh --skip-refine            # never auto-refine on validate fail
./run.sh --al                     # AL loop: each iter merge+retrain then the next, up to 20
./run.sh --al --candidates path/to/pool.cfg
./run.sh --labeled datasets/labeled/new.cfg   # merge new DFT labels before retrain
./run.sh --per-traj               # also fit per-AIMD-trajectory MTPs

# Hyperparameter / parallel overrides
MTP_LEVEL=22 MTP_MAX_DIST=6.5 FORCE_WEIGHT=0.15 ./run.sh
MPI_NPROCS=8 ./run.sh --skip-dataset
TRAIN_FRESH=1 ./run.sh --skip-dataset   # random init; ignore trained pots
```

Direct refine (without full orchestrator):

```bash
./scripts/refine_mtp.sh
./scripts/refine_mtp.sh datasets/initial/train.cfg active_learning/WC_L20_trained.mtp
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
| `MPI_NPROCS` | 19 | Ranks for train / AL |
| `AUTO_RESUME` | 1 | Skip completed workflow steps |
| `AUTO_REFINE` | 1 | Run refine when validation fails |
| `AL_SELECT_THRESHOLD` | 3.0 | Maxvol grade for select-add |
| Force RMS target | ≤ 0.08 eV/Å | Validation gate |

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/HOW_TO_USE.md](docs/HOW_TO_USE.md) | Full walkthrough: setup, train, resume, refine, AL |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | All env vars and defaults |
| [docs/REFINE.md](docs/REFINE.md) | Post-train refine sequence (BFGS continue, force retrain) |
| [docs/ACTIVE_LEARNING.md](docs/ACTIVE_LEARNING.md) | MaxVol AL: already-labeled AIMD leftover frames vs unlabeled MD, pause/resume, merge + retrain |
| [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md) | Local data paths and what Git ignores |

## What is not committed

Ignored by `.gitignore` on purpose:

- VASP `OUTCAR` / charge / wavefunction files and `vac_W_*` trajectory folders  
- Generated `train.cfg`, staging pools, logs  
- Trained potentials, refine intermediates, and active-learning iteration state  
- Staged binary `.bin/mlp`  

Users rebuild datasets and potentials on their machines from local DFT data.

## Citation

When using this repository, please cite:

**Mehdi Nourazar** — WC W-Vacancy MTP Workflows  
Repository: [github.com/mehDN/wc-MTP-workflows](https://github.com/mehDN/wc-MTP-workflows)

### BibTeX

```bibtex
@software{Nourazar2026wcMTPworkflows,
  author    = {Nourazar, Mehdi},
  title     = {wc-MTP-workflows: WC W-Vacancy MTP Workflows},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/mehDN/wc-MTP-workflows},
  note      = {GitHub repository}
}
```

Use `@software{Nourazar2026wcMTPworkflows}` with BibLaTeX.

When publishing results, also cite **MLIP/MTP** (Shapeev) and your DFT settings (PBE, ENCUT 450–500 eV recommended in `datasets/sources.conf`):
 
**Alexander V. Shapeev**  
*Moment Tensor Potentials: A Class of Systematically Improvable Interatomic Potentials*  
Multiscale Modeling & Simulation **14**, 1153–1173 (2016)  
DOI: [https://doi.org/10.1137/15M1054183](https://doi.org/10.1137/15M1054183)

```bibtex
@article{Shapeev2016MTP,
  author  = {Shapeev, Alexander V.},
  title   = {Moment Tensor Potentials: A Class of Systematically Improvable Interatomic Potentials},
  journal = {Multiscale Modeling \& Simulation},
  volume  = {14},
  number  = {3},
  pages   = {1153--1173},
  year    = {2016},
  doi     = {10.1137/15M1054183},
  url     = {https://doi.org/10.1137/15M1054183}
}
```
