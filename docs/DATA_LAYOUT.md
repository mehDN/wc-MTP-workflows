# Data layout and Git policy

This repository separates **committed workflow code** from **local DFT / training products**. Large binary and trajectory data are intentionally excluded so clones stay small and reproducible from your own VASP runs.

---

## Committed (safe for GitHub)

| Path | Role |
|------|------|
| `run.sh` | Main orchestrator |
| `scripts/*` | Pipeline steps |
| `templates/WC_L20.mtp`, `templates/WC_L22.mtp` | Small untrained W–C MTP templates |
| `datasets/sources.conf` | External source list (paths you edit) |
| `docs/*.md`, `README.md` | Documentation |
| `.gitignore` | Ignore rules |
| `*/.gitkeep` | Keep empty local directories in git |

---

## Local only (ignored by `.gitignore`)

### VASP / DFT outputs

- `OUTCAR`, `vasprun.xml`, `CHGCAR`, `WAVECAR`, `CONTCAR`, charge/DOS/PROCAR, etc.  
- Entire AIMD folders matching `vac_W_*/` and `bulk_*/`  
- Any `**/OUTCAR`

### Generated training data

| Path | Role |
|------|------|
| `datasets/initial/staging/` | Per-trajectory converted `.cfg` |
| `datasets/initial/train.cfg` | Working training set |
| `datasets/initial/train_full.cfg` | Pre-subsample / full merge |
| `datasets/initial/manifest.txt` | Build inventory |
| `datasets/initial/composition.txt` | Category counts |
| `datasets/initial/logs/` | convert / merge / subsample logs |
| `datasets/candidates/*` | AL candidate pools (except `.gitkeep`) |
| `datasets/labeled/*` | DFT-labeled AL configs (except `.gitkeep`) |
| `datasets/validation/*` | Holdout sets (except `.gitkeep`) |
| `datasets/static/` | Optional static DFT cfg trees |

### Active learning products

| Path | Role |
|------|------|
| `active_learning/**` | Trained `.mtp`, `.als`, per-iter grades, logs |
| exceptions | `active_learning/.gitkeep` (and empty log dir placeholders if present) |

### Runtime

| Path | Role |
|------|------|
| `logs/` | `run_*.log`, batch job outputs |
| `*.log`, `*.out`, `nohup.out` | Misc runtime logs |

### Secrets / env

- `.env`, `*.pem`, `*.key`, `credentials*`, `.secrets/`

---

## Recommended local tree

```text
wc-MTP-workflows/                 # git clone
├── run.sh
├── scripts/
├── templates/
├── docs/
├── datasets/
│   ├── sources.conf              # edit paths for extra DFT
│   ├── candidates/               # you add *.cfg for AL
│   ├── labeled/                  # you add DFT-labeled *.cfg
│   ├── validation/               # optional holdout.cfg
│   └── initial/                  # created by build_initial_dataset.sh
│       ├── staging/
│       ├── train.cfg
│       └── logs/
├── active_learning/              # created by train / AL
│   ├── WC_L20_trained.mtp
│   ├── logs/
│   └── iter_001/
├── logs/                         # orchestrator run logs
├── vac_W_2300/OUTCAR             # your AIMD (local)
├── vac_W_2500_ML/OUTCAR
└── ...
```

Default AIMD folder names are defined in `scripts/mtp_config.sh`. Point `datasets/sources.conf` at any additional bulk, defect, surface, or NEB data.

---

## `sources.conf` entries

```text
# category:kind:path
bulk:outcar:/abs/path/to/OUTCAR
defect:cfg:datasets/static/defects.cfg
surface:dir:/abs/path/to/cfg_dir/
```

| Field | Meaning |
|-------|---------|
| `category` | Composition bucket: `bulk`, `defect`, `high_t`, `close_cc`, `surface`, `pathway` |
| `kind` | `outcar` (single VASP), `cfg` (single MLIP file), `dir` (directory of `.cfg`) |
| `path` | Absolute or relative to project root |

Legacy lines without category (`kind:path`) are still accepted by the builder.

Target mix guidance (comments in file): ~500–2000 structures total with bulk/defect/high-T balance.

---

## Regenerating ignored data after clone

```bash
# 1. Install MLIP-2, place OUTCARs (or fill sources.conf)
export MLIP_ROOT=/path/to/mlip-2

# 2. Build train.cfg
./run.sh --only dataset

# 3. Train + validate
./run.sh --skip-dataset

# 4. Optional AL
#    put pools in datasets/candidates/
./run.sh --skip-dataset --al
```

Nothing in the ignored paths is required for a clean clone—only for continuing a **local** training campaign.

---

## Size and safety checklist before push

- [ ] No `OUTCAR` or `vac_W_*` under version control (`git status` clean of those)  
- [ ] No `train.cfg` / staging / trained `.mtp` staged  
- [ ] No secrets in `sources.conf` (paths may be machine-specific; prefer relative or documented placeholders)  
- [ ] Only scripts, docs, templates, and `sources.conf` committed  

```bash
git status
git check-ignore -v vac_W_2300/OUTCAR datasets/initial/train.cfg active_learning/ 2>/dev/null || true
```
