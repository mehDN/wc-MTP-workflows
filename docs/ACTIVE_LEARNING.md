# Active learning for the WC W-vacancy MTP

Active learning (AL) expands the training set with configurations where the current potential **extrapolates**—measured by MLIP maxvol / extrapolation **grade**. Selected frames are labeled with DFT, merged into `train.cfg`, and the MTP is retrained.

Physical goal for this project: stabilize the **planar C–C dimer reconstruction** of a W vacancy relative to the unreconstructed vacancy (~3–4 eV energy lowering once the potential is trustworthy).

---

## Prerequisites

1. Initial train set: `datasets/initial/train.cfg`  
2. Trained potential: `active_learning/WC_L${MTP_LEVEL}_trained.mtp`  
3. A **candidate pool** of structures (MLIP `.cfg` format)

If either train set or MTP is missing, run:

```bash
./run.sh
```

If force RMS is still high after train, prefer refine first:

```bash
./run.sh --only refine
# then
./run.sh --al
```

---

## Candidate pools

Candidates are **unlabeled or cheaply generated** structures (MTP MD, LAMMPS, exploratory relaxations, rattled defects, pathway samples).

### Where to put them

```text
datasets/candidates/*.cfg
```

### Resolution order (`resolve_al_candidate_cfg`)

1. Explicit path: `--candidates FILE` or `active_learning.sh` argument  
2. Environment: `AL_CANDIDATE_CFG`  
3. All non-empty `datasets/candidates/*.cfg` (merged if more than one)  
4. Fallback: full AIMD staging trajectories under `datasets/initial/staging/aimd_*.cfg`  

Internal merge files (`_merged_pool.cfg`, `_aimd_staging_pool.cfg`) are skipped as inputs.

### High force-error subset from refine

When refine stage 3 runs, it writes:

```text
active_learning/refine/high_force_error.cfg
```

These are **already DFT-labeled** train configs with the largest MTP force errors. With `AL_PREFER_HIGH_FORCE_ERROR=1` (default), `run_active_learning.sh` notes this file so you can:

- Ensure they remain in `train.cfg` (they should already be), and/or  
- Prioritize related unlabeled MD frames near the same local environments for the next select-add  

They are not unlabeled DFT work by themselves; use them as a focus map for the vacancy / high-force region.

### Generating candidates (examples)

```bash
# After MTP MD / exploratory runs, convert dumps to MLIP cfg, then:
cp my_md_frames.cfg datasets/candidates/

# Or point AL at a path:
./run.sh --al --candidates /path/to/pool.cfg
```

Helper: `scripts/select_from_md.sh` (thin helper for MD frame selection workflows).

---

## One iteration

```bash
./scripts/active_learning.sh <candidate.cfg> [iteration_label]
# or via orchestrator:
./run.sh --al
./run.sh --al --candidates datasets/candidates/md_frames.cfg
```

Grade and select-add run under **MPI** (`run_mlp`, default `MPI_NPROCS=19`).

### Steps inside `active_learning.sh`

| Step | MLIP command | Output under `active_learning/<label>/` |
|------|--------------|----------------------------------------|
| 1 | `mlp calc-grade` | `graded.cfg`, `state.als`, `calc_grade.log` |
| 2 | `mlp select-add` | `dft_queue.cfg`, `selected.cfg`, `select_add.log` |

Configs with grade above `AL_SELECT_THRESHOLD` (default **3.0**) are preferred for DFT.  
`AL_SELECTION_LIMIT` (default **50**) caps how many are queued (`0` = unlimited).

---

## Label with DFT

1. Open `active_learning/<label>/dft_queue.cfg` (or `selected.cfg`).  
2. Convert each structure to a VASP job (or your preferred DFT code).  
3. Use **the same settings** as the original training data:

   - PBE, PAW W_sv + C  
   - ENCUT 450–500 eV  
   - k-spacing ~0.025 Å⁻¹ (or consistent Gamma-only policy)  

4. Convert labeled results back to MLIP `.cfg` (`mlp convert-cfg` from OUTCAR, or equivalent).  
5. Place labeled files in:

```text
datasets/labeled/my_iter001.cfg
```

---

## Merge and retrain

### Explicit labeled file

```bash
./run.sh --labeled datasets/labeled/my_iter001.cfg
```

This merges labels into `train.cfg` (via the workflow) and retrains. With auto-resume, completed upstream steps are skipped.

### Multi-iteration driver

```bash
./scripts/run_active_learning.sh
./scripts/run_active_learning.sh datasets/candidates/pool.cfg
```

For each iteration up to `AL_MAX_ITERATIONS` (default 20):

1. Run select-add → `active_learning/iter_NNN/dft_queue.cfg`  
2. **Stop for DFT** when a non-empty queue is produced (you label offline)  
3. When new labels appear in `datasets/labeled/`, re-run merge + train + next select  

If zero selections are returned, the loop treats the pool as covered for the current thresholds.

---

## Recommended loop (manual)

```text
  MTP MD / explore on unreconstructed W-vacancy
              │
              ▼
   datasets/candidates/*.cfg
              │
              ▼
   ./run.sh --al          # or --only refine first if force RMS high
              │
              ▼
   active_learning/iter_*/dft_queue.cfg  ──►  VASP labels
              │
              ▼
   datasets/labeled/*.cfg
              │
              ▼
   ./run.sh --labeled datasets/labeled/new.cfg
              │
              ▼
   validate / refine; if dimer not stable or forces high → next AL round
              │
              ▼
   optional: MTP_LEVEL=22 if defect force RMSE > 0.1 eV/Å
```

---

## Thresholds cheat sheet

| Variable | Default | When to change |
|----------|---------|----------------|
| `AL_SELECT_THRESHOLD` | 3.0 | Lower (e.g. 2.0–2.5) to select more aggressively early; raise when pool is noisy |
| `AL_SELECTION_LIMIT` | 50 | Raise for large DFT budgets; lower for small clusters |
| `AL_MAX_ITERATIONS` | 20 | Safety cap on automated driver |
| `AL_PREFER_HIGH_FORCE_ERROR` | 1 | Surface refine high-error subset as AL focus |

```bash
AL_SELECT_THRESHOLD=2.5 AL_SELECTION_LIMIT=30 ./run.sh --al
```

---

## Outputs to inspect

| Path | Meaning |
|------|---------|
| `active_learning/WC_L20_trained.mtp` | Current potential |
| `active_learning/logs/train.log` | Last fit log |
| `active_learning/logs/calc_errors_train.log` | Training-set errors |
| `active_learning/logs/train_status.env` | Last train metrics (`FORCE_RMS`, `STEP_LIMIT`, …) |
| `active_learning/refine/high_force_error.cfg` | High force-error DFT subset (from refine) |
| `active_learning/iter_*/dft_queue.cfg` | Structures to label |
| `active_learning/iter_*/calc_grade.log` | Grade statistics |
| `datasets/initial/train.cfg` | Growing training set |
| `active_learning/workflow_state.env` | Orchestrator resume state |

---

## Tips specific to W-vacancy / C–C dimer

1. Include **unreconstructed** vacancy starts and **partially reconstructed** pathway frames in the candidate pool.  
2. Sample **close C–C** distances (category `close_cc` in `sources.conf`) so the dimer well is constrained.  
3. Keep some **bulk** and **high-T** AIMD frames so bulk elastic properties do not drift.  
4. Prefer consistent supercell sizes or explicitly mix sizes to capture finite-size vacancy–vacancy effects (this repo already tracks multiple `vac_W_*` folders).  
5. Stop adding data when grades on your MD pool fall below the select threshold and dimer energetics are stable under retrain.  
6. If BFGS keeps hitting step limits after new labels, run refine before another AL round.

---

## Troubleshooting

| Issue | Action |
|-------|--------|
| “No candidate .cfg pool found” | Add files under `datasets/candidates/` or pass `--candidates` |
| Empty `dft_queue.cfg` | Pool already covered; generate more diverse MD/pathway frames or lower `AL_SELECT_THRESHOLD` |
| Too many DFT jobs | Lower `AL_SELECTION_LIMIT` |
| Forces good on train, bad on vacancy MD | Candidate pool not covering the reconstruction path |
| Template / level mismatch | Set `MTP_LEVEL` consistently; regenerate template via `ensure_mtp_template` |
| MPI launch failures mid-AL | Check `MLP_STAGE`, `ensure_mlp`, renew Kerberos if needed |
