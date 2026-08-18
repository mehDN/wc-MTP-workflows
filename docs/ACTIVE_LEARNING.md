# Active learning for the WC W-vacancy MTP

## Introduction to active learning

In machine learning, **active learning** is a paradigm in which the model selectively queries the most informative unlabeled examples for labeling, rather than labeling data randomly or passively. The goal is to reach high accuracy with far fewer labels, especially valuable when labeling is costly.

Typical loop:
1. Train on the current labeled set.
2. Score a pool (or stream) of unlabeled candidates for informativeness.
3. Query labels for the highest-scoring candidates.
4. Add the new labels and retrain.

### Common scenarios
- **Membership query synthesis** — the model invents new instances and asks for labels.
- **Stream-based selective sampling** — data arrives sequentially; the model decides on-the-fly whether to query each point.
- **Pool-based sampling** — the model selects the best subset from a large fixed pool of unlabeled data (most common in practice).

### Common query strategies
- **Uncertainty sampling** — select points where the model is least confident (least-confident, margin, or entropy).
- **Query-by-committee (QBC)** — train an ensemble and pick points with highest disagreement.
- **Expected model change** — select points that would most alter the current model parameters.
- **Expected error / variance reduction** — choose points expected to most reduce generalization error or variance.
- **D-optimality / MaxVol-based methods** — select points that maximally increase the volume (determinant) of the information matrix / active set in feature (descriptor) space. This produces an *extrapolation grade*.
- Diversity / representativeness / hybrid strategies that combine uncertainty with coverage of the data distribution.

In the domain of machine-learning interatomic potentials (MLIPs), the MaxVol / extrapolation-grade family is especially popular for linear models such as Moment Tensor Potentials (MTPs).

---

## Method used in this repository

This workflow uses the **MLIP MaxVol / extrapolation-grade** strategy (native to the MLIP-2 package).

- A candidate pool of structures is scored with `mlp calc-grade`.
- Configurations whose **extrapolation grade** exceeds `AL_SELECT_THRESHOLD` (default **3.0**) are selected by `mlp select-add` (optionally capped by `AL_SELECTION_LIMIT`).
- Selected structures are written to `dft_queue.cfg`.
- If those configs **already have** VASP `Energy` + forces (typical leftover AIMD/OUTCAR frames), they are merged into `train.cfg` and the MTP is retrained — no new DFT.
- New VASP is requested only when a selection has no Energy + forces (MTP-MD / LAMMPS dumps).

The extrapolation grade \(\gamma\) is derived from the **D-optimality criterion**. An “active set” of the most linearly independent configurations (in the MTP descriptor basis) is maintained. For a new configuration the grade measures how much it would expand the volume of that active set:

- \(\gamma < 1\): interpolation (already well-covered),
- \(\gamma > 1\): extrapolation (outside the current span; potentially informative).

Higher grades indicate more valuable candidates for improving the potential. The approach is computationally cheap (matrix operations) and correlates well with true prediction error.

### Why MaxVol / extrapolation grade?

- **MTP is linear** in its basis functions. The geometric MaxVol criterion is therefore a natural, theoretically grounded uncertainty measure—no ensemble of models is required.
- It is the **standard, well-validated active-learning machinery** of the MLIP package (Shapeev et al.).
- DFT labels are expensive; MaxVol efficiently identifies configurations that most expand the training coverage, minimizing the number of DFT calculations needed for a trustworthy potential.
- The physical target of this repository is reliable description of the **planar C–C dimer reconstruction** of a W vacancy in \(\alpha\)-WC. Reliable extrapolation control is essential for defect and pathway configurations that lie outside the initial AIMD training distribution.

Physical goal: stabilize the planar C–C dimer reconstruction of a W vacancy relative to the unreconstructed vacancy (~3–4 eV energy lowering once the potential is trustworthy).

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

Candidates can be unlabeled (MTP MD, LAMMPS, exploratory relaxations) **or already DFT-labeled** leftover AIMD frames. The initial train set only keeps a stride-subsampled subset (`HIGH_T_STRIDE` / `SUBSAMPLE_STRIDE`). The AL fallback pool is the **full** converted AIMD trajectories; MaxVol then picks the unused frames that most expand coverage, and those labels are reused.

### Where to put them

```text
datasets/candidates/*.cfg
```

### Resolution order (`resolve_al_candidate_cfg`)

1. Explicit path: `--candidates FILE` or `active_learning.sh` argument  
2. Environment: `AL_CANDIDATE_CFG`  
3. All non-empty `datasets/candidates/*.cfg` (merged if more than one)  
4. Fallback: full AIMD staging trajectories under `datasets/initial/staging/aimd_*.cfg` (already DFT-labeled leftover frames; cached as `datasets/candidates/_aimd_staging_pool.cfg`)  

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

Configs with grade above `AL_SELECT_THRESHOLD` (default **3.0**) are preferred.  
`AL_SELECTION_LIMIT` (default **50**) caps how many are queued (`0` = unlimited).

`run_active_learning.sh` then inspects `dft_queue.cfg` with `cfg_label_status.py`. Already-labeled blocks are merged; unlabeled blocks pause for VASP.

---

## Already-labeled selections (AIMD leftover frames)

If the candidate pool is the full AIMD staging set (`datasets/initial/staging/aimd_*.cfg`), selected configs usually **already have** VASP `Energy` + forces from the original OUTCARs.

In that case `run_active_learning.sh`:

1. Detects labeled vs unlabeled blocks (`scripts/cfg_label_status.py`).
2. Merges the labeled ones into `train.cfg` and retrains — **no new VASP**.
3. Reuses an existing `iter_NNN/dft_queue.cfg` on resume (does not re-run grade/select).
4. Writes `iter_NNN/merged.ok` after a successful retrain so that iteration is skipped later.

New VASP is requested only for selections that lack Energy + forces (typical MTP-MD / LAMMPS dumps).

---

## Label with DFT

1. Open `active_learning/<label>/dft_queue.cfg` (or `unlabeled_queue.cfg` if a mixed queue was split).  
2. Convert each **unlabeled** structure to a VASP job (or your preferred DFT code).  
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

1. Run select-add → `active_learning/iter_NNN/dft_queue.cfg` (skipped if that file already exists)  
2. If the queue already has Energy + forces: merge into `train.cfg` and retrain  
3. If some/all selections are unlabeled: **pause** (exit 10, workflow status `paused`) until you put labels in `datasets/labeled/` and re-run  
4. After retrain, continue to the next iteration or stop if validation passes  

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
   active_learning/iter_*/dft_queue.cfg
              │
              ├── already has Energy + forces ──► merge + retrain
              │
              └── unlabeled ──► VASP ──► datasets/labeled/*.cfg
                                      └── ./run.sh --al
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
| `AL_PAUSE_EXIT` | 10 | Driver exit when unlabeled selections still need DFT |

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
| `active_learning/iter_*/dft_queue.cfg` | Selected configs (may already have Energy + forces) |
| `active_learning/iter_*/unlabeled_queue.cfg` | Split remainder that still needs DFT |
| `active_learning/iter_*/merged.ok` | Stamp after successful merge + retrain |
| `active_learning/iter_*/calc_grade.log` | Grade statistics |
| `datasets/initial/train.cfg` | Growing training set |
| `active_learning/workflow_state.env` | Orchestrator resume state |

---

## Constraints imposed specific to W-vacancy / C–C dimer

1. Include **unreconstructed** vacancy starts and **partially reconstructed** pathway frames in the candidate pool.  
2. Sample **close C–C** distances (category `close_cc` in `sources.conf`) so the dimer well is constrained.  
3. Keep some **bulk** and **high-T** AIMD frames so bulk elastic properties do not drift.  
4. Prefer consistent supercell sizes or explicitly mix sizes to capture finite size vacancy–vacancy effects (this repo already tracks multiple `vac_W_*` folders).  
5. Stop adding data when grades on your MD pool fall below the select threshold and dimer energetics are stable under retrain.  
6. If BFGS keeps hitting step limits after new labels, run refine before another AL round.

---

## Troubleshooting

| Issue | Action |
|-------|--------|
| “No candidate .cfg pool found” | Add files under `datasets/candidates/`, pass `--candidates`, or ensure AIMD staging `.cfg` files exist |
| Empty `dft_queue.cfg` | Pool already covered; generate more diverse MD/pathway frames or lower `AL_SELECT_THRESHOLD` |
| AL asks for VASP on AIMD leftover frames | Should not happen: check that `dft_queue.cfg` has `Energy` + `fx` columns; rerun `./run.sh --al` (reuses the queue) |
| AL status `paused` | Unlabeled selections need DFT; save `.cfg` to `datasets/labeled/` and rerun `./run.sh --al` |
| Re-grade of a huge pool on every rerun | Should not happen: existing `iter_NNN/dft_queue.cfg` is reused until `merged.ok` |
| Too many DFT jobs | Lower `AL_SELECTION_LIMIT`; leftover AIMD frames do not need new VASP |
| Forces good on train, bad on vacancy MD | Candidate pool not covering the reconstruction path |
| Template / level mismatch | Set `MTP_LEVEL` consistently; regenerate template via `ensure_mtp_template` |
| MPI launch failures mid-AL | Check `MLP_STAGE`, `ensure_mlp`, renew Kerberos if needed |

---

## References

- Shapeev, A. V. (2016). *Moment Tensor Potentials: A Class of Systematically Improvable Interatomic Potentials*. Multiscale Modeling & Simulation, 14(3), 1153–1173. https://doi.org/10.1137/15M1054183

- Podryabinkin, E. V., & Shapeev, A. V. (2017). *Active learning of linearly parametrized interatomic potentials*. Computational Materials Science, 140, 171–180. https://doi.org/10.1016/j.commatsci.2017.08.031  
  (Original MaxVol / D-optimality active-learning method for linear MLIPs)

- Novikov, I. S., Gubaev, K., Podryabinkin, E. V., & Shapeev, A. V. (2021). *The MLIP package: moment tensor potentials with MPI and active learning*. Machine Learning: Science and Technology, 2(2), 025002. https://doi.org/10.1088/2632-2153/abc9fe  
  (MLIP package implementation of MTP + active learning)

- Settles, B. (2009). *Active Learning Literature Survey*. Computer Sciences Technical Report 1648, University of Wisconsin–Madison.  
  https://digital.library.wisc.edu/1793/60660  
  (General background on active-learning strategies; technical report, no DOI)
