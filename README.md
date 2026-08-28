# Development of Plasma Reactor Models Combining First-Principles and Machine Learning Techniques

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-Multi--objective-9cf)
![MATLAB](https://img.shields.io/badge/MATLAB-DoE%20%26%20Sequencing-orange)
![COMSOL](https://img.shields.io/badge/COMSOL-Multiphysics-blue)

Code for the Diploma thesis, Development of Plasma Reactor Models Combining First-Principles and Machine Learning Techniques — National Technical University of Athens (NTUA), Stamatios Tountas.

---

## Overview

The first-principles model is a COMSOL Multiphysics model of an inductively coupled plasma (ICP) argon
etching reactor. One converged operating point costs roughly an hour of wall time on a Threadripper 
2990WX32-core workstation, which makes the model impractical for optimisation, sensitivity studies or control.

This repository is the  pipeline that compresses it into a small, shape-constrained neural
network surrogate: design of experiments, warm-start-friendly run sequencing, budget ablation,
physics-guided training, hyperparameter search and uncertainty quantification.

|  |  |
| --- | --- |
| **Inputs (4)** | Coil power [W], chamber pressure [Torr], feed flow [sccm], bias voltage [V] |
| **Outputs (10)** | Etch rate at 10 radial positions on the wafer, centre → edge |
| **Surrogate** | MLP 4 → 20 → 20 → 20 → 10, SiLU activations, trained with L-BFGS (Adam as baseline) |
| **Dataset** | 1512 converged COMSOL runs over the 4-D operating window |
| **Budgets studied** | 100, 75, 50, 25, 12.5, 10, 7.5, 5 % of the dataset × 10 data seeds |

---

## Key ideas

**1 · Sequenced execution for warm starts.**
Consecutive COMSOL runs are ordered so that jumps in the E/N proxy, feed flow and pressure stay under
empirical thresholds, letting each solve start from the previous solution instead of from scratch. The
order comes from a nearest-neighbour tour under a Monte-Carlo search over feature weights, scored by a
normalised bottleneck penalty — a heuristic, not an exact bottleneck-TSP solve.

**2 · Greedy K-Centers core-set sampling.**
Farthest-point selection keeps the corners and the sparse regions of the operating window covered when
only 5–12.5 % of the points are used for training. Benchmarked head-to-head against uniform random
selection at every budget and seed.

**3 · Spatially-aware physics loss.**
MSE plus three penalties: positive radial gradients (etch rate must not increase outward), weighted per
interval by the inverse spread of the training slopes; curvature exceeding a per-hinge tolerance `tau_i`
taken from the 99th percentile of the training second differences; and negative etch rates, penalised in
physical units after unscaling.

**4 · Deep ensembles.**
10 → 200 independently initialised members per budget, giving epistemic dispersion over regions of the
operating window the surrogate never saw, with bootstrap resampling of members for error bars on the
scaling curves.

---

## Repository layout

```text
Thesis_Git/
├── 01_generate_random_design.m        # Uniform random DoE over the 4-D window
├── 02_kcenters_point_filler.m         # Greedy farthest-point fill of low-density regions
├── 03_minimax_path_sequencer.m        # Warm-start execution order (weight search + NN tour)
│
├── src/                               # Importable modules
│   ├── models.py                      # SurrogateMLP (20-20-20, SiLU)
│   ├── physics_loss.py                # PhysicsGuidedLoss, SpatiallyAwarePhysicsLoss
│   ├── training.py                    # train_lbfgs, train_adam (early stopping + history)
│   ├── evaluation.py                  # Per-recipe metrics, violation counting, CSV export
│   ├── data_loader.py                 # Split generation + load_precomputed_splits (used by experiments/)
│   ├── load_precomputed_splits.py     # Stand-alone alternative splitter (not imported by experiments/)
│   └── sampling.py                    # Stand-alone alternative splitter with LHS / random / k-centers
│
├── experiments/                       # Executable pipeline, run in numeric order
│   ├── 00_export_data_splits.py
│   ├── 00b_extract_spatial_weights.py
│   ├── 02_train_A.py
│   ├── 02_train_B.py
│   ├── 02_train_C.py
│   ├── 03_optuna_spatial_search.py
│   ├── 04_train_spatially_aware_models.py
│   └── 05_train_deep_ensembles.py
│
├── data/                              # git-ignored
│   ├── v10_Ar_eth_dataset_full.txt    # Raw COMSOL export (tab-separated, with header)
│   └── processed/                     # Generated splits
│       └── spatial_weights/           # Generated per-split tau_i, W_i, W_mono_i
├── models/                            # git-ignored — trained weights + per-epoch histories
└── results/                           # git-ignored — metrics, Optuna DBs, figures
```

`data/`, `models/` and `results/` are excluded from version control; everything in them is reproducible
from the raw dataset by re-running the pipeline.

---

## Data format

`data/v10_Ar_eth_dataset_full.txt` — tab-separated, one row per COMSOL run, with a header line:

| Column | Meaning | Design-space bounds used by the DoE scripts |
| --- | --- | --- |
| `Power` | Coil power [W] | 100 – 1500 |
| `Pressure` | Chamber pressure [Torr] | 0.010 – 0.060 |
| `Feed` | Feed gas flow [sccm] | 20 – 120 |
| `Vbias` | Bias voltage [V] | 50 – 550 |
| `Point1_EtchRate` … `Point10_EtchRate` | Etch rate at 10 radial positions, centre → edge | — |

Input features are standardised with a `StandardScaler` and the 10 outputs with a per-point mean/std,
both fitted **only on the training split** of each budget/seed combination.

---

## Setup

```powershell
conda create -n thesis python=3.11
conda activate thesis
pip install -r requirements.txt
```

For a CUDA build of PyTorch, install it from <https://pytorch.org> instead of relying on
`requirements.txt`. The pipeline is CPU-only by design — the models are tiny and the parallelism is
across runs, not inside them.

---

## Running the pipeline

Run every Python script **from the repository root**, so that `src` is importable and the relative
`data/` and `results/` paths resolve:

```powershell
python experiments\00_export_data_splits.py
```

| # | Script | What it does |
| --- | --- | --- |
| 1 | `01_generate_random_design.m` | Uniform random points over the 4-D window, split into cluster files for parallel COMSOL execution |
| 2 | `02_kcenters_point_filler.m` | Adds points in the emptiest regions of an existing design (greedy farthest-point) and plots the coverage |
| 3 | `03_minimax_path_sequencer.m` | Aggregates the cluster files and searches feature weights for the execution order with the smallest worst-case jump in E/N, flow and pressure |
| 4 | `00_export_data_splits.py` | Writes train / val / test / unseen CSVs for 8 budgets × 2 sampling methods × 10 seeds |
| 5 | `00b_extract_spatial_weights.py` | Derives `tau_i`, `W_i`, `W_mono_i` per split into `data/processed/spatial_weights/` |
| 6 | `02_train_A.py` | Baseline: plain MSE, L-BFGS vs Adam, all budgets/methods/seeds |
| 7 | `02_train_B.py` | Uniform monotonicity penalty, L-BFGS, K-Centers splits |
| 8 | `02_train_C.py` | `lambda_mono` sensitivity sweep (0 → 20) with aggregated accuracy/violation trade-off tables |
| 9 | `03_optuna_spatial_search.py` | 3-objective TPE search (RMSE, MAE, violation rate) over the three lambdas, 50 trials per budget, stored in SQLite |
| 10 | `04_train_spatially_aware_models.py` | Retrains the top Pareto trials — zero-violation first, then lowest RMSE — across all 10 seeds |
| 11 | `05_train_deep_ensembles.py` | Trains up to 200 members per budget and analyses ensemble size vs accuracy, calibration and cost |

Every training script skips work whose output files already exist, so an interrupted run can simply be
restarted. Parallelism is a `ProcessPoolExecutor` with 6 workers and `torch.set_num_threads(1)` inside
each worker; adjust `max_workers` / `NUM_CORES` to your machine.

---

## Naming conventions

Splits, weights and models are keyed by a single prefix:

```text
{method}_{budget}pct_seed{seed}
```

* `method` — `kcenters` or `random`
* `budget` — `100`, `75.0`, `50.0`, `25.0`, `12.5`, `10.0`, `7.5`, `5.0`
* `seed` — `42, 123, 456, 789, 1024, 2024, 3049, 4096, 5012, 6120`

Example: `kcenters_12.5pct_seed42_train.csv`, `weights_kcenters_12.5pct_seed42.csv`. Note that the full
budget is `100`, not `100.0` — the two are different strings and different filenames.

---

## Evaluation

Two held-out tiers are reported for every model:

* **Local** — the test split drawn from inside the active data budget.
* **Global unseen** — every point of the dataset that the budget never touched, i.e. performance in
  regions of the operating window the model was never shown.

Reported metrics are R², RMSE, MAE, MAPE, the **monotonicity violation rate** (share of the 9 radial
intervals with a positive gradient) and a jaggedness index (mean squared second difference). Model
selection follows the violation rate first and error metrics as the tie-break.

---

## Citation

```bibtex
@mastersthesis{tountas2026plasma,
  author = {Tountas, Stamatios},
  title  = {Development of Plasma Reactor Models Combining First-Principles and Machine Learning Techniques},
  school = {National Technical University of Athens},
  year   = {2026},
  type   = {Diploma Thesis} 
}
```
