# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master's thesis: **Multi-Objective Serendipity in Music Recommendation**. The system balances two competing objectives — relevance and unexpectedness — using a weighted scalarization parameter α ∈ [0, 1], optimized with Particle Swarm Optimization (PSO) over the AMBAR music dataset.

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requirements: `numpy<2`, `pandas>=2.0.0`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `scikit-surprise`

## Running Experiments

All scripts are run directly from the repo root with the venv activated. All scripts expect the `AMBAR/` directory to be present.

```bash
# Experiment 1: Validate unimodal hypothesis (RQ1.1) — ~5 min
python experiment1.py

# Experiment 2: Multi-objective optimization across 5 α values — ~15 min
python PSOMOO.py

# Sensitivity analysis (full pipeline, no prereqs) — ~8 min
python sensitivity_full.py

# User variation analysis — ~30 sec, requires recommendations.pkl + user_E_u.pkl
python user_variation.py

# Heterogeneity analysis — ~10 min, requires pickle files from PSOMOO.py
python heterogenity_analysis.py

# PSO optimizer (fast, via interpolation) — ~2-3 min, requires recommendations_optimized_grid.pkl + user_E_u.pkl
python pso_optimizer.py

# Pareto frontier analysis
python pareto_analysis.py
```

## Architecture and Data Flow

### Dataset
`AMBAR/` contains four CSVs: `ratings_info.csv`, `tracks_info.csv`, `users_info.csv`, `artists_info.csv`. Tracks have pipe-separated `styles` and `category_styles` fields that are parsed into lists.

### Core Pipeline (PSOMOO.py is the canonical version)

1. **Data loading** — AMBAR CSVs into pandas DataFrames
2. **Train/test split** — 80/20 per user (temporal if timestamp exists, else random with seed 42)
3. **CF model** — SVD from `scikit-surprise` (50 factors, 20 epochs) trained on the train split; predicts relevance scores
4. **E_u construction** — Each user's "expected style set" built from their training-set listening history (style tags of rated tracks)
5. **Recommendation generation** — For each α value, tracks are scored as `α * relevance + (1-α) * unexpectedness`; unexpectedness = style distance from E_u
6. **Serendipity scoring** — A recommendation is serendipitous if its distance from E_u exceeds `DISTANCE_THRESHOLD` (default 0.7) AND has high relevance
7. **PSO** — Swarm of 20 particles searches α ∈ [0, 1] to maximize aggregate serendipity; uses interpolation over pre-computed grid for speed

### Key Parameters
- **α (alpha)**: 0 = pure unexpectedness, 1 = pure relevance; thesis finding is α=0.25 is optimal
- **E_u**: set of styles from a user's listening history; defines "expected" taste
- **DISTANCE_THRESHOLD**: serendipity cutoff (default 0.7); sensitivity analysis tests robustness across variations of this

### Pickle File Dependencies

Scripts cache intermediate results as `.pkl` files in the root directory:

| File | Produced by | Required by |
|------|-------------|-------------|
| `recommendations.pkl` | `PSOMOO.py` | `user_variation.py`, `pareto_analysis.py` |
| `user_E_u.pkl` | `PSOMOO.py` | `user_variation.py`, `pso_optimizer.py`, `heterogenity_analysis.py` |
| `recommendations_optimized_grid.pkl` | `generate_complete.py` | `pso_optimizer.py` |
| `pso_results.pkl` | `pso_optimizer.py` | analysis scripts |

### Results Directory
`results/` contains subdirectories per analysis type: `exp1/`, `exp2/`, `exp2_full/`, `pso/`, `sensitivity/`, `user_variation/`, `heterogenity/`, `pareto/`, `t_test/`, `a_test/`.
