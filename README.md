# Master's Thesis: Multi-Objective Serendipity in Music Recommendation

### Installation

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Experiments

```bash
# Experiment 1 (Unimodal validation)
python experiment1.py

# Experiment 2 (Multi-objective optimization)
python PSOMOO.py
```

## Experiments

### Experiment 1: Unimodal Hypothesis Validation

Tests whether user satisfaction follows a bell curve with unexpectedness.

```bash
python experiment1.py
```

**Outputs:** Distance-rating analysis, quadratic regression, statistical tests

### Experiment 2: Multi-Objective Optimization

Compares 5 weighting strategies (α = 0.0 to 1.0) balancing distance and relevance.

```bash
python PSOMOO.py
```

**Outputs:** Comparison plots, Pareto frontier, serendipity metrics

### Sensitivity Analysis

Tests robustness of α=0.25 across different serendipity thresholds.

**Quick version** (iterative testing):

```bash
python sensitivity_quick.py  # ~10 sec, requires recommendations.pkl from PSOMOO.py
```

**Full version** (thesis reproducibility):

```bash
python sensitivity_full.py   # ~45 min, complete pipeline from scratch
```

**Result:** α=0.25 robust across 88% of reasonable threshold definitions.

### User Variation Analysis

Analyzes heterogeneity in serendipity scores across users.

```bash
python user_variation_analysis.py  # ~30 sec, requires recommendations.pkl
```

**Key Finding:** High variation (std=0.39) reveals ~60% users achieve serendipity >0.8 while ~30% achieve <0.3, suggesting user segmentation as valuable future work. Median (0.85) more representative than mean (0.64).
