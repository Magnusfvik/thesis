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
