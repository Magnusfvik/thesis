# Thesis Experiment Summary
**Multi-Objective Serendipity in Music Recommendation**

---

## Background and Dataset

The thesis uses the **AMBAR** dataset — a real-world music listening dataset containing user ratings, track metadata (including genre/style tags), and artist information.

Ratings are binarized: a rating of 3–5 is treated as "liked" (mapped to 5), and 1–2 as "not liked" (mapped to 1). This is done because AMBAR ratings are heavily skewed toward high values, making binary treatment more meaningful.

For all experiments, data is split **80/20 per user** into train and test sets. A random sample of **100 users** is drawn for computational feasibility (users must have at least 10–30 training ratings).

---

## Core Concepts Used Across All Experiments

### Expected Set — E_u

For each user *u*, an **Expected Set** E_u represents the music they already "know" — their taste profile. It is built from the genres/styles of all tracks the user rated in training:

> E_u = { style categories of all tracks rated by user u in the training set }

E_u is a set of genre tags (e.g., "Rock", "Electronic", "Jazz") extracted from the user's listening history.

### Unexpectedness — Distance from E_u

A track is considered *unexpected* if its style is far from what the user normally listens to. Distance is measured using **Jaccard distance** between the track's style tags and the user's E_u:

```
Jaccard distance(A, B) = 1 − |A ∩ B| / |A ∪ B|
```

Distance ∈ [0, 1], where 0 = identical styles, 1 = completely different.

### Relevance — Collaborative Filtering (SVD)

A **Singular Value Decomposition (SVD)** model (from the scikit-surprise library) is trained on the user–item rating matrix to predict how much a user would like any given track. This provides a *predicted rating* per (user, track) pair, representing **relevance**.

Parameters: 50 latent factors, 20 training epochs.

---

## Experiment 1 — Does Serendipity Follow a Bell Curve? (RQ1.1)

### Goal

Validate the hypothesis that user satisfaction follows a **unimodal (inverted-U) relationship** with unexpectedness: a little novelty is good, but too much or too little reduces satisfaction.

### Method

For each user in the test set, actual rated tracks are taken and their Jaccard distance from E_u is computed (averaged over a sample of 50 E_u tracks for efficiency). Results are grouped into 10 distance bins ([0.0–0.1), [0.1–0.2), …, [0.9–1.0]) and mean user rating is computed per bin.

A **quadratic regression** is then fit:

```
Rating = β₀ + β₁ × distance + β₂ × distance²
```

If β₂ < 0, the parabola opens downward — confirming a peak at:

```
δ* = −β₁ / (2β₂)
```

Statistical significance of the relationship across bins is tested with a **one-way ANOVA** (null hypothesis: mean ratings are equal across all distance bins).

### What It Tells Us

If the unimodal hypothesis holds: β₂ < 0 (inverted U-shape), p < 0.05 in ANOVA, and δ* falls within observed range. This would justify the thesis premise that there exists an *optimal unexpectedness* rather than simply "more novel = better."

---

## Experiment 2 — Multi-Objective Recommendation with Weighted Scalarization (RQ2)

### Goal

Generate recommendations that jointly optimize **relevance** and **unexpectedness**, and compare five different trade-off strategies.

### Scoring Formula

Each track is scored as a **weighted combination** of the two objectives:

```
Score(u, track) = α × distance(track, E_u) + (1 − α) × cf_score_normalized
```

Where:
- `α ∈ [0, 1]` controls the trade-off
- `distance` is the Jaccard distance from E_u (unexpectedness)
- `cf_score_normalized` is the SVD predicted rating normalized to [0, 1]

The top-10 tracks by combined score are recommended.

**Five strategies tested:**

| α   | Strategy         |
|-----|-----------------|
| 0.0 | Pure CF (relevance only) |
| 0.25 | CF-biased (mostly relevance) |
| 0.5 | Balanced |
| 0.75 | Distance-biased (mostly novel) |
| 1.0 | Pure distance (novelty only) |

### Evaluation Metrics

Each strategy is evaluated on:

- **Serendipity** (Ge et al., 2010): fraction of recommended tracks that are *both* unexpected (distance > 0.7) and relevant (CF score > 1.8)
- **Diversity (ILD)**: average pairwise Jaccard distance *within* each user's top-10 list (intra-list diversity)
- **Accuracy (RMSE/MAE)**: how well the CF component predicts actual held-out ratings
- **Coverage**: number of unique tracks recommended across all users

### Key Finding

α = 0.25 (CF-biased) achieves the best serendipity score. Pure relevance (α = 0.0) produces safe but unsurprising recommendations; pure novelty (α = 1.0) produces surprising but irrelevant recommendations. The sweet spot is a light lean toward novelty on top of a relevance-first base.

---

## Experiment 3 — PSO to Find Optimal α (RQ2.1)

### Goal

Rather than manually testing 5 fixed α values, use **Particle Swarm Optimization (PSO)** to search the continuous space α ∈ [0, 1] and find the α that maximizes serendipity.

### Method

A swarm of **20 particles** each represent a candidate α value. Each particle has a position (current α) and a velocity (direction of search). At each iteration, particles move according to:

```
v(t+1) = w × v(t) + c₁ × r₁ × (personal_best − position) + c₂ × r₂ × (global_best − position)

position(t+1) = position(t) + v(t+1)
```

Where:
- `w = 0.729` — inertia weight (keeps particles moving in current direction)
- `c₁ = c₂ = 1.49445` — cognitive and social acceleration coefficients (standard constriction settings)
- `r₁, r₂` — random values ∈ [0, 1], sampled each iteration

**Fitness function** = serendipity score at that α, evaluated via cubic interpolation over a pre-computed grid of serendipity values (avoids re-running the full recommendation pipeline each iteration).

PSO is run **20 independent times** (different random seeds) for robustness. Position is clipped to [0, 1] at each step.

### Key Finding

PSO consistently converges to α ≈ 0.25, matching and validating the result from Experiment 2. Convergence typically occurs within 15–20 iterations.

---

## Experiment 4 — Sensitivity Analysis (RQ2.2)

### Goal

Test whether the α = 0.25 finding is robust to the choice of serendipity threshold — i.e., does the result change if the definition of "serendipitous" is varied?

### Method

The full pipeline (data loading → SVD training → E_u construction → recommendation generation) is re-run. Serendipity is then evaluated under **11 different threshold combinations**, varying both:

- **Distance threshold** (0.5 to 0.9): how far from E_u a track must be to count as unexpected
- **CF threshold** (1.2 to 2.5): how high the predicted rating must be to count as relevant

For each threshold combination, the α value with highest serendipity is identified. If α = 0.25 wins across the majority of threshold settings, the finding is considered robust.

### Key Finding

α = 0.25 is optimal under ~88% of reasonable threshold definitions, confirming the result is not an artifact of the specific threshold chosen.

---

## Experiment 5 — User Heterogeneity Analysis

### Goal

Determine whether a single optimal α applies to all users, or whether different types of users benefit from different trade-off settings.

### Method

Users are segmented by **listening diversity**: how musically varied their existing listening history is. Diversity is measured as the Intra-List Distance (ILD) of the tracks in their E_u. Users are split into three segments (low / medium / high diversity).

For each segment, the serendipity curve across α values is plotted and the per-segment optimal α is identified. Statistical significance of differences between segments is tested with **ANOVA** and paired **t-tests** with effect sizes (Cohen's d).

### Key Finding

There is significant heterogeneity across users (std ≈ 0.39 in serendipity scores). Roughly 60% of users achieve serendipity > 0.8 while ~30% achieve < 0.3. The median (0.85) is more representative than the mean (0.64), suggesting user segmentation or personalized α values may be valuable future work.

---

## Summary of Methods Used

| Method | Purpose |
|--------|---------|
| SVD (matrix factorization) | Relevance prediction via collaborative filtering |
| Jaccard distance | Unexpectedness: style-based distance from user's taste profile |
| Weighted scalarization | Multi-objective combination of relevance and unexpectedness |
| Quadratic regression | Fitting inverted-U curve to distance–rating relationship |
| One-way ANOVA | Significance test across distance bins |
| Particle Swarm Optimization | Continuous search for optimal α |
| Cubic interpolation | Fast PSO fitness evaluation without rerunning full pipeline |
| ILD (Intra-List Diversity) | Measuring diversity within recommendation lists |
| Sensitivity analysis | Robustness of α = 0.25 across threshold definitions |
| User segmentation + t-tests | Heterogeneity analysis across user types |
