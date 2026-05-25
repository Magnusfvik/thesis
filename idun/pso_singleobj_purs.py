#!/usr/bin/env python3
"""
pso_singleobj_purs.py
=====================
Single-objective PSO on PURS data — validates the grid search optimum.

Identical PSO mechanics to pso_optimizer2.py but with:
  - Input : idun_results_purs/recommendations_fair_complete.pkl
  - Metric: harmonic mean serendipity  (U=distance, R=(cf_score-1)/4)
            same metric used in experiment2_idun.py / pso_moo_idun.py

Purpose:
  Confirm that PSO independently converges to the same α region found
  by the PURS grid search (exp2 best α ≈ 0.35).  If they agree, we have
  two independent methods (deterministic grid + stochastic swarm) pointing
  at the same optimum — a strong validation for the thesis.

Runtime: ~2 minutes locally

Output (in idun_results_purs/):
  pso_singleobj_summary.txt
  pso_singleobj_results.csv
  pso_singleobj_curve.png
"""

import os
import pickle
import warnings
from datetime import datetime
from scipy.interpolate import PchipInterpolator
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUT_DIR  = os.path.join(os.path.dirname(__file__), "idun_results_purs")
RECS_PKL = os.path.join(OUT_DIR, "recommendations_fair_complete.pkl")

print("=" * 70)
print("SINGLE-OBJECTIVE PSO — PURS DATA")
print("  Metric: harmonic mean of unexpectedness and normalised relevance")
print("=" * 70)
print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ---------------------------------------------------------------------------
# 1. Load PURS recommendations
# ---------------------------------------------------------------------------
print("[1/5] Loading PURS recommendations...")

if not os.path.exists(RECS_PKL):
    print(f"  ERROR: {RECS_PKL} not found.")
    print("  Run idun/generate_purs_inspired.py first (or pull from IDUN).")
    raise SystemExit(1)

recs       = pickle.load(open(RECS_PKL, "rb"))
alpha_grid = sorted(recs.keys())
all_users  = list(recs[alpha_grid[0]].keys())

print(f"  α values : {alpha_grid}")
print(f"  Users    : {len(all_users):,}")

# ---------------------------------------------------------------------------
# 2. Compute serendipity for each α using harmonic mean
# ---------------------------------------------------------------------------
print("\n[2/5] Computing harmonic-mean serendipity on grid...")


def harmonic_serendipity(rec_list):
    """
    Per-item serendipity = harmonic mean of:
        U = distance (already in [0,1])
        R = (cf_score - 1) / 4  (normalises [1,5] → [0,1])
    User serendipity = mean over top-10 items.
    Matches the metric in experiment2_idun.py.
    """
    if not rec_list:
        return 0.0
    scores = []
    for r in rec_list:
        U = float(r["distance"])
        R = (float(r["cf_score"]) - 1.0) / 4.0
        denom = U + R
        hm = (2 * U * R / denom) if denom > 0 else 0.0
        scores.append(hm)
    return float(np.mean(scores))


grid_serendipity = {}
for alpha in alpha_grid:
    user_scores = [harmonic_serendipity(recs[alpha][u]) for u in all_users]
    grid_serendipity[alpha] = float(np.mean(user_scores))
    print(f"  α={alpha:.2f} : serendipity={grid_serendipity[alpha]:.4f}")

best_grid_alpha = max(grid_serendipity, key=grid_serendipity.get)
print(f"\n  Grid search best : α={best_grid_alpha:.2f}  "
      f"(serendipity={grid_serendipity[best_grid_alpha]:.4f})")

# PCHIP interpolator (shape-preserving — no spurious oscillations)
alpha_arr  = np.array(alpha_grid)
seren_arr  = np.array([grid_serendipity[a] for a in alpha_grid])
interpolator = PchipInterpolator(alpha_arr, seren_arr)

# ---------------------------------------------------------------------------
# 3. PSO implementation (same hyper-parameters as pso_optimizer2.py)
# ---------------------------------------------------------------------------
print("\n[3/5] Running PSO (20 particles × 30 iterations × 10 runs)...")

W  = 0.729
C1 = 1.49445
C2 = 1.49445
N_PARTICLES  = 20
MAX_ITER     = 30
N_RUNS       = 10


def evaluate(alpha):
    return float(np.clip(interpolator(np.clip(alpha, 0.0, 1.0)), 0.0, 1.0))


def run_pso(seed):
    np.random.seed(seed)
    positions  = np.random.uniform(0.0, 1.0, N_PARTICLES)
    velocities = np.random.uniform(-0.1, 0.1, N_PARTICLES)
    pbest_pos  = positions.copy()
    pbest_fit  = np.array([evaluate(p) for p in positions])
    gbest_pos  = pbest_pos[np.argmax(pbest_fit)]
    gbest_fit  = pbest_fit.max()
    history    = []

    for _ in range(MAX_ITER):
        r1 = np.random.random(N_PARTICLES)
        r2 = np.random.random(N_PARTICLES)
        velocities = (W * velocities
                      + C1 * r1 * (pbest_pos - positions)
                      + C2 * r2 * (gbest_pos - positions))
        positions  = np.clip(positions + velocities, 0.0, 1.0)
        fitness    = np.array([evaluate(p) for p in positions])

        improved = fitness > pbest_fit
        pbest_pos[improved] = positions[improved]
        pbest_fit[improved] = fitness[improved]

        if fitness.max() > gbest_fit:
            gbest_fit = fitness.max()
            gbest_pos = positions[np.argmax(fitness)]

        history.append({"gbest_pos": gbest_pos, "gbest_fit": gbest_fit,
                        "diversity": float(np.std(positions))})

    return gbest_pos, gbest_fit, history


run_results = []
for i in range(N_RUNS):
    pos, fit, hist = run_pso(seed=i)
    run_results.append({"run": i + 1, "alpha": pos, "serendipity": fit, "history": hist})
    print(f"  Run {i+1:2d}: α={pos:.4f}  serendipity={fit:.4f}")

optimal_alphas = np.array([r["alpha"]      for r in run_results])
serendipities  = np.array([r["serendipity"] for r in run_results])

mean_alpha = optimal_alphas.mean()
std_alpha  = optimal_alphas.std()
mean_seren = serendipities.mean()
std_seren  = serendipities.std()

# ---------------------------------------------------------------------------
# 4. Statistical comparison: PSO optimum vs pure CF (α=0.0)
# ---------------------------------------------------------------------------
print("\n[4/5] Statistical comparison: PSO optimum vs pure CF...")

# Per-user serendipity at PSO mean α (use nearest grid point)
nearest_alpha = min(alpha_grid, key=lambda a: abs(a - mean_alpha))
pso_user_scores = [harmonic_serendipity(recs[nearest_alpha][u]) for u in all_users]
cf_user_scores  = [harmonic_serendipity(recs[0.0][u])           for u in all_users]

t_stat, p_val = stats.ttest_rel(pso_user_scores, cf_user_scores)
diff          = np.array(pso_user_scores) - np.array(cf_user_scores)
cohens_d      = diff.mean() / diff.std()
ci_low, ci_high = stats.t.interval(
    0.95, len(diff) - 1, loc=diff.mean(), scale=stats.sem(diff)
)

print(f"  PSO optimum (α≈{mean_alpha:.2f}) vs pure CF (α=0.00)")
print(f"  Mean serendipity — PSO: {np.mean(pso_user_scores):.4f}  |  "
      f"CF: {np.mean(cf_user_scores):.4f}")
print(f"  Mean difference : {diff.mean():+.4f}")
print(f"  t({len(diff)-1}) = {t_stat:.3f},  p = {p_val:.6f}")
print(f"  Cohen's d       : {cohens_d:.3f}")
print(f"  95% CI          : [{ci_low:.4f}, {ci_high:.4f}]")

# ---------------------------------------------------------------------------
# 5. Save results
# ---------------------------------------------------------------------------
print("\n[5/5] Saving results...")

os.makedirs(OUT_DIR, exist_ok=True)

# CSV
df = pd.DataFrame({
    "run":        [r["run"]        for r in run_results],
    "alpha":      [r["alpha"]      for r in run_results],
    "serendipity":[r["serendipity"] for r in run_results],
})
csv_path = os.path.join(OUT_DIR, "pso_singleobj_results.csv")
df.to_csv(csv_path, index=False, float_format="%.4f")
print(f"  Saved: {csv_path}")

# Summary text
summary_path = os.path.join(OUT_DIR, "pso_singleobj_summary.txt")
with open(summary_path, "w") as fh:
    fh.write("=" * 70 + "\n")
    fh.write("SINGLE-OBJECTIVE PSO — PURS DATA\n")
    fh.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    fh.write("=" * 70 + "\n\n")
    fh.write("GRID SEARCH (baseline)\n")
    fh.write(f"  Best alpha     : {best_grid_alpha:.4f}\n")
    fh.write(f"  Best serendipity: {grid_serendipity[best_grid_alpha]:.4f}\n\n")
    fh.write(f"PSO RESULTS ({N_RUNS} independent runs)\n")
    fh.write(f"  Mean alpha     : {mean_alpha:.4f} ± {std_alpha:.4f}\n")
    fh.write(f"  Range          : [{optimal_alphas.min():.4f}, "
             f"{optimal_alphas.max():.4f}]\n")
    fh.write(f"  Mean serendipity: {mean_seren:.4f} ± {std_seren:.4f}\n\n")
    fh.write("INDIVIDUAL RUNS\n")
    for r in run_results:
        fh.write(f"  Run {r['run']:2d}: alpha={r['alpha']:.4f}  "
                 f"serendipity={r['serendipity']:.4f}\n")
    fh.write("\nSTATISTICAL COMPARISON (PSO optimum vs pure CF)\n")
    fh.write(f"  PSO α≈{mean_alpha:.2f}  serendipity={np.mean(pso_user_scores):.4f}\n")
    fh.write(f"  Pure CF α=0.00  serendipity={np.mean(cf_user_scores):.4f}\n")
    fh.write(f"  Mean difference : {diff.mean():+.4f}\n")
    fh.write(f"  t-statistic     : {t_stat:.3f}\n")
    fh.write(f"  p-value         : {p_val:.6f}\n")
    fh.write(f"  Cohen's d       : {cohens_d:.3f}\n")
    fh.write(f"  95% CI          : [{ci_low:.4f}, {ci_high:.4f}]\n\n")
    agrees = abs(mean_alpha - best_grid_alpha) < 0.10
    fh.write(f"PSO vs grid agreement (|Δα| < 0.10): "
             f"{'YES' if agrees else 'NO'}\n")
    fh.write(f"Alpha difference: {abs(mean_alpha - best_grid_alpha):.4f}\n")

print(f"  Saved: {summary_path}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Single-Objective PSO — PURS Data (Harmonic Mean Serendipity)",
             fontsize=13, fontweight="bold")

# Panel 1: serendipity curve + PSO convergence points
ax = axes[0]
x_smooth = np.linspace(0, 1, 300)
ax.plot(x_smooth, interpolator(x_smooth), "-", color="steelblue",
        linewidth=2, label="PCHIP interpolation")
ax.plot(alpha_arr, seren_arr, "o", color="steelblue", markersize=8,
        label="Grid points")
ax.scatter(optimal_alphas, serendipities, c="crimson", s=80, zorder=5,
           alpha=0.7, label=f"PSO runs (n={N_RUNS})")
ax.axvline(mean_alpha, color="crimson", linestyle="--", linewidth=2,
           label=f"PSO mean α={mean_alpha:.3f}")
ax.axvline(best_grid_alpha, color="navy", linestyle=":", linewidth=2,
           label=f"Grid best α={best_grid_alpha:.2f}")
ax.set_xlabel("α", fontweight="bold")
ax.set_ylabel("Serendipity (harmonic mean)", fontweight="bold")
ax.set_title("Serendipity Curve + PSO Convergence")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 2: convergence of one example run
ax = axes[1]
hist = run_results[0]["history"]
ax.plot([h["gbest_fit"] for h in hist], color="green", linewidth=2)
ax.set_xlabel("Iteration", fontweight="bold")
ax.set_ylabel("Best serendipity found", fontweight="bold")
ax.set_title("PSO Convergence (Run 1)")
ax.grid(alpha=0.3)

# Panel 3: distribution of optimal α across runs
ax = axes[2]
ax.hist(optimal_alphas, bins=10, color="orange", edgecolor="black", alpha=0.8)
ax.axvline(mean_alpha, color="red", linestyle="--", linewidth=2,
           label=f"Mean={mean_alpha:.3f} ± {std_alpha:.3f}")
ax.set_xlabel("Optimal α found by PSO", fontweight="bold")
ax.set_ylabel("Frequency", fontweight="bold")
ax.set_title(f"PSO Robustness ({N_RUNS} runs)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "pso_singleobj_curve.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {fig_path}")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Grid search best : α={best_grid_alpha:.2f}  "
      f"serendipity={grid_serendipity[best_grid_alpha]:.4f}")
print(f"  PSO mean         : α={mean_alpha:.4f} ± {std_alpha:.4f}  "
      f"serendipity={mean_seren:.4f}")
print(f"  |Δα|             : {abs(mean_alpha - best_grid_alpha):.4f}")
print(f"  Methods agree    : {'YES ✓' if abs(mean_alpha-best_grid_alpha)<0.10 else 'NO ✗'}")
print(f"  Statistical test : t={t_stat:.3f}, p={p_val:.6f}, d={cohens_d:.3f}")
print("=" * 70)
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
