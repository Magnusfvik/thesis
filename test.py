"""
PSO Optimizer for Fair Generation Data
=======================================

Uses the clean, fair-generated recommendations to validate
grid search findings via bio-inspired optimization.

Reads: recommendations_fair_complete.pkl
Outputs: pso_fair_results.pkl, pso_fair_analysis.png

Runtime: ~30 minutes (20 runs)
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.stats import ttest_rel
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PSO VALIDATION - FAIR GENERATION DATA")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

PSO_CONFIG = {
    'n_particles': 20,
    'n_iterations': 30,
    'n_runs': 20,        # EC standard
    'w': 0.729,
    'c1': 1.49445,
    'c2': 1.49445,
    'alpha_min': 0.0,
    'alpha_max': 1.0,
}

THRESHOLDS = {
    'distance': 0.7,
    'cf_score': 1.8,
}

# ============================================================================
# LOAD FAIR GENERATION DATA
# ============================================================================

print("[1/5] Loading fair generation data...")

try:
    recs = pickle.load(open('recommendations_fair_complete.pkl', 'rb'))
    alpha_grid = sorted(recs.keys())
    print(f"  ✓ Loaded {len(alpha_grid)} α values: {alpha_grid}")
except FileNotFoundError:
    print("  ❌ ERROR: recommendations_fair_complete.pkl not found!")
    print("     Run generate_fair_complete.py first.")
    exit(1)

# ============================================================================
# COMPUTE SERENDIPITY AT GRID POINTS
# ============================================================================

print("\n[2/5] Computing serendipity at grid points...")

def compute_serendipity(recs_dict):
    """Calculate mean serendipity across users"""
    user_scores = []
    for user_id, rec_list in recs_dict.items():
        n_ser = sum(1 for r in rec_list 
                   if r['distance'] > THRESHOLDS['distance'] 
                   and r['cf_score'] > THRESHOLDS['cf_score'])
        user_scores.append(n_ser / len(rec_list) if rec_list else 0)
    return np.array(user_scores)

grid_serendipity_mean = []
grid_serendipity_per_user = {}

print()
for alpha in alpha_grid:
    per_user = compute_serendipity(recs[alpha])
    grid_serendipity_per_user[alpha] = per_user
    grid_serendipity_mean.append(per_user.mean())
    print(f"  α={alpha:.2f}: {per_user.mean():.4f} (±{per_user.std():.4f})")

grid_alphas = np.array(alpha_grid)
grid_scores = np.array(grid_serendipity_mean)

best_grid_idx = np.argmax(grid_scores)
best_grid_alpha = grid_alphas[best_grid_idx]
best_grid_score = grid_scores[best_grid_idx]

print(f"\n  Grid search best: α={best_grid_alpha:.2f}, serendipity={best_grid_score:.4f}")

# ============================================================================
# BUILD INTERPOLATION
# ============================================================================

print("\n[3/5] Building PCHIP interpolation...")

cs = PchipInterpolator(grid_alphas, grid_scores)

alpha_dense = np.linspace(0.0, 1.0, 1000)
scores_dense = np.clip(cs(alpha_dense), 0.0, 1.0)

interp_peak_idx = np.argmax(scores_dense)
interp_peak_alpha = alpha_dense[interp_peak_idx]
interp_peak_score = scores_dense[interp_peak_idx]

print(f"  Interpolation peak: α={interp_peak_alpha:.4f}, serendipity={interp_peak_score:.4f}")

def fitness(alpha):
    """PSO fitness function"""
    return float(np.clip(cs(np.clip(alpha, 0.0, 1.0)), 0.0, 1.0))

# ============================================================================
# RUN PSO
# ============================================================================

print(f"\n[4/5] Running PSO ({PSO_CONFIG['n_runs']} runs)...")
print(f"  Particles: {PSO_CONFIG['n_particles']}, Iterations: {PSO_CONFIG['n_iterations']}\n")

def run_pso(seed, config):
    """Single PSO run"""
    rng = np.random.RandomState(seed)
    
    n_p = config['n_particles']
    n_i = config['n_iterations']
    w, c1, c2 = config['w'], config['c1'], config['c2']
    lo, hi = config['alpha_min'], config['alpha_max']
    
    positions = rng.uniform(lo, hi, n_p)
    velocities = rng.uniform(-(hi-lo)*0.1, (hi-lo)*0.1, n_p)
    
    pbest_pos = positions.copy()
    pbest_score = np.array([fitness(p) for p in positions])
    
    gbest_idx = np.argmax(pbest_score)
    gbest_pos = pbest_pos[gbest_idx]
    gbest_score = pbest_score[gbest_idx]
    
    history = np.zeros(n_i)
    
    for iteration in range(n_i):
        r1, r2 = rng.uniform(0, 1, n_p), rng.uniform(0, 1, n_p)
        
        velocities = (w * velocities + 
                     c1 * r1 * (pbest_pos - positions) +
                     c2 * r2 * (gbest_pos - positions))
        
        positions = np.clip(positions + velocities, lo, hi)
        scores = np.array([fitness(p) for p in positions])
        
        improved = scores > pbest_score
        pbest_pos[improved] = positions[improved]
        pbest_score[improved] = scores[improved]
        
        run_best_idx = np.argmax(pbest_score)
        if pbest_score[run_best_idx] > gbest_score:
            gbest_pos = pbest_pos[run_best_idx]
            gbest_score = pbest_score[run_best_idx]
        
        history[iteration] = gbest_score
    
    return gbest_pos, gbest_score, history

# Execute runs
run_results = []
all_histories = []

for run in range(PSO_CONFIG['n_runs']):
    best_alpha, best_score, history = run_pso(100 + run, PSO_CONFIG)
    run_results.append({'run': run+1, 'alpha': best_alpha, 'score': best_score})
    all_histories.append(history)
    print(f"  Run {run+1:2d}/{PSO_CONFIG['n_runs']}: α={best_alpha:.4f}, serendipity={best_score:.4f}")

pso_alphas = np.array([r['alpha'] for r in run_results])
pso_scores = np.array([r['score'] for r in run_results])

pso_mean_alpha = pso_alphas.mean()
pso_std_alpha = pso_alphas.std()
pso_mean_score = pso_scores.mean()

print(f"\n  PSO Mean: α={pso_mean_alpha:.4f} ± {pso_std_alpha:.4f}")
print(f"  Grid Best: α={best_grid_alpha:.2f}")
print(f"  Difference: {abs(pso_mean_alpha - best_grid_alpha):.4f}")

# ============================================================================
# STATISTICAL VALIDATION
# ============================================================================

print("\n[5/5] Statistical validation...")

closest_to_pso = min(alpha_grid, key=lambda a: abs(a - pso_mean_alpha))
baseline_alpha = alpha_grid[0]

pso_per_user = grid_serendipity_per_user[closest_to_pso]
baseline_per_user = grid_serendipity_per_user[baseline_alpha]

t_stat, p_value = ttest_rel(pso_per_user, baseline_per_user)
mean_diff = pso_per_user.mean() - baseline_per_user.mean()
cohen_d = mean_diff / np.std(pso_per_user - baseline_per_user)

print(f"\n  Optimal (α={closest_to_pso:.2f}) vs Baseline (α={baseline_alpha:.2f}):")
print(f"    Mean difference: {mean_diff:+.4f}")
print(f"    t={t_stat:.3f}, p={p_value:.4f}, d={cohen_d:.3f}")

agreement = abs(pso_mean_alpha - best_grid_alpha)
if agreement < 0.05:
    print(f"\n  ✓✓✓ PSO VALIDATES GRID SEARCH!")
    print(f"      Both methods agree: α ≈ {pso_mean_alpha:.2f}")
else:
    print(f"\n  ⚠ Methods show difference of {agreement:.2f}")

# ============================================================================
# SAVE & VISUALIZE
# ============================================================================

results = {
    'grid': {
        'alphas': grid_alphas.tolist(),
        'scores': grid_scores.tolist(),
        'best_alpha': float(best_grid_alpha),
        'best_score': float(best_grid_score),
    },
    'pso': {
        'runs': run_results,
        'mean_alpha': float(pso_mean_alpha),
        'std_alpha': float(pso_std_alpha),
        'mean_score': float(pso_mean_score),
    },
    'stats': {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohen_d),
        'methods_agree': bool(agreement < 0.05),
    }
}

pickle.dump(results, open('pso_fair_results.pkl', 'wb'))
print("\n✓ Saved: pso_fair_results.pkl")

# Quick visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(alpha_dense, scores_dense, 'b-', alpha=0.5, label='PCHIP')
ax1.scatter(grid_alphas, grid_scores, s=80, c='blue', zorder=5, label='Grid')
ax1.scatter(pso_alphas, pso_scores, s=60, c='red', alpha=0.6, label='PSO')
ax1.axvline(pso_mean_alpha, c='red', ls='--', label=f'PSO mean: {pso_mean_alpha:.3f}')
ax1.set_xlabel('α')
ax1.set_ylabel('Serendipity')
ax1.set_title('PSO Validation')
ax1.legend()
ax1.grid(True, alpha=0.3)

for h in all_histories:
    ax2.plot(h, 'r-', alpha=0.3, linewidth=1)
ax2.plot(np.mean(all_histories, axis=0), 'r-', linewidth=2, label='Mean')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Best Fitness')
ax2.set_title('PSO Convergence')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pso_fair_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: pso_fair_analysis.png")

print(f"\n{'='*80}")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}\n")