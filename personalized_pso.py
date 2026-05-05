"""
Personalized PSO Analysis
==========================

Runs individual PSO for each user to find personalized optimal alpha,
then compares against global alpha on per-user serendipity.

Requires:
- recommendations_fair_complete.pkl
- user_E_u.pkl (for metadata)

Runtime: ~5-10 minutes
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.stats import ttest_rel, wilcoxon
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PERSONALIZED PSO ANALYSIS")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1/7] Loading data...")

recs = pickle.load(open('recommendations_fair_complete.pkl', 'rb'))
alpha_values = sorted(recs.keys())
all_users = list(recs[alpha_values[0]].keys())

print(f"  ✓ {len(alpha_values)} alpha values: {alpha_values}")
print(f"  ✓ {len(all_users)} users")

DISTANCE_THRESHOLD = 0.7
CF_THRESHOLD = 1.8
GLOBAL_ALPHA = 0.35  # From Experiment 2

# ============================================================================
# COMPUTE PER-USER SERENDIPITY AT EACH ALPHA
# ============================================================================

print("\n[2/7] Computing per-user serendipity matrix...")

def calculate_serendipity(rec_list):
    if not rec_list:
        return 0.0
    n_ser = sum(1 for r in rec_list
                if r['distance'] > DISTANCE_THRESHOLD
                and r['cf_score'] > CF_THRESHOLD)
    return n_ser / len(rec_list)

# user_alpha_matrix[user_id][alpha] = serendipity score
user_alpha_matrix = {}
for user_id in all_users:
    user_alpha_matrix[user_id] = {}
    for alpha in alpha_values:
        s = calculate_serendipity(recs[alpha][user_id])
        user_alpha_matrix[user_id][alpha] = s

print(f"  ✓ Built {len(all_users)} x {len(alpha_values)} serendipity matrix")

# Quick check: how many users already hit 1.0 at global alpha
n_perfect_global = sum(
    1 for u in all_users 
    if user_alpha_matrix[u].get(GLOBAL_ALPHA, 0) == 1.0
)
print(f"\n  Users already at serendipity=1.0 under global α={GLOBAL_ALPHA}: "
      f"{n_perfect_global}/{len(all_users)}")
print(f"  Users who might benefit from personalization: "
      f"{len(all_users) - n_perfect_global}/{len(all_users)}")

# ============================================================================
# PSO IMPLEMENTATION
# ============================================================================

def run_pso_for_user(fitness_fn, n_particles=20, n_iterations=30, seed=42):
    """
    Run PSO to maximize fitness_fn over alpha in [0, 1].
    Returns (best_alpha, best_fitness, convergence_history)
    """
    np.random.seed(seed)
    
    # PSO parameters (constriction coefficients - same as Experiment 2)
    w = 0.729
    c1 = 1.49445
    c2 = 1.49445
    
    # Initialize particles
    positions = np.random.uniform(0, 1, n_particles)
    velocities = np.random.uniform(-0.1, 0.1, n_particles)
    
    personal_best_pos = positions.copy()
    personal_best_fit = np.array([fitness_fn(p) for p in positions])
    
    global_best_idx = np.argmax(personal_best_fit)
    global_best_pos = personal_best_pos[global_best_idx]
    global_best_fit = personal_best_fit[global_best_idx]
    
    convergence = [global_best_fit]
    
    for iteration in range(n_iterations):
        r1 = np.random.uniform(0, 1, n_particles)
        r2 = np.random.uniform(0, 1, n_particles)
        
        # Update velocities
        velocities = (w * velocities +
                     c1 * r1 * (personal_best_pos - positions) +
                     c2 * r2 * (global_best_pos - positions))
        
        # Update positions
        positions = np.clip(positions + velocities, 0, 1)
        
        # Evaluate fitness
        fitness = np.array([fitness_fn(p) for p in positions])
        
        # Update personal bests
        improved = fitness > personal_best_fit
        personal_best_pos[improved] = positions[improved]
        personal_best_fit[improved] = fitness[improved]
        
        # Update global best
        best_idx = np.argmax(personal_best_fit)
        if personal_best_fit[best_idx] > global_best_fit:
            global_best_pos = personal_best_pos[best_idx]
            global_best_fit = personal_best_fit[best_idx]
        
        convergence.append(global_best_fit)
    
    return global_best_pos, global_best_fit, convergence

# ============================================================================
# RUN PERSONALIZED PSO FOR EACH USER
# ============================================================================

print("\n[3/7] Running personalized PSO for each user...")
print("  (20 particles, 30 iterations per user)")

personalized_results = {}

for i, user_id in enumerate(all_users):
    if (i + 1) % 20 == 0:
        print(f"  Progress: {i+1}/{len(all_users)} users")
    
    # Get this user's serendipity scores
    user_scores = np.array([user_alpha_matrix[user_id][a] for a in alpha_values])
    
    # Handle flat landscapes (user already at 1.0 everywhere in plateau)
    if np.max(user_scores) == 0:
        # User gets no serendipity at any alpha - just use global
        personalized_results[user_id] = {
            'personalized_alpha': GLOBAL_ALPHA,
            'personalized_serendipity': 0.0,
            'global_serendipity': 0.0,
            'improvement': 0.0,
            'convergence': [0.0] * 31,
            'flat_landscape': True
        }
        continue
    
    # Fit PCHIP interpolation of this user's curve
    try:
        pchip = PchipInterpolator(alpha_values, user_scores)
        fitness_fn = lambda a: float(np.clip(pchip(a), 0, 1))
    except Exception:
        # Fallback: linear interpolation
        fitness_fn = lambda a: float(np.interp(a, alpha_values, user_scores))
    
    # Run PSO
    best_alpha, best_fitness, convergence = run_pso_for_user(
        fitness_fn, 
        n_particles=20, 
        n_iterations=30,
        seed=42 + i  # Different seed per user
    )
    
    # Get global alpha serendipity for this user
    global_serendipity = user_alpha_matrix[user_id].get(GLOBAL_ALPHA, 0.0)
    
    # Get personalized serendipity - use nearest grid point for fair comparison
    nearest_alpha = min(alpha_values, key=lambda a: abs(a - best_alpha))
    personalized_serendipity = user_alpha_matrix[user_id][nearest_alpha]
    
    personalized_results[user_id] = {
        'personalized_alpha': best_alpha,
        'nearest_grid_alpha': nearest_alpha,
        'personalized_serendipity': personalized_serendipity,
        'global_serendipity': global_serendipity,
        'improvement': personalized_serendipity - global_serendipity,
        'convergence': convergence,
        'flat_landscape': False
    }

print(f"  ✓ Completed PSO for all {len(all_users)} users")

# ============================================================================
# AGGREGATE RESULTS
# ============================================================================

print("\n[4/7] Aggregating results...")

results_df = pd.DataFrame([
    {
        'user_id': uid,
        'personalized_alpha': r['personalized_alpha'],
        'personalized_serendipity': r['personalized_serendipity'],
        'global_serendipity': r['global_serendipity'],
        'improvement': r['improvement'],
        'flat_landscape': r['flat_landscape']
    }
    for uid, r in personalized_results.items()
])

print("\n  SUMMARY STATISTICS:")
print(f"  Global α={GLOBAL_ALPHA} - Mean serendipity: "
      f"{results_df['global_serendipity'].mean():.4f} "
      f"(std={results_df['global_serendipity'].std():.4f})")
print(f"  Personalized α      - Mean serendipity: "
      f"{results_df['personalized_serendipity'].mean():.4f} "
      f"(std={results_df['personalized_serendipity'].std():.4f})")
print(f"  Mean improvement: {results_df['improvement'].mean():.4f}")
print(f"  Users improved: "
      f"{(results_df['improvement'] > 0).sum()}/{len(all_users)}")
print(f"  Users unchanged: "
      f"{(results_df['improvement'] == 0).sum()}/{len(all_users)}")
print(f"  Users worse: "
      f"{(results_df['improvement'] < 0).sum()}/{len(all_users)}")

print(f"\n  Personalized α distribution:")
print(f"  Mean:   {results_df['personalized_alpha'].mean():.4f}")
print(f"  Std:    {results_df['personalized_alpha'].std():.4f}")
print(f"  Median: {results_df['personalized_alpha'].median():.4f}")
print(f"  Min:    {results_df['personalized_alpha'].min():.4f}")
print(f"  Max:    {results_df['personalized_alpha'].max():.4f}")

# ============================================================================
# STATISTICAL TESTS
# ============================================================================

print("\n[5/7] Running statistical tests...")

global_scores = results_df['global_serendipity'].values
personalized_scores = results_df['personalized_serendipity'].values

# Paired t-test
t_stat, p_val = ttest_rel(personalized_scores, global_scores)
mean_diff = np.mean(personalized_scores - global_scores)
cohen_d = mean_diff / np.std(personalized_scores - global_scores) if np.std(personalized_scores - global_scores) > 0 else 0

# Wilcoxon signed-rank (non-parametric, robust to ceiling effects)
try:
    w_stat, w_pval = wilcoxon(personalized_scores, global_scores)
except ValueError:
    w_stat, w_pval = np.nan, np.nan

print(f"\n  Paired t-test (personalized vs global α={GLOBAL_ALPHA}):")
print(f"    Mean difference: {mean_diff:+.4f}")
print(f"    t({len(all_users)-1}) = {t_stat:.4f}, p = {p_val:.4f}")
print(f"    Cohen's d = {cohen_d:.4f}")
print(f"\n  Wilcoxon signed-rank test:")
print(f"    W = {w_stat}, p = {w_pval:.4f}")

if p_val < 0.05:
    print(f"\n  → Personalized α significantly outperforms global α (p<0.05)")
else:
    print(f"\n  → No significant difference between personalized and global α (p≥0.05)")
    print(f"     This confirms global α={GLOBAL_ALPHA} is robust across users")

# Focus on users who didn't reach ceiling under global alpha
suboptimal_users = results_df[results_df['global_serendipity'] < 1.0]
print(f"\n  Among {len(suboptimal_users)} users below ceiling under global α:")
if len(suboptimal_users) > 0:
    print(f"    Mean improvement: "
          f"{suboptimal_users['improvement'].mean():+.4f}")
    print(f"    Users who improved: "
          f"{(suboptimal_users['improvement'] > 0).sum()}/{len(suboptimal_users)}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n[6/7] Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- Plot 1: Distribution of personalized alpha values ---
ax1 = axes[0, 0]
ax1.hist(results_df['personalized_alpha'], bins=20, 
         color='#3498db', edgecolor='black', alpha=0.7)
ax1.axvline(GLOBAL_ALPHA, color='red', linewidth=2.5, 
            linestyle='--', label=f'Global α={GLOBAL_ALPHA}')
ax1.axvline(results_df['personalized_alpha'].mean(), 
            color='orange', linewidth=2.5,
            linestyle='-', 
            label=f"Mean personalized α={results_df['personalized_alpha'].mean():.3f}")
ax1.set_xlabel('Personalized α', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Users', fontsize=12, fontweight='bold')
ax1.set_title('Distribution of Personalized Optimal α', 
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# --- Plot 2: Global vs personalized serendipity scatter ---
ax2 = axes[0, 1]
ax2.scatter(results_df['global_serendipity'], 
            results_df['personalized_serendipity'],
            alpha=0.6, color='#2ecc71', edgecolor='black', s=60)
ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='No improvement line')
ax2.set_xlabel(f'Global α={GLOBAL_ALPHA} Serendipity', 
               fontsize=12, fontweight='bold')
ax2.set_ylabel('Personalized α Serendipity', fontsize=12, fontweight='bold')
ax2.set_title('Global vs Personalized Serendipity\n(points above line = improved)', 
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-0.05, 1.05)
ax2.set_ylim(-0.05, 1.05)

# --- Plot 3: Improvement distribution ---
ax3 = axes[1, 0]
improvements = results_df['improvement']
colors_imp = ['#e74c3c' if x < 0 else '#95a5a6' if x == 0 else '#2ecc71' 
              for x in improvements]
ax3.bar(range(len(improvements)), 
        sorted(improvements), 
        color=sorted(colors_imp, key=lambda c: 
                     {'#e74c3c': 0, '#95a5a6': 1, '#2ecc71': 2}[c]),
        edgecolor='black', linewidth=0.5)
ax3.axhline(0, color='black', linewidth=1.5)
ax3.axhline(improvements.mean(), color='blue', linewidth=2,
            linestyle='--', label=f'Mean={improvements.mean():.3f}')
ax3.set_xlabel('Users (sorted by improvement)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Serendipity Improvement', fontsize=12, fontweight='bold')
ax3.set_title('Per-User Improvement: Personalized vs Global α', 
              fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# --- Plot 4: Sample convergence curves for suboptimal users ---
ax4 = axes[1, 1]
suboptimal_ids = suboptimal_users['user_id'].tolist()[:10]  # Show up to 10

if suboptimal_ids:
    for uid in suboptimal_ids:
        conv = personalized_results[uid]['convergence']
        ax4.plot(conv, alpha=0.5, linewidth=1.5, color='#3498db')
    
    # Average convergence
    all_conv = [personalized_results[uid]['convergence'] 
                for uid in suboptimal_ids]
    mean_conv = np.mean(all_conv, axis=0)
    ax4.plot(mean_conv, color='red', linewidth=3, 
             label='Mean convergence', zorder=10)
    
    ax4.set_xlabel('PSO Iteration', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Best Fitness (Serendipity)', fontsize=12, fontweight='bold')
    ax4.set_title('PSO Convergence for Suboptimal Users\n'
                  f'(users below ceiling under global α={GLOBAL_ALPHA})', 
                  fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'All users at ceiling\nunder global α', 
             ha='center', va='center', fontsize=14,
             transform=ax4.transAxes)
    ax4.set_title('PSO Convergence', fontsize=13, fontweight='bold')

plt.suptitle('Personalized PSO Analysis', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('personalized_pso_results.png', dpi=300, bbox_inches='tight')
plt.savefig('personalized_pso_results.pdf', bbox_inches='tight')
print("  ✓ Saved: personalized_pso_results.png/pdf")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n[7/7] Saving results...")

results_df.to_csv('personalized_pso_results.csv', index=False, float_format='%.4f')
print("  ✓ Saved: personalized_pso_results.csv")

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print(f"\n  Global α={GLOBAL_ALPHA}:")
print(f"    Mean serendipity: {results_df['global_serendipity'].mean():.4f}")
print(f"    Std:              {results_df['global_serendipity'].std():.4f}")
print(f"\n  Personalized α (PSO):")
print(f"    Mean serendipity: {results_df['personalized_serendipity'].mean():.4f}")
print(f"    Std:              {results_df['personalized_serendipity'].std():.4f}")
print(f"    Mean α:           {results_df['personalized_alpha'].mean():.4f}")
print(f"    Std α:            {results_df['personalized_alpha'].std():.4f}")
print(f"\n  Statistical test:")
print(f"    t({len(all_users)-1})={t_stat:.3f}, p={p_val:.4f}, d={cohen_d:.4f}")
print(f"    Wilcoxon: W={w_stat}, p={w_pval:.4f}")
print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)