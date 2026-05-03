"""
Generate Thesis Tables & Figures
=================================

Creates publication-ready tables and figures for thesis from
fair generation + PSO results.

Outputs:
- thesis_table_grid_results.csv (Table 1)
- thesis_table_pso_results.csv (Table 2)
- thesis_table_statistics.csv (Table 3)
- thesis_figure_main.png/pdf (Main results figure)

Runtime: ~1 minute
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

print("=" * 80)
print("GENERATING THESIS TABLES & FIGURES")
print("=" * 80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("[1/4] Loading results...")

recs = pickle.load(open('recommendations_fair_complete.pkl', 'rb'))
pso_results = pickle.load(open('pso_fair_results.pkl', 'rb'))

alpha_grid = sorted(recs.keys())
print(f"  ✓ Loaded fair generation: {len(alpha_grid)} α values")
print(f"  ✓ Loaded PSO results: {len(pso_results['pso']['runs'])} runs")

# ============================================================================
# TABLE 1: GRID SEARCH RESULTS
# ============================================================================

print("\n[2/4] Creating Table 1: Grid Search Results...")

THRESHOLDS = {'distance': 0.7, 'cf_score': 1.8}

def compute_serendipity(recs_dict):
    user_scores = []
    for user_id, rec_list in recs_dict.items():
        n_ser = sum(1 for r in rec_list 
                   if r['distance'] > THRESHOLDS['distance'] 
                   and r['cf_score'] > THRESHOLDS['cf_score'])
        user_scores.append(n_ser / len(rec_list) if rec_list else 0)
    return np.array(user_scores)

table1_data = []
per_user_data = {}

for alpha in alpha_grid:
    scores = compute_serendipity(recs[alpha])
    per_user_data[alpha] = scores
    
    method_name = {
        0.0: 'Pure CF',
        0.15: 'CF-Heavy',
        0.25: 'CF-Biased',
        0.3: 'CF-Biased High',
        0.35: 'Near-Balanced',
        0.4: 'Near-Balanced High',
        0.5: 'Balanced',
        0.65: 'Distance-Biased Low',
        0.75: 'Distance-Biased',
        0.85: 'Distance-Heavy',
        1.0: 'Pure Distance'
    }.get(alpha, f'α={alpha}')
    
    table1_data.append({
        'Method': method_name,
        'Alpha': alpha,
        'Mean': scores.mean(),
        'Std_Dev': scores.std(),
        'Median': np.median(scores),
        'N_Users': len(scores)
    })

df_table1 = pd.DataFrame(table1_data)
df_table1.to_csv('thesis_table_grid_results.csv', index=False, float_format='%.4f')
print(f"  ✓ Saved: thesis_table_grid_results.csv")

# ============================================================================
# TABLE 2: PSO RESULTS
# ============================================================================

print("\n[3/4] Creating Table 2: PSO Results...")

pso_runs = pso_results['pso']['runs']
table2_data = []

for run_data in pso_runs:
    table2_data.append({
        'Run': run_data['run'],
        'Converged_Alpha': run_data['alpha'],
        'Serendipity': run_data['score']
    })

# Add summary row
table2_data.append({
    'Run': 'Mean ± SD',
    'Converged_Alpha': f"{pso_results['pso']['mean_alpha']:.4f} ± {pso_results['pso']['std_alpha']:.4f}",
    'Serendipity': f"{pso_results['pso']['mean_score']:.4f}"
})

df_table2 = pd.DataFrame(table2_data)
df_table2.to_csv('thesis_table_pso_results.csv', index=False)
print(f"  ✓ Saved: thesis_table_pso_results.csv")

# ============================================================================
# TABLE 3: STATISTICAL TESTS
# ============================================================================

print("\n[4/4] Creating Table 3: Statistical Tests...")

# Find optimal and baseline
best_idx = df_table1['Mean'].idxmax()
optimal_alpha = df_table1.loc[best_idx, 'Alpha']
baseline_alpha = 0.0

# PSO closest
pso_mean_alpha = pso_results['pso']['mean_alpha']
pso_closest = min(alpha_grid, key=lambda a: abs(a - pso_mean_alpha))

# Compute stats
optimal_scores = per_user_data[optimal_alpha]
baseline_scores = per_user_data[baseline_alpha]
pso_scores = per_user_data[pso_closest]

# Optimal vs baseline
t1, p1 = ttest_rel(optimal_scores, baseline_scores)
d1 = (optimal_scores.mean() - baseline_scores.mean()) / np.std(optimal_scores - baseline_scores)

# PSO vs baseline
t2, p2 = ttest_rel(pso_scores, baseline_scores)
d2 = (pso_scores.mean() - baseline_scores.mean()) / np.std(pso_scores - baseline_scores)

table3_data = [
    {
        'Comparison': f'Optimal (α={optimal_alpha}) vs Pure CF (α={baseline_alpha})',
        'Mean_Diff': optimal_scores.mean() - baseline_scores.mean(),
        't_statistic': t1,
        'p_value': p1,
        'Cohens_d': d1,
        'Significant': 'Yes' if p1 < 0.05 else 'No'
    },
    {
        'Comparison': f'PSO (α={pso_closest}) vs Pure CF (α={baseline_alpha})',
        'Mean_Diff': pso_scores.mean() - baseline_scores.mean(),
        't_statistic': t2,
        'p_value': p2,
        'Cohens_d': d2,
        'Significant': 'Yes' if p2 < 0.05 else 'No'
    }
]

df_table3 = pd.DataFrame(table3_data)
df_table3.to_csv('thesis_table_statistics.csv', index=False, float_format='%.4f')
print(f"  ✓ Saved: thesis_table_statistics.csv")

# ============================================================================
# MAIN FIGURE
# ============================================================================

print("\nCreating main thesis figure...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Panel A: Main curve
ax1 = fig.add_subplot(gs[0, :])
alphas = df_table1['Alpha'].values
means = df_table1['Mean'].values
stds = df_table1['Std_Dev'].values

ax1.plot(alphas, means, 'o-', linewidth=2.5, markersize=10, color='steelblue', label='Grid search')
ax1.fill_between(alphas, means-stds, means+stds, alpha=0.2, color='steelblue')

# Mark optimal
opt_idx = df_table1['Mean'].idxmax()
opt_alpha = alphas[opt_idx]
opt_mean = means[opt_idx]
ax1.plot(opt_alpha, opt_mean, 'r*', markersize=25, label=f'Optimal: α={opt_alpha:.2f}', zorder=5)

# PSO points
pso_alphas = [r['alpha'] for r in pso_runs]
pso_scores = [r['score'] for r in pso_runs]
ax1.scatter(pso_alphas, pso_scores, s=60, c='red', alpha=0.5, label='PSO runs', zorder=4)

ax1.axhline(baseline_scores.mean(), color='gray', linestyle=':', linewidth=2, 
           label=f'Pure CF baseline ({baseline_scores.mean():.3f})')

ax1.set_xlabel('Alpha (α) - Weight on Unexpectedness', fontsize=12, fontweight='bold')
ax1.set_ylabel('Serendipity Score', fontsize=12, fontweight='bold')
ax1.set_title('A. Serendipity Across Weighting Strategies (Grid Search + PSO Validation)', 
             fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, loc='lower left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(0, 1.05)

# Panel B: PSO convergence
ax2 = fig.add_subplot(gs[1, 0])
pso_alpha_vals = [r['alpha'] for r in pso_runs]
runs = list(range(1, len(pso_runs)+1))

ax2.bar(runs, pso_alpha_vals, color='coral', alpha=0.7, edgecolor='darkred')
ax2.axhline(pso_mean_alpha, color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {pso_mean_alpha:.3f}±{pso_results["pso"]["std_alpha"]:.3f}')
ax2.axhline(opt_alpha, color='blue', linestyle='--', linewidth=2,
           label=f'Grid optimal: {opt_alpha:.2f}')

ax2.set_xlabel('PSO Run', fontsize=11, fontweight='bold')
ax2.set_ylabel('Converged Alpha', fontsize=11, fontweight='bold')
ax2.set_title('B. PSO Convergence Across 20 Runs', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 1)

# Panel C: Effect sizes
ax3 = fig.add_subplot(gs[1, 1])

comparisons = ['Optimal\nvs\nPure CF', 'PSO\nvs\nPure CF']
effect_sizes = [d1, d2]
p_values = [p1, p2]

colors = ['green' if p < 0.001 else 'orange' if p < 0.05 else 'red' for p in p_values]
bars = ax3.bar(comparisons, effect_sizes, color=colors, alpha=0.7, edgecolor='black')

# Add significance markers
for i, (bar, p) in enumerate(zip(bars, p_values)):
    height = bar.get_height()
    sig_marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    ax3.text(bar.get_x() + bar.get_width()/2, height + 0.05, sig_marker,
            ha='center', fontsize=14, fontweight='bold')

ax3.axhline(0.2, color='gray', linestyle=':', label='Small effect')
ax3.axhline(0.5, color='gray', linestyle='--', label='Medium effect')
ax3.axhline(0.8, color='gray', linestyle='-', label='Large effect')

ax3.set_ylabel("Cohen's d (Effect Size)", fontsize=11, fontweight='bold')
ax3.set_title("C. Statistical Effect Sizes", fontsize=12, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, max(effect_sizes) + 0.2)

plt.savefig('thesis_figure_main.png', dpi=300, bbox_inches='tight')
plt.savefig('thesis_figure_main.pdf', bbox_inches='tight')
print("  ✓ Saved: thesis_figure_main.png/pdf")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("THESIS MATERIALS GENERATED")
print("=" * 80)
print()
print("TABLES:")
print("  ✓ thesis_table_grid_results.csv    (Table 1: Grid search)")
print("  ✓ thesis_table_pso_results.csv     (Table 2: PSO runs)")
print("  ✓ thesis_table_statistics.csv      (Table 3: Statistical tests)")
print()
print("FIGURES:")
print("  ✓ thesis_figure_main.png/pdf       (Main results figure)")
print()
print("KEY FINDINGS:")
print(f"  Grid optimal: α={opt_alpha:.2f} (serendipity={opt_mean:.3f})")
print(f"  PSO mean:     α={pso_mean_alpha:.3f}±{pso_results['pso']['std_alpha']:.3f}")
print(f"  Agreement:    {abs(opt_alpha - pso_mean_alpha):.3f} difference")
print(f"  Statistical:  t={t1:.2f}, p={p1:.4f}, d={d1:.2f}")
print()
print("=" * 80)