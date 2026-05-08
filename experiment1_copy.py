"""
Complete Pareto Front Analysis (CORRECTED)
===========================================

Analyzes multi-objective trade-offs between unexpectedness and relevance.
Identifies plateau of optimal solutions rather than single knee point.

Input: recommendations_fair_complete.pkl
Output: 
  - pareto_objectives.csv
  - pareto_front.csv
  - pareto_analysis_corrected.pdf/png
  
Runtime: ~1 minute
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("="*80)
print("PARETO FRONT ANALYSIS (CORRECTED)")
print("="*80)

# ============================================================================
# STEP 1: Calculate Objectives
# ============================================================================

print("\n[1/5] Calculating objectives...")

recs = pickle.load(open('recommendations_fair_complete.pkl', 'rb'))
alpha_values = sorted(recs.keys())

objectives = {
    'alpha': [],
    'mean_unexpectedness': [],
    'mean_relevance': [],
    'serendipity': []
}

for alpha in alpha_values:
    all_unexpectedness = []
    all_relevance = []
    all_serendipity = []
    
    for user_id, user_recs in recs[alpha].items():
        distances = [r['distance'] for r in user_recs]
        all_unexpectedness.append(np.mean(distances))
        
        cf_scores = [r['cf_score'] for r in user_recs]
        all_relevance.append(np.mean(cf_scores))
        
        serendipitous_count = sum(
            1 for r in user_recs 
            if r['distance'] > 0.7 and r['cf_score'] > 1.8
        )
        all_serendipity.append(serendipitous_count / len(user_recs))
    
    objectives['alpha'].append(alpha)
    objectives['mean_unexpectedness'].append(np.mean(all_unexpectedness))
    objectives['mean_relevance'].append(np.mean(all_relevance))
    objectives['serendipity'].append(np.mean(all_serendipity))

df_objectives = pd.DataFrame(objectives)
df_objectives.to_csv('pareto_objectives.csv', index=False)

print("  ✓ Objectives calculated")
print("\n" + df_objectives.to_string(index=False))

# ============================================================================
# STEP 2: Identify Pareto Front
# ============================================================================

print("\n[2/5] Identifying Pareto front...")

def is_pareto_optimal(index, objectives_df):
    """Check if solution is non-dominated (Pareto-optimal)"""
    current_unexpectedness = objectives_df.loc[index, 'mean_unexpectedness']
    current_relevance = objectives_df.loc[index, 'mean_relevance']
    
    for i, row in objectives_df.iterrows():
        if i == index:
            continue
        
        other_unexpectedness = row['mean_unexpectedness']
        other_relevance = row['mean_relevance']
        
        # Does other solution dominate current?
        if (other_unexpectedness >= current_unexpectedness and 
            other_relevance >= current_relevance and
            (other_unexpectedness > current_unexpectedness or 
             other_relevance > current_relevance)):
            return False
    
    return True

df_objectives['pareto_optimal'] = [
    is_pareto_optimal(i, df_objectives) 
    for i in df_objectives.index
]

pareto_front = df_objectives[df_objectives['pareto_optimal']].copy()
pareto_front = pareto_front.sort_values('mean_unexpectedness')

print(f"  ✓ Pareto front identified")
print(f"  Total solutions: {len(df_objectives)}")
print(f"  Pareto-optimal: {len(pareto_front)}")
print(f"  Dominated: {len(df_objectives) - len(pareto_front)}")

pareto_front.to_csv('pareto_front.csv', index=False)

# ============================================================================
# STEP 3: Plateau Detection (CORRECTED)
# ============================================================================

print("\n[3/5] Detecting plateau...")

max_serendipity = df_objectives['serendipity'].max()
plateau = df_objectives[df_objectives['serendipity'] >= max_serendipity * 0.999].copy()

print(f"\nMaximum serendipity: {max_serendipity:.3f}")
print(f"\nSerendipity Plateau:")
print(f"  α range: [{plateau['alpha'].min():.2f}, {plateau['alpha'].max():.2f}]")
print(f"  Number of solutions: {len(plateau)}")

print(f"\nSolutions in plateau:")
print(plateau[['alpha', 'mean_unexpectedness', 'mean_relevance', 'serendipity']].to_string(index=False))

# Representative knee point: middle of plateau
# (since plateau has 3-4 solutions, pick the middle one)
middle_idx = len(plateau) // 2
knee_point = plateau.iloc[middle_idx]

print(f"\n{'='*80}")
print("REPRESENTATIVE KNEE POINT (Center of Plateau)")
print(f"{'='*80}")
print(f"  Selected from {len(plateau)} plateau solutions")
print(f"  α = {knee_point['alpha']:.2f}")
print(f"  Weight on relevance: {(1-knee_point['alpha'])*100:.0f}%")
print(f"  Weight on unexpectedness: {knee_point['alpha']*100:.0f}%")
print(f"  Mean unexpectedness: {knee_point['mean_unexpectedness']:.3f}")
print(f"  Mean relevance: {knee_point['mean_relevance']:.3f}")
print(f"  Serendipity outcome: {knee_point['serendipity']:.3f}")

# ============================================================================
# STEP 4: Visualization
# ============================================================================

print("\n[4/5] Generating visualizations...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- Panel 1: Pareto Front ---

# Plot all solutions
ax1.plot(df_objectives['mean_unexpectedness'],
        df_objectives['mean_relevance'],
        'o-', linewidth=2.5, markersize=10, color='steelblue',
        label='Pareto Front (all non-dominated)', zorder=2)

# Highlight plateau
plateau_mask = (df_objectives['alpha'] >= plateau['alpha'].min()) & \
               (df_objectives['alpha'] <= plateau['alpha'].max())
plateau_data = df_objectives[plateau_mask]

ax1.scatter(plateau_data['mean_unexpectedness'],
           plateau_data['mean_relevance'],
           s=400, marker='*', color='gold',
           edgecolor='black', linewidth=2,
           label=f'Plateau (α∈[{plateau["alpha"].min():.2f},{plateau["alpha"].max():.2f}])',
           zorder=10)

# Annotate extremes
pure_cf = df_objectives[df_objectives['alpha'] == 0.0].iloc[0]
pure_dist = df_objectives[df_objectives['alpha'] == 1.0].iloc[0]

ax1.annotate('Pure CF\n(α=0.0)', 
            (pure_cf['mean_unexpectedness'], pure_cf['mean_relevance']),
            xytext=(-60, 20), textcoords='offset points',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
            arrowprops=dict(arrowstyle='->', lw=2))

ax1.annotate('Pure Distance\n(α=1.0)', 
            (pure_dist['mean_unexpectedness'], pure_dist['mean_relevance']),
            xytext=(10, -30), textcoords='offset points',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
            arrowprops=dict(arrowstyle='->', lw=2))

# Annotate representative knee
ax1.annotate(f'Knee Point\n(α={knee_point["alpha"]:.2f})', 
            (knee_point['mean_unexpectedness'], knee_point['mean_relevance']),
            xytext=(30, 30), textcoords='offset points',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='green'))

ax1.set_xlabel('Mean Unexpectedness (Distance)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Mean Relevance (CF Prediction)', fontsize=13, fontweight='bold')
ax1.set_title('Pareto Front: All Solutions Non-Dominated', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.3)

# --- Panel 2: Serendipity Outcome ---

ax2.plot(df_objectives['alpha'], 
        df_objectives['serendipity'],
        'o-', linewidth=3, markersize=10, color='steelblue',
        label='Serendipity')

# Highlight plateau region
ax2.axvspan(plateau['alpha'].min(), plateau['alpha'].max(),
           color='gold', alpha=0.3, label='Plateau Region')

# Plot plateau points
ax2.scatter(plateau_data['alpha'],
           plateau_data['serendipity'],
           s=400, marker='*', color='gold',
           edgecolor='black', linewidth=2,
           zorder=10)

# Mark representative knee
ax2.axvline(knee_point['alpha'], color='green', linestyle='--', 
           linewidth=2.5, alpha=0.7, label=f'Knee (α={knee_point["alpha"]:.2f})')

ax2.set_xlabel('α (Weight on Unexpectedness)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Serendipity', fontsize=13, fontweight='bold')
ax2.set_title('Serendipity Maximized at Plateau', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.05)
ax2.set_xlim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('pareto_analysis_corrected.pdf', dpi=300, bbox_inches='tight')
plt.savefig('pareto_analysis_corrected.png', dpi=300, bbox_inches='tight')

print("  ✓ Saved: pareto_analysis_corrected.pdf/png")

# ============================================================================
# STEP 5: Summary Report
# ============================================================================

print("\n[5/5] Generating summary report...")

print("\n" + "="*80)
print("TRADE-OFF ANALYSIS")
print("="*80)

# Compare extremes vs knee
print(f"\nPure CF (α=0.0):")
print(f"  Unexpectedness: {pure_cf['mean_unexpectedness']:.3f}")
print(f"  Relevance: {pure_cf['mean_relevance']:.3f}")
print(f"  Serendipity: {pure_cf['serendipity']:.3f}")

print(f"\nKnee Point (α={knee_point['alpha']:.2f}):")
print(f"  Unexpectedness: {knee_point['mean_unexpectedness']:.3f} "
      f"(+{knee_point['mean_unexpectedness'] - pure_cf['mean_unexpectedness']:.3f})")
print(f"  Relevance: {knee_point['mean_relevance']:.3f} "
      f"({knee_point['mean_relevance'] - pure_cf['mean_relevance']:.3f})")
print(f"  Serendipity: {knee_point['serendipity']:.3f} "
      f"(+{knee_point['serendipity'] - pure_cf['serendipity']:.3f})")

print(f"\nPure Distance (α=1.0):")
print(f"  Unexpectedness: {pure_dist['mean_unexpectedness']:.3f}")
print(f"  Relevance: {pure_dist['mean_relevance']:.3f}")
print(f"  Serendipity: {pure_dist['serendipity']:.3f}")

# Calculate trade-offs
unexp_gain_pct = ((knee_point['mean_unexpectedness'] - pure_cf['mean_unexpectedness']) / 
                   pure_cf['mean_unexpectedness'] * 100)
rel_loss_pct = ((pure_cf['mean_relevance'] - knee_point['mean_relevance']) / 
                 pure_cf['mean_relevance'] * 100)

print(f"\nTrade-off at Knee Point:")
print(f"  Unexpectedness gain: +{unexp_gain_pct:.1f}%")
print(f"  Relevance sacrifice: -{rel_loss_pct:.1f}%")
if rel_loss_pct > 0:
    print(f"  Trade-off ratio: {unexp_gain_pct/rel_loss_pct:.2f}:1")
    print(f"  → {unexp_gain_pct:.1f}% unexpectedness gain for {rel_loss_pct:.1f}% relevance cost")

ser_gain_pct = ((knee_point['serendipity'] - pure_cf['serendipity']) / 
                 pure_cf['serendipity'] * 100)
print(f"  Serendipity improvement: +{ser_gain_pct:.1f}%")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nFiles generated:")
print("  - pareto_objectives.csv (all objectives)")
print("  - pareto_front.csv (Pareto-optimal solutions)")
print("  - pareto_analysis_corrected.pdf (visualization)")
print("  - pareto_analysis_corrected.png (visualization)")