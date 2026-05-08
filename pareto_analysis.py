"""
Complete Pareto Front Analysis
===============================

Analyzes multi-objective trade-offs between unexpectedness and relevance.

Input: recommendations_fair_complete.pkl (from fair generation)
Output: 
  - pareto_objectives.csv (all objectives)
  - pareto_front.csv (Pareto-optimal solutions)
  - pareto_front_analysis.pdf/png (visualization)
  - Console report with knee point analysis

Runtime: ~1 minute
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

print("="*80)
print("PARETO FRONT ANALYSIS")
print("="*80)


# Step 1: Calculate objectives
"""
Step 1: Calculate Mean Unexpectedness and Mean Relevance
=========================================================

For each α value, calculate the two objectives across all users.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load your existing fair generation results
recs = pickle.load(open('recommendations_fair_complete.pkl', 'rb'))

# Alpha values you tested
alpha_values = sorted(recs.keys())

# Storage for objectives
objectives = {
    'alpha': [],
    'mean_unexpectedness': [],
    'mean_relevance': [],
    'serendipity': []
}

# For each alpha
for alpha in alpha_values:
    
    # Collect unexpectedness and relevance across all users
    all_unexpectedness = []
    all_relevance = []
    all_serendipity = []
    
    for user_id, user_recs in recs[alpha].items():
        
        # Extract distances (unexpectedness)
        distances = [r['distance'] for r in user_recs]
        mean_distance = np.mean(distances)
        all_unexpectedness.append(mean_distance)
        
        # Extract CF predictions (relevance)
        cf_scores = [r['cf_score'] for r in user_recs]
        mean_cf = np.mean(cf_scores)
        all_relevance.append(mean_cf)
        
        # Calculate serendipity for this user
        serendipitous_count = sum(
            1 for r in user_recs 
            if r['distance'] > 0.7 and r['cf_score'] > 1.8
        )
        user_serendipity = serendipitous_count / len(user_recs)
        all_serendipity.append(user_serendipity)
    
    # Store aggregated objectives
    objectives['alpha'].append(alpha)
    objectives['mean_unexpectedness'].append(np.mean(all_unexpectedness))
    objectives['mean_relevance'].append(np.mean(all_relevance))
    objectives['serendipity'].append(np.mean(all_serendipity))

# Convert to DataFrame
df_objectives = pd.DataFrame(objectives)
df_objectives.to_csv('pareto_objectives.csv', index=False)

print("Objectives calculated:")
print(df_objectives.to_string(index=False))

# Step 2: Identify Pareto front
"""
Step 2: Identify Pareto-Optimal Solutions
==========================================

A solution is Pareto-optimal (non-dominated) if no other solution
is better in both objectives simultaneously.

For maximization problems:
  Solution A dominates B if:
    f1(A) >= f1(B) AND f2(A) >= f2(B) AND at least one is strictly better
"""

def is_pareto_optimal(index, objectives_df):
    """
    Check if solution at given index is Pareto-optimal.
    
    Args:
        index: Index of solution to check
        objectives_df: DataFrame with 'mean_unexpectedness' and 'mean_relevance'
    
    Returns:
        True if solution is non-dominated (Pareto-optimal)
    """
    current_unexpectedness = objectives_df.loc[index, 'mean_unexpectedness']
    current_relevance = objectives_df.loc[index, 'mean_relevance']
    
    # Check if any other solution dominates this one
    for i, row in objectives_df.iterrows():
        if i == index:
            continue
        
        other_unexpectedness = row['mean_unexpectedness']
        other_relevance = row['mean_relevance']
        
        # Does the other solution dominate current?
        # (Better or equal in both, strictly better in at least one)
        if (other_unexpectedness >= current_unexpectedness and 
            other_relevance >= current_relevance and
            (other_unexpectedness > current_unexpectedness or 
             other_relevance > current_relevance)):
            return False  # Current solution is dominated
    
    return True  # No other solution dominates it


# Identify Pareto front
df_objectives['pareto_optimal'] = [
    is_pareto_optimal(i, df_objectives) 
    for i in df_objectives.index
]

# Extract Pareto front
pareto_front = df_objectives[df_objectives['pareto_optimal']].copy()
pareto_front = pareto_front.sort_values('mean_unexpectedness')

print("\nPareto Front:")
print(pareto_front[['alpha', 'mean_unexpectedness', 'mean_relevance', 'serendipity']].to_string(index=False))

# Save Pareto front
pareto_front.to_csv('pareto_front.csv', index=False)

# Step 3: Find knee point
"""
Step 3: Knee Point Detection
=============================

The knee point is where the marginal gain in one objective
equals the marginal loss in the other - the "best compromise".

Three methods:
  1. Maximum distance from line connecting extremes
  2. Maximum curvature
  3. Normalized trade-off ratio
"""

def knee_point_distance_method(pareto_df):
    """
    Find knee point as maximum distance from line connecting extremes.
    """
    # Get extreme points
    unexpectedness = pareto_df['mean_unexpectedness'].values
    relevance = pareto_df['mean_relevance'].values
    
    # Normalize to [0, 1]
    unexp_norm = (unexpectedness - unexpectedness.min()) / (unexpectedness.max() - unexpectedness.min())
    rel_norm = (relevance - relevance.min()) / (relevance.max() - relevance.min())
    
    # Line from worst to best (assume worst = min both, best = max both)
    # For our case: worst is likely α=1.0, best is somewhere in middle
    
    # Calculate distance from each point to diagonal line y = x
    distances = []
    for i in range(len(unexp_norm)):
        # Distance to line from (0,0) to (1,1)
        # Using formula: |ax + by + c| / sqrt(a² + b²)
        # Line: x - y = 0  (so a=1, b=-1, c=0)
        x, y = unexp_norm[i], rel_norm[i]
        dist = abs(x - y) / np.sqrt(2)
        distances.append(dist)
    
    knee_idx = np.argmax(distances)
    return pareto_df.iloc[knee_idx]


def knee_point_curvature_method(pareto_df):
    """
    Find knee point as maximum curvature.
    """
    from scipy.interpolate import UnivariateSpline
    
    unexpectedness = pareto_df['mean_unexpectedness'].values
    relevance = pareto_df['mean_relevance'].values
    
    # Fit spline
    spline = UnivariateSpline(unexpectedness, relevance, s=0)
    
    # Calculate second derivative (curvature)
    curvature = np.abs(spline.derivative(n=2)(unexpectedness))
    
    knee_idx = np.argmax(curvature)
    return pareto_df.iloc[knee_idx]


def knee_point_tradeoff_method(pareto_df):
    """
    Find knee point where trade-off ratio changes most.
    """
    unexpectedness = pareto_df['mean_unexpectedness'].values
    relevance = pareto_df['mean_relevance'].values
    
    # Calculate trade-off ratios (slope between consecutive points)
    trade_offs = []
    for i in range(len(unexpectedness) - 1):
        delta_unexp = unexpectedness[i+1] - unexpectedness[i]
        delta_rel = relevance[i+1] - relevance[i]
        
        if delta_unexp != 0:
            trade_off = abs(delta_rel / delta_unexp)
            trade_offs.append(trade_off)
        else:
            trade_offs.append(0)
    
    # Knee is where trade-off ratio changes most
    if len(trade_offs) > 1:
        ratio_changes = []
        for i in range(len(trade_offs) - 1):
            change = abs(trade_offs[i+1] - trade_offs[i])
            ratio_changes.append(change)
        
        knee_idx = np.argmax(ratio_changes) + 1  # +1 because we look at changes
        return pareto_df.iloc[knee_idx]
    else:
        return pareto_df.iloc[0]


# Apply all three methods
print("\n" + "="*80)
print("KNEE POINT DETECTION")
print("="*80)

knee_distance = knee_point_distance_method(pareto_front)
print(f"\nMethod 1 - Distance from line:")
print(f"  α = {knee_distance['alpha']:.2f}")
print(f"  Unexpectedness = {knee_distance['mean_unexpectedness']:.3f}")
print(f"  Relevance = {knee_distance['mean_relevance']:.3f}")
print(f"  Serendipity = {knee_distance['serendipity']:.3f}")

knee_curvature = knee_point_curvature_method(pareto_front)
print(f"\nMethod 2 - Maximum curvature:")
print(f"  α = {knee_curvature['alpha']:.2f}")
print(f"  Unexpectedness = {knee_curvature['mean_unexpectedness']:.3f}")
print(f"  Relevance = {knee_curvature['mean_relevance']:.3f}")
print(f"  Serendipity = {knee_curvature['serendipity']:.3f}")

knee_tradeoff = knee_point_tradeoff_method(pareto_front)
print(f"\nMethod 3 - Trade-off ratio:")
print(f"  α = {knee_tradeoff['alpha']:.2f}")
print(f"  Unexpectedness = {knee_tradeoff['mean_unexpectedness']:.3f}")
print(f"  Relevance = {knee_tradeoff['mean_relevance']:.3f}")
print(f"  Serendipity = {knee_tradeoff['serendipity']:.3f}")

# Consensus knee point (if methods agree)
knee_alphas = [knee_distance['alpha'], knee_curvature['alpha'], knee_tradeoff['alpha']]
print(f"\nConsensus: All methods identify α ∈ [{min(knee_alphas):.2f}, {max(knee_alphas):.2f}]")

# Step 4: Visualize
"""
Step 4: Visualize Pareto Front
===============================
"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- Panel 1: Pareto Front ---

# Plot all solutions
ax1.scatter(df_objectives['mean_unexpectedness'], 
           df_objectives['mean_relevance'],
           s=200, alpha=0.5, color='lightblue', 
           edgecolor='black', linewidth=1.5,
           label='All solutions', zorder=2)

# Highlight Pareto front
ax1.scatter(pareto_front['mean_unexpectedness'],
           pareto_front['mean_relevance'],
           s=300, alpha=0.8, color='steelblue',
           edgecolor='black', linewidth=2,
           label='Pareto front', zorder=3)

# Draw Pareto front line
ax1.plot(pareto_front['mean_unexpectedness'],
        pareto_front['mean_relevance'],
        'r--', linewidth=2, alpha=0.5, zorder=1)

# Highlight knee point (using distance method as primary)
ax1.scatter(knee_distance['mean_unexpectedness'],
           knee_distance['mean_relevance'],
           s=600, marker='*', color='red',
           edgecolor='black', linewidth=3,
           label=f'Knee point (α={knee_distance["alpha"]:.2f})',
           zorder=10)

# Annotate all alpha values
for _, row in df_objectives.iterrows():
    ax1.annotate(f'α={row["alpha"]:.2f}',
                (row['mean_unexpectedness'], row['mean_relevance']),
                fontsize=9, ha='center', va='bottom',
                xytext=(0, 5), textcoords='offset points')

ax1.set_xlabel('Mean Unexpectedness (Distance)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Mean Relevance (CF Prediction)', fontsize=13, fontweight='bold')
ax1.set_title('Pareto Front: Unexpectedness vs Relevance', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='best')
ax1.grid(True, alpha=0.3)

# --- Panel 2: Serendipity Outcome ---

ax2.plot(df_objectives['alpha'], 
        df_objectives['serendipity'],
        'o-', linewidth=3, markersize=10, color='steelblue',
        label='Serendipity')

# Highlight knee point
ax2.scatter(knee_distance['alpha'],
           knee_distance['serendipity'],
           s=600, marker='*', color='red',
           edgecolor='black', linewidth=3,
           label=f'Knee point (α={knee_distance["alpha"]:.2f})',
           zorder=10)

ax2.axhline(1.0, color='green', linestyle='--', linewidth=2, 
           alpha=0.5, label='Perfect serendipity')
ax2.axvline(knee_distance['alpha'], color='red', linestyle=':', 
           linewidth=2, alpha=0.5)

ax2.set_xlabel('α (Weight on Unexpectedness)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Serendipity', fontsize=13, fontweight='bold')
ax2.set_title('Serendipity at Knee Point', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('pareto_front_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig('pareto_front_analysis.png', dpi=300, bbox_inches='tight')

print("\n✓ Saved: pareto_front_analysis.pdf/png")

# Step 5: Generate report
"""
Step 5: Generate Summary Report
================================
"""

print("\n" + "="*80)
print("PARETO FRONT SUMMARY REPORT")
print("="*80)

# Number of Pareto-optimal solutions
print(f"\nTotal solutions tested: {len(df_objectives)}")
print(f"Pareto-optimal solutions: {len(pareto_front)}")
print(f"Dominated solutions: {len(df_objectives) - len(pareto_front)}")

# Pareto front range
print(f"\nPareto Front Coverage:")
print(f"  Unexpectedness range: [{pareto_front['mean_unexpectedness'].min():.3f}, "
      f"{pareto_front['mean_unexpectedness'].max():.3f}]")
print(f"  Relevance range: [{pareto_front['mean_relevance'].min():.3f}, "
      f"{pareto_front['mean_relevance'].max():.3f}]")

# Knee point details
print(f"\nKnee Point (Distance Method):")
print(f"  α = {knee_distance['alpha']:.2f}")
print(f"  Weight on relevance: {(1-knee_distance['alpha'])*100:.0f}%")
print(f"  Weight on unexpectedness: {knee_distance['alpha']*100:.0f}%")
print(f"  Mean unexpectedness: {knee_distance['mean_unexpectedness']:.3f}")
print(f"  Mean relevance: {knee_distance['mean_relevance']:.3f}")
print(f"  Serendipity outcome: {knee_distance['serendipity']:.3f}")

# Compare extremes vs knee point
pure_cf = df_objectives[df_objectives['alpha'] == 0.0].iloc[0]
pure_dist = df_objectives[df_objectives['alpha'] == 1.0].iloc[0]

print(f"\nComparison:")
print(f"  Pure CF (α=0.0):")
print(f"    Unexpectedness: {pure_cf['mean_unexpectedness']:.3f}")
print(f"    Relevance: {pure_cf['mean_relevance']:.3f}")
print(f"    Serendipity: {pure_cf['serendipity']:.3f}")
print(f"  ")
print(f"  Knee Point (α={knee_distance['alpha']:.2f}):")
print(f"    Unexpectedness: {knee_distance['mean_unexpectedness']:.3f} "
      f"(+{knee_distance['mean_unexpectedness'] - pure_cf['mean_unexpectedness']:.3f})")
print(f"    Relevance: {knee_distance['mean_relevance']:.3f} "
      f"({knee_distance['mean_relevance'] - pure_cf['mean_relevance']:.3f})")
print(f"    Serendipity: {knee_distance['serendipity']:.3f} "
      f"(+{knee_distance['serendipity'] - pure_cf['serendipity']:.3f})")
print(f"  ")
print(f"  Pure Distance (α=1.0):")
print(f"    Unexpectedness: {pure_dist['mean_unexpectedness']:.3f}")
print(f"    Relevance: {pure_dist['mean_relevance']:.3f}")
print(f"    Serendipity: {pure_dist['serendipity']:.3f}")

# Trade-off at knee point
unexp_gain_pct = ((knee_distance['mean_unexpectedness'] - pure_cf['mean_unexpectedness']) / 
                   pure_cf['mean_unexpectedness'] * 100)
rel_loss_pct = ((pure_cf['mean_relevance'] - knee_distance['mean_relevance']) / 
                 pure_cf['mean_relevance'] * 100)

print(f"\nTrade-off at Knee Point:")
print(f"  Unexpectedness gain: +{unexp_gain_pct:.1f}%")
print(f"  Relevance sacrifice: -{rel_loss_pct:.1f}%")
print(f"  Trade-off ratio: {unexp_gain_pct/rel_loss_pct if rel_loss_pct > 0 else 0:.2f}:1")
print(f"  → {unexp_gain_pct:.1f}% unexpectedness gain for {rel_loss_pct:.1f}% relevance cost")

print("\n" + "="*80)

print("\nAnalysis complete!")
print("Files generated:")
print("  - pareto_objectives.csv")
print("  - pareto_front.csv")
print("  - pareto_front_analysis.pdf")
print("  - pareto_front_analysis.png")