"""
Sensitivity Analysis Heatmap Visualization
==========================================

Creates a publication-quality heatmap showing serendipity scores across
different threshold combinations and alpha values.

This visualization shows:
1. Color gradient representing serendipity magnitude (green=high, red=low)
2. Stars marking the winning alpha for each threshold combination
3. Clear visual pattern: α=0.25 dominates the "reasonable" middle region

Output: sensitivity_heatmap.png (for thesis)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# DATA
# ============================================================================

# Sensitivity analysis results
data = {
    'Thresholds': [
        '(0.5, 2.5)',
        '(0.6, 2.0)', 
        '(0.65, 1.9)',
        '(0.68, 1.85)',
        '(0.7, 1.8)',
        '(0.72, 1.75)',
        '(0.75, 1.7)',
        '(0.7, 1.6)',
        '(0.7, 2.0)',
        '(0.8, 1.5)',
        '(0.9, 1.2)',
    ],
    'Description': [
        'Low dist., high CF',
        'Medium-low',
        'Stricter than Ge',
        'Very close to Ge',
        'Ge et al. (original)',
        'Slightly different',
        'More lenient',
        'Lower CF req.',
        'Higher CF req.',
        'High dist., low CF',
        'Extreme distance',
    ],
    'α=0.0': [0.214, 0.693, 0.699, 0.682, 0.645, 0.628, 0.572, 0.645, 0.610, 0.529, 0.361],
    'α=0.25': [0.200, 0.846, 0.957, 0.976, 0.985, 0.995, 0.993, 0.998, 0.844, 0.988, 0.879],
    'α=0.5': [0.161, 0.761, 0.895, 0.934, 0.963, 0.987, 0.991, 1.000, 0.761, 1.000, 0.997],
    'α=0.75': [0.133, 0.706, 0.858, 0.906, 0.947, 0.975, 0.985, 1.000, 0.706, 1.000, 1.000],
    'α=1.0': [0.001, 0.082, 0.119, 0.150, 0.204, 0.241, 0.307, 0.454, 0.082, 0.591, 0.938],
}

df = pd.DataFrame(data)

# Identify winners for each threshold
winners = []
for i in range(len(df)):
    row = df.iloc[i]
    alpha_cols = ['α=0.0', 'α=0.25', 'α=0.5', 'α=0.75', 'α=1.0']
    max_val = row[alpha_cols].max()
    winner_alphas = [col for col in alpha_cols if row[col] == max_val]
    winners.append(winner_alphas)

# ============================================================================
# CREATE HEATMAP
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 8))

# Prepare data matrix for heatmap
alpha_cols = ['α=0.0', 'α=0.25', 'α=0.5', 'α=0.75', 'α=1.0']
heatmap_data = df[alpha_cols].values

# Create heatmap with diverging colormap
# Use RdYlGn (Red-Yellow-Green) where Green = high serendipity
sns.heatmap(
    heatmap_data,
    annot=False,  # We'll add custom annotations
    fmt='.3f',
    cmap='RdYlGn',
    vmin=0.0,
    vmax=1.0,
    cbar_kws={'label': 'Serendipity Score'},
    linewidths=0.5,
    linecolor='gray',
    ax=ax
)

# Add custom annotations with values and stars
for i in range(len(df)):
    for j, col in enumerate(alpha_cols):
        value = df.iloc[i][col]
        
        # Check if this is a winner
        is_winner = col in winners[i]
        
        # Format text
        if is_winner:
            text = f'{value:.2f}\n★'
            weight = 'bold'
            fontsize = 11
        else:
            text = f'{value:.2f}'
            weight = 'normal'
            fontsize = 10
        
        # Choose text color based on background
        # Dark text for light backgrounds, light text for dark backgrounds
        if value > 0.5:
            color = 'black'
        else:
            color = 'white'
        
        ax.text(j + 0.5, i + 0.5, text,
                ha='center', va='center',
                color=color, fontsize=fontsize, weight=weight)

# Set labels
ax.set_xticklabels(alpha_cols, rotation=0, ha='center')
ax.set_yticklabels(df['Description'], rotation=0, ha='right')

ax.set_xlabel('Weighting Strategy', fontsize=12, fontweight='bold')
ax.set_ylabel('Threshold Definition', fontsize=12, fontweight='bold')
ax.set_title('Sensitivity Analysis: Serendipity Across Threshold Variations\n(★ = Winner)', 
             fontsize=14, fontweight='bold', pad=20)

# Add boxes around "reasonable range" (rows 1-8)
# These are the thresholds close to Ge et al.
reasonable_start = 1
reasonable_end = 9

rect = plt.Rectangle(
    (-0.05, reasonable_start - 0.05),
    len(alpha_cols) + 0.1,
    reasonable_end - reasonable_start + 0.1,
    fill=False,
    edgecolor='blue',
    linewidth=2.5,
    linestyle='--'
)
ax.add_patch(rect)

# Add annotation for reasonable range
ax.text(
    len(alpha_cols) + 0.3, 
    (reasonable_start + reasonable_end) / 2,
    'Reasonable\nrange\n(±0.05 dist,\n±0.2 CF)',
    fontsize=9,
    va='center',
    ha='left',
    color='blue',
    fontweight='bold'
)

# Highlight Ge et al. row (row 4)
ge_row = 4
rect_ge = plt.Rectangle(
    (-0.05, ge_row - 0.05),
    len(alpha_cols) + 0.1,
    1.1,
    fill=False,
    edgecolor='darkgreen',
    linewidth=3,
    linestyle='-'
)
ax.add_patch(rect_ge)

plt.tight_layout()

# Save figure
output_file = 'sensitivity_heatmap.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f'✓ Saved: {output_file}')

# Also save as PDF for LaTeX
output_pdf = 'sensitivity_heatmap.pdf'
plt.savefig(output_pdf, bbox_inches='tight')
print(f'✓ Saved: {output_pdf}')

plt.close()

# ============================================================================
# CREATE SUMMARY STATISTICS TABLE
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

# Count wins per alpha
alpha_names = {
    'α=0.0': 'Pure CF',
    'α=0.25': 'CF-Biased',
    'α=0.5': 'Balanced',
    'α=0.75': 'Distance-Biased',
    'α=1.0': 'Pure Distance'
}

win_counts = {col: 0 for col in alpha_cols}

for winner_list in winners:
    for winner in winner_list:
        win_counts[winner] += 1

print("\nOverall Win Counts:")
for col in alpha_cols:
    count = win_counts[col]
    total = len(df)
    pct = 100 * count / total
    print(f"  {alpha_names[col]:20s}: {count}/{total} ({pct:.1f}%)")

# Count wins in reasonable range only
reasonable_winners = winners[reasonable_start:reasonable_end]
reasonable_win_counts = {col: 0 for col in alpha_cols}

for winner_list in reasonable_winners:
    for winner in winner_list:
        reasonable_win_counts[winner] += 1

print(f"\nReasonable Range Win Counts (rows {reasonable_start+1}-{reasonable_end}):")
for col in alpha_cols:
    count = reasonable_win_counts[col]
    total = len(reasonable_winners)
    pct = 100 * count / total
    print(f"  {alpha_names[col]:20s}: {count}/{total} ({pct:.1f}%)")

# Identify α=0.25 performance
alpha_025_overall = win_counts['α=0.25']
alpha_025_reasonable = reasonable_win_counts['α=0.25']

print("\n" + "=" * 80)
print("KEY FINDING")
print("=" * 80)
print(f"\nCF-Biased (α=0.25) Performance:")
print(f"  Overall:          {alpha_025_overall}/{len(df)} combinations ({100*alpha_025_overall/len(df):.0f}%)")
print(f"  Reasonable range: {alpha_025_reasonable}/{len(reasonable_winners)} combinations ({100*alpha_025_reasonable/len(reasonable_winners):.0f}%)")
print(f"\n✓ Demonstrates robust performance across balanced serendipity definitions!")

# ============================================================================
# CREATE SIMPLE BAR CHART OF WINS
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Overall wins
ax1.bar(range(len(alpha_cols)), [win_counts[col] for col in alpha_cols], 
        color=['gray', 'gold', 'skyblue', 'lightgreen', 'salmon'])
ax1.set_xticks(range(len(alpha_cols)))
ax1.set_xticklabels([alpha_names[col] for col in alpha_cols], rotation=45, ha='right')
ax1.set_ylabel('Number of Wins', fontweight='bold')
ax1.set_title('Overall: Wins Across All 11 Threshold Combinations', fontweight='bold')
ax1.set_ylim(0, max(win_counts.values()) + 1)

# Add value labels on bars
for i, col in enumerate(alpha_cols):
    count = win_counts[col]
    ax1.text(i, count + 0.1, f'{count}', ha='center', va='bottom', fontweight='bold')

# Reasonable range wins
ax2.bar(range(len(alpha_cols)), [reasonable_win_counts[col] for col in alpha_cols],
        color=['gray', 'gold', 'skyblue', 'lightgreen', 'salmon'])
ax2.set_xticks(range(len(alpha_cols)))
ax2.set_xticklabels([alpha_names[col] for col in alpha_cols], rotation=45, ha='right')
ax2.set_ylabel('Number of Wins', fontweight='bold')
ax2.set_title('Reasonable Range: Wins in Balanced Definitions\n(±0.05 distance, ±0.2 CF)', fontweight='bold')
ax2.set_ylim(0, len(reasonable_winners) + 1)

# Add value labels
for i, col in enumerate(alpha_cols):
    count = reasonable_win_counts[col]
    ax2.text(i, count + 0.1, f'{count}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()

output_bars = 'sensitivity_wins_comparison.png'
plt.savefig(output_bars, dpi=300, bbox_inches='tight')
print(f'\n✓ Saved: {output_bars}')

output_bars_pdf = 'sensitivity_wins_comparison.pdf'
plt.savefig(output_bars_pdf, bbox_inches='tight')
print(f'✓ Saved: {output_bars_pdf}')

plt.close()

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
print("\nGenerated files:")
print("  1. sensitivity_heatmap.png (main visualization for thesis)")
print("  2. sensitivity_heatmap.pdf (vector format for LaTeX)")
print("  3. sensitivity_wins_comparison.png (supplementary bar charts)")
print("  4. sensitivity_wins_comparison.pdf (vector format)")
print("\nRecommended for thesis: Use sensitivity_heatmap.pdf in your LaTeX document")
print()