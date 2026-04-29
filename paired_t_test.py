"""
Statistical Significance Testing for Experiment 2
==================================================

Tests whether differences in serendipity scores between α values
are statistically significant using paired t-tests.

Requires: recommendations.pkl, user_E_u.pkl
"""

import numpy as np
import pandas as pd
import pickle
from scipy import stats
from itertools import combinations

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading pre-computed recommendations...")
recommendations = pickle.load(open('recommendations.pkl', 'rb'))
user_E_u = pickle.load(open('user_E_u.pkl', 'rb'))

alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
alpha_names = {
    0.0: 'Pure CF (α=0.0)',
    0.25: 'CF-Biased (α=0.25)',
    0.5: 'Balanced (α=0.5)',
    0.75: 'Distance-Biased (α=0.75)',
    1.0: 'Pure Distance (α=1.0)'
}

# ============================================================================
# CALCULATE PER-USER SERENDIPITY
# ============================================================================

def calculate_user_serendipity(user_recs, distance_threshold=0.7, cf_threshold=1.8):
    """Calculate serendipity for one user"""
    if not user_recs:
        return 0.0
    
    serendipitous = 0
    for rec in user_recs:
        if rec['distance'] > distance_threshold and rec['cf_score'] > cf_threshold:
            serendipitous += 1
    
    return serendipitous / len(user_recs)


print("\nCalculating per-user serendipity scores...")

# Store per-user scores
user_scores = {alpha: [] for alpha in alphas}

for user_id in user_E_u.keys():
    for alpha in alphas:
        user_recs = recommendations[alpha][user_id]
        score = calculate_user_serendipity(user_recs)
        user_scores[alpha].append(score)

# Convert to numpy arrays
for alpha in alphas:
    user_scores[alpha] = np.array(user_scores[alpha])
    print(f"  {alpha_names[alpha]:30s}: mean={np.mean(user_scores[alpha]):.3f}, "
          f"std={np.std(user_scores[alpha]):.3f}")

# ============================================================================
# PAIRED T-TESTS: Compare all pairs
# ============================================================================

print("\n" + "=" * 80)
print("PAIRED T-TEST RESULTS")
print("=" * 80)
print("\nComparing all pairs of α values (N=100 users, paired samples)\n")

results = []

for alpha1, alpha2 in combinations(alphas, 2):
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(user_scores[alpha1], user_scores[alpha2])
    
    # Effect size (Cohen's d for paired samples)
    diff = user_scores[alpha1] - user_scores[alpha2]
    cohens_d = np.mean(diff) / np.std(diff)
    
    # Mean difference
    mean_diff = np.mean(user_scores[alpha1]) - np.mean(user_scores[alpha2])
    
    results.append({
        'Alpha_1': alpha1,
        'Alpha_2': alpha2,
        'Mean_1': np.mean(user_scores[alpha1]),
        'Mean_2': np.mean(user_scores[alpha2]),
        'Mean_Diff': mean_diff,
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05
    })
    
    # Print result
    sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
    
    print(f"{alpha_names[alpha1]:30s} vs {alpha_names[alpha2]:30s}")
    print(f"  Mean difference: {mean_diff:+.4f}")
    print(f"  t-statistic: {t_stat:+.3f}")
    print(f"  p-value: {p_value:.4f} {sig_marker}")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print()

# ============================================================================
# SUMMARY TABLE
# ============================================================================

df = pd.DataFrame(results)

print("=" * 80)
print("SUMMARY: Statistical Significance Matrix")
print("=" * 80)
print()

# Create matrix showing which comparisons are significant
pivot_table = []
for a1 in alphas:
    row = []
    for a2 in alphas:
        if a1 == a2:
            row.append("-")
        elif a1 < a2:
            # Find the comparison
            comp = df[(df['Alpha_1'] == a1) & (df['Alpha_2'] == a2)]
            if len(comp) > 0:
                p = comp.iloc[0]['p_value']
                if p < 0.001:
                    row.append("***")
                elif p < 0.01:
                    row.append("**")
                elif p < 0.05:
                    row.append("*")
                else:
                    row.append("n.s.")
        else:
            row.append("")
    pivot_table.append(row)

# Print matrix
print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant\n")
print(f"{'':10s}", end="")
for a in alphas:
    print(f"α={a:<6.2f}", end="")
print()

for i, a1 in enumerate(alphas):
    print(f"α={a1:<6.2f}", end="")
    for j, a2 in enumerate(alphas):
        print(f"{pivot_table[i][j]:^8s}", end="")
    print()

# ============================================================================
# KEY FINDINGS FOR α=0.25
# ============================================================================

print("\n" + "=" * 80)
print("KEY FINDINGS: Is α=0.25 Significantly Better?")
print("=" * 80)
print()

alpha_025_comparisons = df[(df['Alpha_1'] == 0.25) | (df['Alpha_2'] == 0.25)]

print(f"CF-Biased (α=0.25) vs Other Methods:\n")

for _, row in alpha_025_comparisons.iterrows():
    if row['Alpha_1'] == 0.25:
        other_alpha = row['Alpha_2']
        diff = -row['Mean_Diff']  # Flip sign
    else:
        other_alpha = row['Alpha_1']
        diff = row['Mean_Diff']
    
    sig = "SIGNIFICANT" if row['significant'] else "NOT significant"
    
    print(f"  vs {alpha_names[other_alpha]:30s}: "
          f"Δ={diff:+.4f}, p={row['p_value']:.4f} ({sig})")

print()

# Count significant wins
sig_wins = sum(1 for _, row in alpha_025_comparisons.iterrows() if row['significant'])
total_comparisons = len(alpha_025_comparisons)

print(f"α=0.25 is significantly different from {sig_wins}/{total_comparisons} other methods")

# ============================================================================
# SAVE RESULTS
# ============================================================================

df.to_csv('statistical_significance_results.csv', index=False, float_format='%.4f')
print(f"\n✓ Saved: statistical_significance_results.csv")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

if sig_wins >= 3:
    print("✓✓✓ STRONG EVIDENCE")
    print(f"    α=0.25 is statistically significantly better than {sig_wins} out of 4 alternatives.")
    print("    The superiority of CF-Biased is not due to random chance.")
elif sig_wins >= 2:
    print("✓✓ MODERATE EVIDENCE")
    print(f"    α=0.25 is significantly better than {sig_wins} alternatives.")
    print("    Some differences may be due to chance.")
else:
    print("⚠ WEAK EVIDENCE")
    print(f"    α=0.25 is only significantly better than {sig_wins} alternative(s).")
    print("    Differences may be due to random variation.")

print()