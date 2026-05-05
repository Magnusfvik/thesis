"""
User Heterogeneity Analysis
============================

Investigates whether optimal α weighting varies across user types.

Two segmentation strategies:
1. Listening diversity (primary - theoretically grounded)
2. Pure CF serendipity (robustness check)

Outputs:
- Figures showing serendipity curves per segment
- Tables with optimal α and statistics per segment
- Statistical tests (ANOVA, t-tests, effect sizes)

Runtime: ~10 minutes
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_rel
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("USER HETEROGENEITY ANALYSIS")
print("=" * 80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("[1/6] Loading data...")

recs = pickle.load(open('recommendations_fair_complete.pkl', 'rb'))
user_E_u = pickle.load(open('user_E_u.pkl', 'rb'))

alpha_values = sorted(recs.keys())
all_users = list(recs[alpha_values[0]].keys())

print(f"  ✓ Loaded {len(alpha_values)} α values")
print(f"  ✓ Loaded {len(all_users)} users")

# Load metadata
import pandas as pd
tracks_df = pd.read_csv('AMBAR/tracks_info.csv')

def parse_styles(style_string):
    if pd.isna(style_string):
        return []
    if isinstance(style_string, list):
        return style_string
    if isinstance(style_string, str):
        return [s.strip() for s in style_string.split('|')]
    return []

tracks_df['category_styles'] = tracks_df['category_styles'].apply(parse_styles)
track_metadata = {}
for _, track in tracks_df.iterrows():
    track_metadata[track['track_id']] = set(track['category_styles'])

# ============================================================================
# COMPUTE SERENDIPITY FOR ALL (USER, ALPHA) PAIRS
# ============================================================================

print("\n[2/6] Computing serendipity matrix...")

DISTANCE_THRESHOLD = 0.7
CF_THRESHOLD = 1.8

def calculate_serendipity(rec_list):
    """Calculate serendipity for recommendation list"""
    if not rec_list:
        return 0.0
    n_ser = sum(1 for r in rec_list 
               if r['distance'] > DISTANCE_THRESHOLD 
               and r['cf_score'] > CF_THRESHOLD)
    return n_ser / len(rec_list)

# Build matrix: serendipity[alpha][user_id]
serendipity = {}
for alpha in alpha_values:
    serendipity[alpha] = {}
    for user_id in all_users:
        s = calculate_serendipity(recs[alpha][user_id])
        serendipity[alpha][user_id] = s

print(f"  ✓ Computed serendipity for {len(alpha_values)} × {len(all_users)} combinations")

# ============================================================================
# SEGMENTATION STRATEGY 1: LISTENING DIVERSITY
# ============================================================================

print("\n[3/6] Segmenting users by listening diversity...")

def jaccard_distance(set1, set2):
    if len(set1) == 0 or len(set2) == 0:
        return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 1 - (intersection / union) if union > 0 else 1.0

def user_diversity(E_u_tracks):
    """Calculate ILD of user's listening history"""
    if len(E_u_tracks) < 2:
        return 0.0
    
    E_u_list = list(E_u_tracks)
    distances = []
    
    for i in range(len(E_u_list)):
        for j in range(i+1, len(E_u_list)):
            if E_u_list[i] in track_metadata and E_u_list[j] in track_metadata:
                d = jaccard_distance(
                    track_metadata[E_u_list[i]], 
                    track_metadata[E_u_list[j]]
                )
                distances.append(d)
    
    return np.mean(distances) if distances else 0.0

# Calculate diversity for each user
user_diversities = {}
for user_id in all_users:
    E_u_data = user_E_u[user_id]
    E_u_tracks = E_u_data['E_u_ids'] if isinstance(E_u_data, dict) else E_u_data
    user_diversities[user_id] = user_diversity(E_u_tracks)

# Split on tertiles
diversity_values = sorted(user_diversities.values())
low_threshold = np.percentile(diversity_values, 33)
high_threshold = np.percentile(diversity_values, 67)

diversity_segments = {
    'Explorers (High Diversity)': [u for u, d in user_diversities.items() if d > high_threshold],
    'Balanced': [u for u, d in user_diversities.items() if low_threshold <= d <= high_threshold],
    'Exploiters (Low Diversity)': [u for u, d in user_diversities.items() if d < low_threshold]
}

print()
for seg_name, user_list in diversity_segments.items():
    avg_div = np.mean([user_diversities[u] for u in user_list])
    print(f"  {seg_name}: n={len(user_list)}, avg diversity={avg_div:.3f}")

# ============================================================================
# SEGMENTATION STRATEGY 2: PURE CF SERENDIPITY (ROBUSTNESS CHECK)
# ============================================================================

print("\n[4/6] Segmenting users by pure CF serendipity (robustness)...")

pure_cf_serendipity = serendipity[0.0]

cf_segments = {
    'High CF Serendipity': [u for u, s in pure_cf_serendipity.items() if s > 0.7],
    'Medium CF Serendipity': [u for u, s in pure_cf_serendipity.items() if 0.3 <= s <= 0.7],
    'Low CF Serendipity': [u for u, s in pure_cf_serendipity.items() if s < 0.3]
}

print()
for seg_name, user_list in cf_segments.items():
    avg_cf = np.mean([pure_cf_serendipity[u] for u in user_list])
    print(f"  {seg_name}: n={len(user_list)}, avg CF serendipity={avg_cf:.3f}")

# ============================================================================
# ANALYSIS: PLOT CURVES PER SEGMENT
# ============================================================================

print("\n[5/6] Generating visualizations...")

def analyze_segment(segments_dict, title_suffix):
    """Analyze and plot curves for a segmentation strategy"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    results_table = []
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for (seg_name, user_list), color in zip(segments_dict.items(), colors):
        # Calculate mean serendipity across α for this segment
        seg_serendipity = {}
        for alpha in alpha_values:
            scores = [serendipity[alpha][u] for u in user_list]
            seg_serendipity[alpha] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
        
        # Plot curve
        alphas = list(seg_serendipity.keys())
        means = [seg_serendipity[a]['mean'] for a in alphas]
        stds = [seg_serendipity[a]['std'] for a in alphas]
        
        ax1.plot(alphas, means, 'o-', label=seg_name, linewidth=2.5, 
                color=color, markersize=8)
        ax1.fill_between(alphas, 
                         np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds),
                         alpha=0.2, color=color)
        
        # Find optimal α for this segment
        optimal_alpha = max(seg_serendipity.keys(), 
                           key=lambda a: seg_serendipity[a]['mean'])
        optimal_mean = seg_serendipity[optimal_alpha]['mean']
        optimal_std = seg_serendipity[optimal_alpha]['std']
        
        # Mark optimal
        ax1.plot(optimal_alpha, optimal_mean, '*', 
                markersize=20, color=color, markeredgecolor='black', 
                markeredgewidth=1.5, zorder=10)
        
        # Statistical test: Optimal vs Pure CF within this segment
        optimal_scores = seg_serendipity[optimal_alpha]['scores']
        baseline_scores = seg_serendipity[0.0]['scores']
        
        t_stat, p_value = ttest_rel(optimal_scores, baseline_scores)
        mean_diff = np.mean(optimal_scores) - np.mean(baseline_scores)
        cohen_d = mean_diff / np.std(np.array(optimal_scores) - np.array(baseline_scores))
        
        results_table.append({
            'Segment': seg_name,
            'N': len(user_list),
            'Optimal_Alpha': optimal_alpha,
            'Optimal_Serendipity_Mean': optimal_mean,
            'Optimal_Serendipity_Std': optimal_std,
            'Baseline_Serendipity': np.mean(baseline_scores),
            'Improvement': mean_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'Cohens_d': cohen_d
        })
    
    # Format plot 1
    ax1.set_xlabel('α (Weight on Unexpectedness)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Serendipity', fontsize=12, fontweight='bold')
    ax1.set_title(f'Serendipity by User Type - {title_suffix}', 
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(0, 1.05)
    
    # Plot 2: Optimal α distribution
    optimal_alphas_per_segment = {}
    for seg_name, user_list in segments_dict.items():
        user_optimal = []
        for user_id in user_list:
            # Find this user's optimal α
            best_alpha = max(alpha_values, 
                           key=lambda a: serendipity[a][user_id])
            user_optimal.append(best_alpha)
        optimal_alphas_per_segment[seg_name] = user_optimal
    
    positions = list(range(len(segments_dict)))
    bp = ax2.boxplot([optimal_alphas_per_segment[seg] for seg in segments_dict.keys()],
                     positions=positions,
                     labels=[seg.split('(')[0].strip() for seg in segments_dict.keys()],
                     patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax2.set_ylabel('Optimal α (per user)', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of User-Specific Optimal α', 
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    return fig, pd.DataFrame(results_table), optimal_alphas_per_segment

# Analyze diversity segmentation
fig1, df1, opt_alpha_div = analyze_segment(diversity_segments, 
                                            "Listening Diversity Segmentation")
plt.savefig('user_heterogeneity_diversity.png', dpi=300, bbox_inches='tight')
plt.savefig('user_heterogeneity_diversity.pdf', bbox_inches='tight')
print("  ✓ Saved: user_heterogeneity_diversity.png/pdf")

# Analyze CF segmentation  
fig2, df2, opt_alpha_cf = analyze_segment(cf_segments, 
                                          "Pure CF Serendipity Segmentation")
plt.savefig('user_heterogeneity_cf.png', dpi=300, bbox_inches='tight')
plt.savefig('user_heterogeneity_cf.pdf', bbox_inches='tight')
print("  ✓ Saved: user_heterogeneity_cf.png/pdf")

# ============================================================================
# STATISTICAL TESTS
# ============================================================================

print("\n[6/6] Running statistical tests...")

# ANOVA: Does optimal α differ across diversity segments?
F_div, p_div = f_oneway(*[opt_alpha_div[seg] for seg in diversity_segments.keys()])
print(f"\n  Diversity Segmentation ANOVA:")
print(f"    F(2, {len(all_users)-3}) = {F_div:.3f}, p = {p_div:.4f}")

if p_div < 0.05:
    print(f"    → Optimal α DOES differ by listening diversity (p<0.05)")
else:
    print(f"    → Optimal α does NOT differ by listening diversity (p≥0.05)")

# ANOVA: Does optimal α differ across CF segments?
F_cf, p_cf = f_oneway(*[opt_alpha_cf[seg] for seg in cf_segments.keys()])
print(f"\n  CF Serendipity Segmentation ANOVA:")
print(f"    F(2, {len(all_users)-3}) = {F_cf:.3f}, p = {p_cf:.4f}")

if p_cf < 0.05:
    print(f"    → Optimal α DOES differ by CF serendipity (p<0.05)")
else:
    print(f"    → Optimal α does NOT differ by CF serendipity (p≥0.05)")

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Save tables
df1.to_csv('user_heterogeneity_diversity_results.csv', index=False, float_format='%.4f')
df2.to_csv('user_heterogeneity_cf_results.csv', index=False, float_format='%.4f')
print("\n  ✓ Saved: user_heterogeneity_diversity_results.csv")
print("  ✓ Saved: user_heterogeneity_cf_results.csv")

# Print summary tables
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print("\nDIVERSITY SEGMENTATION:")
print(df1.to_string(index=False))
print("\nCF SERENDIPITY SEGMENTATION:")
print(df2.to_string(index=False))

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)

