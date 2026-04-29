import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
import ast
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from surprise import SVD
from surprise import Dataset, Reader
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set style for better plots
plt.style.use('ggplot')
sns.set_palette("husl")

print("=" * 80)
print("EXPERIMENT 2: Multi-Objective Recommendation")
print("Weighted Scalarization: Balancing Unexpectedness and Relevance")
print("=" * 80)

# ============================================================================
# PART 1: LOAD DATA AND TRAIN CF MODEL
# ============================================================================

print("\n[1/10] Loading AMBAR dataset...")

ratings_df = pd.read_csv('AMBAR/ratings_info.csv')
tracks_df = pd.read_csv('AMBAR/tracks_info.csv')

print(f"  ✓ Ratings: {len(ratings_df):,} rows")
print(f"  ✓ Tracks: {len(tracks_df):,} rows")
print(f"  ✓ Users: {ratings_df['user_id'].nunique():,} unique")

# For faster development, use sample
USE_SAMPLE = False  # Set to False for full experiment
if USE_SAMPLE:
    print("\n  [DEBUG] Using sample for faster testing...")
    ratings_df = ratings_df.sample(n=200000, random_state=42)
    print(f"  ✓ Sampled to {len(ratings_df):,} ratings")

print("\n[2/10] Parsing track metadata...")

def parse_styles(style_string):
    """Parse pipe-separated styles"""
    if pd.isna(style_string):
        return []
    if isinstance(style_string, list):
        return style_string
    if isinstance(style_string, str):
        return [s.strip() for s in style_string.split('|')]
    return []

tracks_df['styles'] = tracks_df['styles'].apply(parse_styles)
tracks_df['category_styles'] = tracks_df['category_styles'].apply(parse_styles)

print(f"  ✓ Parsed {len(tracks_df)} tracks")

print("\n[3/10] Creating train/test split (80/20 per user)...")

train_data = []
test_data = []

for user_id in ratings_df['user_id'].unique():
    user_ratings = ratings_df[ratings_df['user_id'] == user_id].copy()
    
    # Sort by timestamp if available, otherwise random
    if 'timestamp' in user_ratings.columns:
        user_ratings = user_ratings.sort_values('timestamp')
    else:
        user_ratings = user_ratings.sample(frac=1, random_state=42)
    
    # 80/20 split
    n = len(user_ratings)
    n_train = int(0.8 * n)
    
    train = user_ratings.iloc[:n_train]
    test = user_ratings.iloc[n_train:]
    
    train_data.append(train)
    test_data.append(test)

train_df = pd.concat(train_data, ignore_index=True)
test_df = pd.concat(test_data, ignore_index=True)

print(f"  ✓ Train set: {len(train_df):,} ratings")
print(f"  ✓ Test set: {len(test_df):,} ratings")

print("\n[4/10] Training Collaborative Filtering model...")

# Train CF model
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train_df[['user_id', 'track_id', 'rating']], reader)

svd = SVD(n_factors=50, n_epochs=20, verbose=True)
trainset = data.build_full_trainset()
svd.fit(trainset)

print("  ✓ CF model trained successfully!")

print("\n[5/10] Sampling users for experiment...")

# Filter: users with at least 20 train ratings and 5 test ratings
train_counts = train_df.groupby('user_id').size()
test_counts = test_df.groupby('user_id').size()

eligible_users = train_counts[
    (train_counts >= 10) & 
    (test_counts >= 3)
].index

print(f"  ✓ Eligible users: {len(eligible_users):,}")

# Sample 100 users
np.random.seed(42)
sample_users = np.random.choice(eligible_users, size=min(100, len(eligible_users)), replace=False)

print(f"  ✓ Sampled {len(sample_users)} users for experiment")

# Filter data to sample users
train_sample = train_df[train_df['user_id'].isin(sample_users)]
test_sample = test_df[test_df['user_id'].isin(sample_users)]

print(f"  ✓ Sample train ratings: {len(train_sample):,}")
print(f"  ✓ Sample test ratings: {len(test_sample):,}")

# ============================================================================
# PART 2: BUILD EXPECTED SETS
# ============================================================================

print("\n[6/10] Building Expected Sets (E_u) for each user...")

def build_E_u_minimal_categories(user_id, train_df, tracks_df):
    """Build minimal E_u using split category styles"""
    user_train = train_df[train_df['user_id'] == user_id]
    rated_track_ids = set(user_train['track_id'].values)
    
    rated_tracks_info = tracks_df[tracks_df['track_id'].isin(rated_track_ids)]
    
    # Collect SPLIT category styles
    all_categories = []
    for cat_list in rated_tracks_info['category_styles']:
        if isinstance(cat_list, list):
            for compound_cat in cat_list:
                # Split 'Rock|Pop' → ['Rock', 'Pop']
                if '|' in compound_cat:
                    all_categories.extend([c.strip() for c in compound_cat.split('|')])
                else:
                    all_categories.append(compound_cat)
    
    E_u_categories = set(all_categories)
    return rated_track_ids, E_u_categories

# Build E_u for all sample users
user_E_u = {}

for user_id in sample_users:
    E_u_ids, E_u_categories = build_E_u_minimal_categories(user_id, train_sample, tracks_df)
    user_E_u[user_id] = {
        'E_u_ids': E_u_ids,
        'E_u_categories': E_u_categories
    }

E_u_sizes = [len(data['E_u_ids']) for data in user_E_u.values()]
print(f"  ✓ Built E_u for {len(user_E_u)} users")
print(f"  ✓ E_u size: mean={np.mean(E_u_sizes):.0f}, median={np.median(E_u_sizes):.0f}")

# ============================================================================
# PART 3: IMPLEMENT WEIGHTED MULTI-OBJECTIVE RECOMMENDATION
# ============================================================================

print("\n[7/10] Implementing weighted multi-objective recommendation...")

def jaccard_distance(set_A, set_B):
    """Compute Jaccard distance between two sets"""
    if not set_A or not set_B:
        return 1.0
    
    intersection = len(set_A & set_B)
    union = len(set_A | set_B)
    
    if union == 0:
        return 1.0
    
    similarity = intersection / union
    return 1.0 - similarity

def compute_track_distance(track_id, E_u_categories, tracks_df):
    """Compute distance from track to E_u using category styles"""
    track_info = tracks_df[tracks_df['track_id'] == track_id]
    if len(track_info) == 0:
        return 1.0
    
    track_categories = track_info.iloc[0]['category_styles']
    
    # Split compound categories
    track_cat_set = set()
    if isinstance(track_categories, list):
        for compound_cat in track_categories:
            if '|' in compound_cat:
                track_cat_set.update([c.strip() for c in compound_cat.split('|')])
            else:
                track_cat_set.add(compound_cat)
    
    if not track_cat_set:
        return 1.0
    
    return jaccard_distance(track_cat_set, E_u_categories)

def weighted_recommendation(user_id, candidate_tracks, E_u_categories, svd_model, alpha=0.5, n=10):
    """
    Generate recommendations using weighted scalarization
    
    Score = alpha × distance + (1-alpha) × cf_score_normalized
    
    alpha=0   → Pure CF (maximize relevance only)
    alpha=0.5 → Balanced
    alpha=1   → Pure distance (maximize unexpectedness only)
    """
    scores = []
    
    for track_id in candidate_tracks:
        # Objective 1: Distance from E_u (unexpectedness)
        distance = compute_track_distance(track_id, E_u_categories, tracks_df)
        
        # Objective 2: CF predicted rating (relevance)
        cf_pred = svd_model.predict(user_id, track_id).est
        cf_score_normalized = (cf_pred - 1) / 4  # Normalize to [0,1]
        
        # Weighted combination
        combined_score = alpha * distance + (1 - alpha) * cf_score_normalized
        
        scores.append({
            'track_id': track_id,
            'combined_score': combined_score,
            'distance': distance,
            'cf_score': cf_pred
        })
    
    # Sort by combined score and return top-N
    scores.sort(key=lambda x: x['combined_score'], reverse=True)
    return scores[:n]

print("  ✓ Weighted recommendation function implemented")

# ============================================================================
# PART 4: GENERATE RECOMMENDATIONS FOR DIFFERENT ALPHA VALUES
# ============================================================================

print("\n[8/10] Generating recommendations with different alpha values...")

# Alpha values to test
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
alpha_names = {
    0.0: 'Pure CF (α=0.0)',
    0.25: 'CF-Biased (α=0.25)',
    0.5: 'Balanced (α=0.5)',
    0.75: 'Distance-Biased (α=0.75)',
    1.0: 'Pure Distance (α=1.0)'
}

# Store recommendations for each method
recommendations = {alpha: {} for alpha in alphas}

print("  Processing users...")
for idx, user_id in enumerate(sample_users):
    if (idx + 1) % 20 == 0:
        print(f"    {idx + 1}/{len(sample_users)} users processed...")
    
    # Get user's E_u
    E_u_categories = user_E_u[user_id]['E_u_categories']
    E_u_ids = user_E_u[user_id]['E_u_ids']
    
    # Get candidate tracks (unrated in training)
    all_tracks = set(tracks_df['track_id'])
    candidates = list(all_tracks - E_u_ids)
    
    # Limit candidates for speed (otherwise 400k+ tracks per user!)
    if len(candidates) > 5000:
        candidates = np.random.choice(candidates, size=5000, replace=False)
    
    # Generate recommendations for each alpha
    for alpha in alphas:
        recs = weighted_recommendation(user_id, candidates, E_u_categories, svd, alpha=alpha, n=10)
        recommendations[alpha][user_id] = recs

print(f"  ✓ Generated recommendations for {len(sample_users)} users × {len(alphas)} methods")

#debug

if len(recommendations[0.0]) > 0:
    test_user = list(recommendations[0.0].keys())[0]
    
    print("\n=== DIVERSITY DEBUG ===")
    print(f"Testing user: {test_user}")
    
    # Pure CF recommendations
    cf_recs = recommendations[0.0][test_user]
    print(f"\nPure CF (α=0.0) - First 3 tracks:")
    for i in range(min(3, len(cf_recs))):
        track_id = cf_recs[i]['track_id']
        track_info = tracks_df[tracks_df['track_id'] == track_id].iloc[0]
        cats = track_info['category_styles']
        print(f"  Track {track_id}: {cats}")
    
    # Pure Distance recommendations
    dist_recs = recommendations[1.0][test_user]
    print(f"\nPure Distance (α=1.0) - First 3 tracks:")
    for i in range(min(3, len(dist_recs))):
        track_id = dist_recs[i]['track_id']
        track_info = tracks_df[tracks_df['track_id'] == track_id].iloc[0]
        cats = track_info['category_styles']
        print(f"  Track {track_id}: {cats}")
    
    print("======================\n")

# ============================================================================
# PART 5: EVALUATION METRICS
# ============================================================================

print("\n[9/10] Evaluating recommendations...")

def evaluate_accuracy(recommendations_dict, test_df):
    """Evaluate accuracy using RMSE and MAE on test set"""
    predictions = []
    actuals = []
    
    for user_id, recs in recommendations_dict.items():
        # Get user's test ratings
        user_test = test_df[test_df['user_id'] == user_id]
        
        for rec in recs:
            track_id = rec['track_id']
            
            # Check if this track is in test set
            test_rating = user_test[user_test['track_id'] == track_id]
            if len(test_rating) > 0:
                predictions.append(rec['cf_score'])
                actuals.append(test_rating.iloc[0]['rating'])
    
    if len(predictions) == 0:
        return {'RMSE': np.nan, 'MAE': np.nan, 'n_hits': 0}
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    
    return {'RMSE': rmse, 'MAE': mae, 'n_hits': len(predictions)}

def evaluate_accuracy_all_test(recommendations_dict, test_df, svd_model):
    """Evaluate CF model accuracy on all test items (not just recommended)"""
    predictions = []
    actuals = []
    
    for user_id in recommendations_dict.keys():
        user_test = test_df[test_df['user_id'] == user_id]
        
        for _, row in user_test.iterrows():
            track_id = row['track_id']
            actual_rating = row['rating']
            
            # Predict using CF model
            pred = svd_model.predict(user_id, track_id).est
            
            predictions.append(pred)
            actuals.append(actual_rating)
    
    if len(predictions) == 0:
        return {'RMSE': np.nan, 'MAE': np.nan}
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    
    return {'RMSE': rmse, 'MAE': mae}

def evaluate_diversity(recommendations_dict):
    """Compute average intra-list distance (ILD) - diversity BETWEEN tracks"""
    ilds = []
    
    for user_id, recs in recommendations_dict.items():
        if len(recs) < 2:
            continue
        
        # Get category sets for all recommended tracks
        track_categories = []
        for rec in recs:
            track_id = rec['track_id']
            track_info = tracks_df[tracks_df['track_id'] == track_id]
            
            if len(track_info) == 0:
                continue
            
            # Parse categories
            cats = track_info.iloc[0]['category_styles']
            cat_set = set()
            if isinstance(cats, list):
                for compound_cat in cats:
                    if '|' in compound_cat:
                        cat_set.update([c.strip() for c in compound_cat.split('|')])
                    else:
                        cat_set.add(compound_cat)
            
            if cat_set:
                track_categories.append(cat_set)
        
        # Compute pairwise Jaccard distances between tracks
        if len(track_categories) < 2:
            continue
        
        pairwise_distances = []
        for i in range(len(track_categories)):
            for j in range(i+1, len(track_categories)):
                dist = jaccard_distance(track_categories[i], track_categories[j])
                pairwise_distances.append(dist)
        
        if pairwise_distances:
            ilds.append(np.mean(pairwise_distances))
    
    # Move debug OUTSIDE loop (only print once)
    # if len(ilds) > 0:
    #     print(f"[DEBUG] Average ILD across {len(ilds)} users: {np.mean(ilds):.3f}")
    
    return np.mean(ilds) if ilds else 0.0


def evaluate_serendipity(recommendations_dict, user_E_u):
    """
    Serendipity score based on Ge et al. (2010)
    
    Serendipitous = Unexpected AND Relevant
    - Unexpected: distance > 0.5
    - Relevant: CF score > 3.0
    """
    serendipity_scores = []
    
    for user_id, recs in recommendations_dict.items():
        serendipitous = 0
        
        for rec in recs:
            # Check if unexpected (distant from E_u)
            is_unexpected = rec['distance'] > 0.7
            
            # Check if relevant (high CF score)
            is_relevant = rec['cf_score'] > 1.8
            
            if is_unexpected and is_relevant:
                serendipitous += 1
        
        serendipity_scores.append(serendipitous / len(recs) if recs else 0)
    
    return np.mean(serendipity_scores)

def evaluate_coverage(recommendations_dict):
    """Catalog coverage: % of unique items recommended"""
    unique_items = set()
    
    for user_id, recs in recommendations_dict.items():
        for rec in recs:
            unique_items.add(rec['track_id'])
    
    return len(unique_items)

def evaluate_avg_distance(recommendations_dict):
    """Average distance from E_u across all recommendations"""
    distances = []
    
    for user_id, recs in recommendations_dict.items():
        for rec in recs:
            distances.append(rec['distance'])
    
    return np.mean(distances) if distances else 0.0

def evaluate_avg_cf_score(recommendations_dict):
    """Average CF predicted rating across all recommendations"""
    scores = []
    
    for user_id, recs in recommendations_dict.items():
        for rec in recs:
            scores.append(rec['cf_score'])
    
    return np.mean(scores) if scores else 0.0

# Evaluate all methods
results = {}

for alpha in alphas:
    print(f"  Evaluating {alpha_names[alpha]}...")
    
    results[alpha] = {
        'accuracy': evaluate_accuracy_all_test(recommendations[alpha], test_sample, svd),
        'diversity': evaluate_diversity(recommendations[alpha]),
        'serendipity': evaluate_serendipity(recommendations[alpha], user_E_u),
        'coverage': evaluate_coverage(recommendations[alpha]),
        #'avg_distance': evaluate_avg_distance(recommendations[alpha]),
        #'avg_cf_score': evaluate_avg_cf_score(recommendations[alpha])
    }

print("  ✓ Evaluation complete!")

# ============================================================================
# PART 6: DISPLAY RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 2 RESULTS")
print("=" * 80)

# Create results table
results_table = []

for alpha in alphas:
    r = results[alpha]
    results_table.append({
        'Method': alpha_names[alpha],
        'Alpha': alpha,
        'RMSE': r['accuracy']['RMSE'],
        'MAE': r['accuracy']['MAE'],
        'Diversity (ILD)': r['diversity'],
        'Serendipity': r['serendipity'],
        'Coverage': r['coverage'],
        #'Avg Distance': r['avg_distance'],
        #'Avg CF Score': r['avg_cf_score']
    })

results_df = pd.DataFrame(results_table)

print("\n📊 COMPARISON TABLE:")
print(results_df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

# Find best method per metric
print("\n🏆 BEST PERFORMERS:")

# Only show metrics that have valid values
if not results_df['RMSE'].isna().all():
    print(f"  Lowest RMSE:        {results_df.loc[results_df['RMSE'].idxmin(), 'Method']}")

print(f"  Highest Diversity:  {results_df.loc[results_df['Diversity (ILD)'].idxmax(), 'Method']}")
print(f"  Highest Serendipity: {results_df.loc[results_df['Serendipity'].idxmax(), 'Method']}")
print(f"  Highest Coverage:   {results_df.loc[results_df['Coverage'].idxmax(), 'Method']}")
# ========================================================
# PART 7: VISUALIZATIONS
# ============================================================================

print("\n[10/10] Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Experiment 2: Multi-Objective Trade-off Analysis', 
             fontsize=16, fontweight='bold', y=0.995)

# Plot 1: RMSE by Alpha
ax1 = axes[0, 0]
ax1.plot(results_df['Alpha'], results_df['RMSE'], 'o-', linewidth=2, markersize=8, color='#e74c3c')
ax1.set_xlabel('Alpha (Weight on Distance)', fontsize=11, fontweight='bold')
ax1.set_ylabel('RMSE', fontsize=11, fontweight='bold')
ax1.set_title('Accuracy vs Alpha', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)
ax1.set_xticks(alphas)

# Plot 2: Diversity by Alpha
ax2 = axes[0, 1]
ax2.plot(results_df['Alpha'], results_df['Diversity (ILD)'], 'o-', linewidth=2, markersize=8, color='#3498db')
ax2.set_xlabel('Alpha (Weight on Distance)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Intra-List Distance', fontsize=11, fontweight='bold')
ax2.set_title('Diversity vs Alpha', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)
ax2.set_xticks(alphas)

# Plot 3: Serendipity by Alpha
ax3 = axes[0, 2]
ax3.plot(results_df['Alpha'], results_df['Serendipity'], 'o-', linewidth=2, markersize=8, color='#2ecc71')
ax3.set_xlabel('Alpha (Weight on Distance)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Serendipity Score', fontsize=11, fontweight='bold')
ax3.set_title('Serendipity vs Alpha', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)
ax3.set_xticks(alphas)

# Plot 4: Average Distance by Alpha
ax4 = axes[1, 0]
ax4.plot(results_df['Alpha'], results_df['Avg Distance'], 'o-', linewidth=2, markersize=8, color='#9b59b6')
ax4.set_xlabel('Alpha (Weight on Distance)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Average Distance from E_u', fontsize=11, fontweight='bold')
ax4.set_title('Unexpectedness vs Alpha', fontsize=12, fontweight='bold')
ax4.grid(alpha=0.3)
ax4.set_xticks(alphas)

# Plot 5: Average CF Score by Alpha
ax5 = axes[1, 1]
ax5.plot(results_df['Alpha'], results_df['Avg CF Score'], 'o-', linewidth=2, markersize=8, color='#e67e22')
ax5.set_xlabel('Alpha (Weight on Distance)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Average Predicted Rating', fontsize=11, fontweight='bold')
ax5.set_title('Relevance vs Alpha', fontsize=12, fontweight='bold')
ax5.grid(alpha=0.3)
ax5.set_xticks(alphas)

# Plot 6: Coverage by Alpha
ax6 = axes[1, 2]
ax6.bar(results_df['Alpha'], results_df['Coverage'], color='#1abc9c', alpha=0.7, edgecolor='black')
ax6.set_xlabel('Alpha (Weight on Distance)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Unique Items Recommended', fontsize=11, fontweight='bold')
ax6.set_title('Catalog Coverage vs Alpha', fontsize=12, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)
ax6.set_xticks(alphas)

plt.tight_layout()
plt.savefig('experiment2_multiobjective_analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: experiment2_multiobjective_analysis.png")

# Create Pareto frontier plot
fig2, ax = plt.subplots(figsize=(10, 8))

# Plot each alpha as a point on distance vs CF score space
colors = ['#e74c3c', '#e67e22', '#f39c12', '#3498db', '#9b59b6']
for i, alpha in enumerate(alphas):
    r = results[alpha]
    ax.scatter(r['avg_distance'], r['avg_cf_score'], 
              s=300, c=colors[i], alpha=0.7, edgecolors='black', linewidth=2,
              label=alpha_names[alpha], zorder=10)
    
    # Add alpha label
    ax.annotate(f'α={alpha}', 
               xy=(r['avg_distance'], r['avg_cf_score']),
               xytext=(10, 10), textcoords='offset points',
               fontsize=10, fontweight='bold')

# Connect points to show trade-off curve
ax.plot(results_df['Avg Distance'], results_df['Avg CF Score'], 
       '--', linewidth=2, color='gray', alpha=0.5, zorder=5)

ax.set_xlabel('Unexpectedness (Distance from E_u)', fontsize=13, fontweight='bold')
ax.set_ylabel('Relevance (CF Predicted Rating)', fontsize=13, fontweight='bold')
ax.set_title('Trade-off Frontier: Unexpectedness vs Relevance', 
            fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('experiment2_pareto_frontier.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: experiment2_pareto_frontier.png")

plt.show()

# ============================================================================
# PART 8: SAVE RESULTS
# ============================================================================

print("\n[SAVING] Exporting results...")

# Save results table
results_df.to_csv('experiment2_results.csv', index=False)
print("  ✓ Saved: experiment2_results.csv")

# Save summary
with open('experiment2_summary.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("EXPERIMENT 2: MULTI-OBJECTIVE RECOMMENDATION\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Sample size: {len(sample_users)} users\n")
    f.write(f"Recommendations per user: 10\n")
    f.write(f"Alpha values tested: {alphas}\n\n")
    f.write("RESULTS:\n")
    f.write(results_df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    f.write("\n\n")
    f.write("KEY FINDINGS:\n")
    f.write(f"- Best accuracy (lowest RMSE): {results_df.loc[results_df['RMSE'].idxmin(), 'Method']}\n")
    f.write(f"- Best diversity: {results_df.loc[results_df['Diversity (ILD)'].idxmax(), 'Method']}\n")
    f.write(f"- Best serendipity: {results_df.loc[results_df['Serendipity'].idxmax(), 'Method']}\n")
    f.write(f"- Best coverage: {results_df.loc[results_df['Coverage'].idxmax(), 'Method']}\n")

print("  ✓ Saved: experiment2_summary.txt")

print("\n" + "=" * 80)
print("✅ EXPERIMENT 2 COMPLETE!")
print("=" * 80)
print("\n💡 KEY INSIGHT:")
if results[0.5]['serendipity'] > results[0.0]['serendipity']:
    print("  Balanced approach (α=0.5) achieves HIGHER serendipity than pure CF!")
    print("  This validates the multi-objective approach.")
else:
    print("  Results show trade-offs between unexpectedness and relevance.")
print("\n📁 Outputs:")
print("  - experiment2_results.csv")
print("  - experiment2_summary.txt")
print("  - experiment2_multiobjective_analysis.png")
print("  - experiment2_pareto_frontier.png")

# ============================================================================
# SAVE FOR SENSITIVITY ANALYSIS
# ============================================================================

import pickle

print("\n[SAVING] Exporting recommendations for sensitivity analysis...")

pickle.dump(recommendations, open('recommendations.pkl', 'wb'))
pickle.dump(user_E_u, open('user_E_u.pkl', 'wb'))

print("  ✓ Saved: recommendations.pkl")
print("  ✓ Saved: user_E_u.pkl")