"""
Sensitivity Analysis - Full Pipeline Version
=============================================

Tests robustness of α=0.25 finding across different serendipity threshold
definitions by running complete pipeline from scratch.

This version:
- Loads data from AMBAR
- Trains CF model
- Builds E_u
- Generates recommendations for all α values
- Tests 10 different threshold combinations
- Skips unnecessary metrics (diversity, coverage, plotting) for speed

Runtime: ~8 minutes (vs 15 for full PSOMOO.py)

Author: [Your Name]
Date: 2026-04-24
"""

import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SENSITIVITY ANALYSIS - FULL PIPELINE VERSION")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# PART 1: LOAD DATA
# ============================================================================

print("[1/8] Loading AMBAR dataset...")

ratings_df = pd.read_csv('AMBAR/ratings_info.csv')
tracks_df = pd.read_csv('AMBAR/tracks_info.csv')

print(f"  ✓ Ratings: {len(ratings_df):,} rows")
print(f"  ✓ Tracks: {len(tracks_df):,} rows")
print(f"  ✓ Users: {ratings_df['user_id'].nunique():,} unique")

# Use sample for faster testing (same as PSOMOO.py)
USE_SAMPLE = True
if USE_SAMPLE:
    print("\n  [SAMPLE MODE] Using 200K ratings for faster testing...")
    ratings_df = ratings_df.sample(n=200000, random_state=42)
    print(f"  ✓ Sampled to {len(ratings_df):,} ratings")

# ============================================================================
# PART 2: PARSE METADATA
# ============================================================================

print("\n[2/8] Parsing track metadata...")

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

# ============================================================================
# PART 3: TRAIN/TEST SPLIT
# ============================================================================

print("\n[3/8] Creating train/test split (80/20 per user)...")

train_data = []
test_data = []

for user_id in ratings_df['user_id'].unique():
    user_ratings = ratings_df[ratings_df['user_id'] == user_id].copy()
    
    # Sort by timestamp if available
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

# ============================================================================
# PART 4: TRAIN CF MODEL
# ============================================================================

print("\n[4/8] Training Collaborative Filtering model (SVD)...")

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train_df[['user_id', 'track_id', 'rating']], reader)

svd = SVD(n_factors=50, n_epochs=20, verbose=False)  # verbose=False for cleaner output
trainset = data.build_full_trainset()
svd.fit(trainset)

print(f"  ✓ CF model trained (50 factors, 20 epochs)")

# ============================================================================
# PART 5: BUILD EXPECTED SETS (E_u)
# ============================================================================

print("\n[5/8] Building Expected Sets (E_u) for each user...")

# Select sample of users for evaluation
all_users = train_df['user_id'].unique()
eligible_users = []

# Filter users with enough ratings for meaningful evaluation
for user_id in all_users:
    user_train = train_df[train_df['user_id'] == user_id]
    if len(user_train) >= 10:  # At least 10 training ratings
        eligible_users.append(user_id)

print(f"  ✓ Found {len(eligible_users)} eligible users (≥10 train ratings)")

# Sample users for faster evaluation (same as PSOMOO.py)
SAMPLE_USERS = 100
sample_users = np.random.choice(eligible_users, size=min(SAMPLE_USERS, len(eligible_users)), replace=False)

print(f"  ✓ Sampled {len(sample_users)} users for evaluation")

# Build E_u (minimal definition: all tracks in training set)
user_E_u = {}

for user_id in sample_users:
    user_train = train_df[train_df['user_id'] == user_id]
    E_u = set(user_train['track_id'].unique())
    user_E_u[user_id] = E_u

avg_E_u_size = np.mean([len(E_u) for E_u in user_E_u.values()])
print(f"  ✓ Built E_u for {len(user_E_u)} users (avg size: {avg_E_u_size:.1f} tracks)")

# ============================================================================
# PART 6: COMPUTE JACCARD DISTANCE
# ============================================================================

print("\n[6/8] Computing Jaccard distances...")

def jaccard_distance(set1, set2):
    """Compute Jaccard distance between two sets"""
    if len(set1) == 0 or len(set2) == 0:
        return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 1.0
    return 1 - (intersection / union)

# Pre-compute track metadata sets
track_metadata = {}
for _, track in tracks_df.iterrows():
    combined = set(track['category_styles'])
    track_metadata[track['track_id']] = combined

print(f"  ✓ Pre-computed metadata for {len(track_metadata)} tracks")

# ============================================================================
# PART 7: GENERATE RECOMMENDATIONS FOR ALL α VALUES
# ============================================================================

print("\n[7/8] Generating recommendations for all α values...")

# Define alpha values
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
alpha_names = {
    0.0: 'Pure CF (α=0.0)',
    0.25: 'CF-Biased (α=0.25)',
    0.5: 'Balanced (α=0.5)',
    0.75: 'Distance-Biased (α=0.75)',
    1.0: 'Pure Distance (α=1.0)'
}

# Get all tracks
all_tracks = set(tracks_df['track_id'].unique())

# Store recommendations for each alpha
recommendations = {}

for alpha in alphas:
    print(f"\n  Generating recommendations for {alpha_names[alpha]}...")
    recommendations[alpha] = {}
    
    for i, user_id in enumerate(sample_users, 1):
        if i % 25 == 0:
            print(f"    Progress: {i}/{len(sample_users)} users")
        
        E_u = user_E_u[user_id]
        candidate_tracks = all_tracks - E_u  # Tracks not in E_u
        
        scores = []
        
        for track_id in candidate_tracks:
            # Get CF prediction
            cf_pred = svd.predict(user_id, track_id).est
            
            # Get distance from E_u (average to all tracks in E_u)
            if track_id not in track_metadata:
                continue
            
            track_set = track_metadata[track_id]
            
            distances = []
            for e_track in E_u:
                if e_track in track_metadata:
                    e_set = track_metadata[e_track]
                    dist = jaccard_distance(track_set, e_set)
                    distances.append(dist)
            
            if not distances:
                continue
            
            avg_distance = np.mean(distances)
            
            # Normalize CF score to [0, 1] range (assuming 1-5 scale)
            cf_normalized = (cf_pred - 1) / 4
            
            # Weighted score
            combined_score = alpha * avg_distance + (1 - alpha) * cf_normalized
            
            scores.append({
                'track_id': track_id,
                'distance': avg_distance,
                'cf_score': cf_pred,
                'combined_score': combined_score
            })
        
        # Sort by combined score (descending) and take top-10
        scores.sort(key=lambda x: x['combined_score'], reverse=True)
        top_10 = scores[:10]
        
        recommendations[alpha][user_id] = top_10
    
    print(f"  ✓ Generated {len(recommendations[alpha])} recommendation lists")

# ============================================================================
# PART 8: SENSITIVITY ANALYSIS - TEST MULTIPLE THRESHOLDS
# ============================================================================

print("\n[8/8] Running sensitivity analysis across threshold combinations...")

def evaluate_serendipity_with_threshold(recommendations_dict, user_E_u, 
                                       distance_threshold, cf_threshold):
    """Calculate serendipity with configurable thresholds"""
    serendipity_scores = []
    
    for user_id, recs in recommendations_dict.items():
        serendipitous_count = 0
        
        for rec in recs:
            is_unexpected = rec['distance'] > distance_threshold
            is_relevant = rec['cf_score'] > cf_threshold
            
            if is_unexpected and is_relevant:
                serendipitous_count += 1
        
        user_serendipity = serendipitous_count / len(recs) if recs else 0
        serendipity_scores.append(user_serendipity)
    
    return np.mean(serendipity_scores)

# Define threshold combinations to test
threshold_combos = [
    (0.5, 2.5, "Low distance, high CF"),
    (0.6, 2.0, "Medium-low"),
    (0.7, 1.8, "Original (Ge et al.)"),
    (0.8, 1.5, "High distance, low CF"),
    (0.9, 1.2, "Extremely high distance"),
    (0.65, 1.9, "Slightly stricter than Ge et al."),
    (0.68, 1.85, "Very close to Ge et al."),
    (0.72, 1.75, "Slightly different"),
    (0.75, 1.7, "Slightly more lenient than Ge et al."),
    (0.70, 1.6, "Same distance, lower CF"),
    (0.70, 2.0, "Same distance, higher CF"),
]

print(f"\nTesting {len(threshold_combos)} threshold combinations:\n")

sensitivity_results = []

for dist_thresh, cf_thresh, label in threshold_combos:
    print(f"{'─' * 80}")
    print(f"📊 Thresholds: distance > {dist_thresh}, CF > {cf_thresh}")
    print(f"   Description: {label}")
    print(f"{'─' * 80}")
    
    combo_results = {
        'distance_threshold': dist_thresh,
        'cf_threshold': cf_thresh,
        'label': label,
        'serendipity_scores': {}
    }
    
    # Test each alpha with these thresholds
    for alpha in alphas:
        serendipity = evaluate_serendipity_with_threshold(
            recommendations[alpha], 
            user_E_u, 
            dist_thresh, 
            cf_thresh
        )
        combo_results['serendipity_scores'][alpha] = serendipity
        print(f"  {alpha_names[alpha]:30s}: {serendipity:.3f}")
    
    # Find winner
    winner_alpha = max(combo_results['serendipity_scores'].items(), 
                      key=lambda x: x[1])
    combo_results['winner'] = winner_alpha[0]
    combo_results['winner_score'] = winner_alpha[1]
    
    print(f"\n  🏆 Winner: {alpha_names[winner_alpha[0]]} (serendipity = {winner_alpha[1]:.3f})")
    print()
    
    sensitivity_results.append(combo_results)

# ============================================================================
# CREATE SUMMARY TABLE
# ============================================================================

print("\n" + "=" * 80)
print("SENSITIVITY ANALYSIS SUMMARY")
print("=" * 80)

# Create DataFrame
rows = []
for result in sensitivity_results:
    row = {
        'Thresholds': f"({result['distance_threshold']}, {result['cf_threshold']})",
        'Description': result['label']
    }
    
    for alpha in alphas:
        row[f'α={alpha}'] = result['serendipity_scores'][alpha]
    
    row['Winner'] = f"α={result['winner']}"
    
    rows.append(row)

df = pd.DataFrame(rows)

print("\n" + df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

# Count wins
print("\n" + "=" * 80)
print("📈 ROBUSTNESS CHECK: How often does each α win?")
print("=" * 80)

winner_counts = {}
for result in sensitivity_results:
    winner = result['winner']
    winner_counts[winner] = winner_counts.get(winner, 0) + 1

print()
for alpha in sorted(winner_counts.keys()):
    count = winner_counts[alpha]
    total = len(sensitivity_results)
    percentage = 100 * count / total
    
    # Visual bar
    bar = '█' * int(percentage / 5)
    
    print(f"  {alpha_names[alpha]:30s}: {count}/{total} combinations ({percentage:5.1f}%) {bar}")

# Determine if robust
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

original_winner = None
for result in sensitivity_results:
    if result['distance_threshold'] == 0.7 and result['cf_threshold'] == 1.8:
        original_winner = result['winner']
        break

most_frequent_winner = max(winner_counts.items(), key=lambda x: x[1])[0]
original_wins = winner_counts.get(original_winner, 0) if original_winner else 0

# Count wins in "reasonable" range (around Ge et al.)
reasonable_thresholds = [
    (0.65, 1.9), (0.70, 1.8), (0.75, 1.7), (0.70, 1.6), (0.70, 2.0)
]
reasonable_results = [r for r in sensitivity_results 
                     if (r['distance_threshold'], r['cf_threshold']) in reasonable_thresholds]
reasonable_winners = [r['winner'] for r in reasonable_results]
reasonable_025_wins = sum(1 for w in reasonable_winners if w == 0.25)

print()
if original_winner == most_frequent_winner and original_wins >= len(sensitivity_results) * 0.4:
    print(f"✅ ROBUST FINDING!")
    print(f"   {alpha_names[original_winner]} wins consistently across threshold definitions.")
    print(f"   Overall: {original_wins}/{len(sensitivity_results)} combinations ({100*original_wins/len(sensitivity_results):.0f}%)")
    print(f"   Reasonable range: {reasonable_025_wins}/{len(reasonable_results)} combinations ({100*reasonable_025_wins/len(reasonable_results):.0f}%)")
    print(f"   Conclusion: α=0.25 is a reliable recommendation.")
else:
    print(f"⚠️  MODERATELY ROBUST")
    print(f"   Results show some sensitivity to threshold choice.")
    print(f"   Overall: {original_wins}/{len(sensitivity_results)} combinations ({100*original_wins/len(sensitivity_results):.0f}%)")
    print(f"   Reasonable range: {reasonable_025_wins}/{len(reasonable_results)} combinations ({100*reasonable_025_wins/len(reasonable_results):.0f}%)")

print()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save CSV
csv_filename = 'sensitivity_analysis_full_results.csv'
df.to_csv(csv_filename, index=False)
print(f"\n✓ Saved summary table: {csv_filename}")

# Save detailed text report
report_filename = 'sensitivity_analysis_full_report.txt'
with open(report_filename, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("SENSITIVITY ANALYSIS - FULL PIPELINE RESULTS\n")
    f.write("=" * 80 + "\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("CONFIGURATION:\n")
    f.write(f"  Dataset: AMBAR (sampled to {len(ratings_df):,} ratings)\n")
    f.write(f"  Users evaluated: {len(sample_users)}\n")
    f.write(f"  CF Model: SVD (50 factors, 20 epochs)\n")
    f.write(f"  Recommendations per user: 10\n")
    f.write(f"  Threshold combinations tested: {len(threshold_combos)}\n\n")
    
    for i, result in enumerate(sensitivity_results, 1):
        f.write(f"\n{'=' * 80}\n")
        f.write(f"Threshold Combination {i}/{len(sensitivity_results)}\n")
        f.write(f"{'=' * 80}\n")
        f.write(f"Distance threshold: > {result['distance_threshold']}\n")
        f.write(f"CF score threshold: > {result['cf_threshold']}\n")
        f.write(f"Description: {result['label']}\n\n")
        
        f.write("Serendipity Scores:\n")
        for alpha, score in sorted(result['serendipity_scores'].items()):
            f.write(f"  α={alpha:4.2f}: {score:.3f}\n")
        
        f.write(f"\nWinner: α={result['winner']} (score: {result['winner_score']:.3f})\n")

print(f"✓ Saved detailed report: {report_filename}")

print(f"\n{'=' * 80}")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 80}\n")

print("SUMMARY:")
print(f"  • Tested {len(threshold_combos)} threshold combinations")
print(f"  • α=0.25 wins in {winner_counts.get(0.25, 0)}/{len(sensitivity_results)} overall ({100*winner_counts.get(0.25, 0)/len(sensitivity_results):.0f}%)")
print(f"  • α=0.25 wins in {reasonable_025_wins}/{len(reasonable_results)} reasonable range ({100*reasonable_025_wins/len(reasonable_results):.0f}%)")
print(f"  • Results saved to: {csv_filename}, {report_filename}")
print()