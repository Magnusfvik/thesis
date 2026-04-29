"""
Alpha Granularity: Generate Neighbors of α=0.25
================================================

Generates recommendations ONLY for α=0.20 and α=0.30 to verify that
α=0.25 is a local maximum.

Runtime: ~40-60 minutes (2 α values × 100 users)

Strategy: Same as PSOMOO.py but for only 2 specific α values
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from surprise import SVD, Dataset, Reader
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GENERATE NEIGHBORS: α=0.20 and α=0.30")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Only generate these two
ALPHAS_TO_GENERATE = [0.20, 0.30]

alpha_names = {
    0.20: 'CF-Biased-Low (α=0.20)',
    0.30: 'CF-Biased-High (α=0.30)'
}

# ============================================================================
# LOAD EXISTING DATA
# ============================================================================

print("[1/7] Loading existing recommendations...")

try:
    existing_recs = pickle.load(open('recommendations.pkl', 'rb'))
    user_E_u = pickle.load(open('user_E_u.pkl', 'rb'))
    print(f"  ✓ Loaded existing recommendations for {len(existing_recs)} α values")
    print(f"  ✓ Loaded {len(user_E_u)} users")
except FileNotFoundError:
    print("  ❌ ERROR: recommendations.pkl not found!")
    print("     Please run PSOMOO.py first.")
    exit(1)

# Get user IDs
user_ids = list(user_E_u.keys())
print(f"  ✓ Will generate for {len(user_ids)} users")

# ============================================================================
# LOAD AMBAR DATA
# ============================================================================

print("\n[2/7] Loading AMBAR dataset...")

ratings_df = pd.read_csv('AMBAR/ratings_info.csv')
tracks_df = pd.read_csv('AMBAR/tracks_info.csv')

print(f"  ✓ Ratings: {len(ratings_df):,} rows")
print(f"  ✓ Tracks: {len(tracks_df):,} rows")

# Use same sample as PSOMOO.py
USE_SAMPLE = True
if USE_SAMPLE:
    print("\n  [SAMPLE MODE] Using 200K ratings...")
    ratings_df = ratings_df.sample(n=200000, random_state=42)
    print(f"  ✓ Sampled to {len(ratings_df):,} ratings")

# ============================================================================
# PARSE METADATA
# ============================================================================

print("\n[3/7] Parsing track metadata...")

def parse_styles(style_string):
    if pd.isna(style_string):
        return []
    if isinstance(style_string, list):
        return style_string
    if isinstance(style_string, str):
        return [s.strip() for s in style_string.split('|')]
    return []

tracks_df['category_styles'] = tracks_df['category_styles'].apply(parse_styles)

# Pre-compute track metadata
track_metadata = {}
for _, track in tracks_df.iterrows():
    track_metadata[track['track_id']] = set(track['category_styles'])

print(f"  ✓ Pre-computed metadata for {len(track_metadata)} tracks")

# Get all tracks
all_tracks = set(tracks_df['track_id'].unique())

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

print("\n[4/7] Creating train/test split (80/20)...")

train_data = []
test_data = []

for user_id in ratings_df['user_id'].unique():
    user_ratings = ratings_df[ratings_df['user_id'] == user_id].copy()
    
    if 'timestamp' in user_ratings.columns:
        user_ratings = user_ratings.sort_values('timestamp')
    else:
        user_ratings = user_ratings.sample(frac=1, random_state=42)
    
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
# TRAIN CF MODEL
# ============================================================================

print("\n[5/7] Training CF model (SVD)...")

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train_df[['user_id', 'track_id', 'rating']], reader)

svd = SVD(n_factors=50, n_epochs=20, verbose=False)
trainset = data.build_full_trainset()
svd.fit(trainset)

print(f"  ✓ CF model trained (50 factors, 20 epochs)")

# ============================================================================
# DISTANCE FUNCTION
# ============================================================================

def jaccard_distance(set1, set2):
    """Compute Jaccard distance between two sets"""
    if len(set1) == 0 or len(set2) == 0:
        return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 1 - (intersection / union) if union > 0 else 1.0

# ============================================================================
# GENERATE RECOMMENDATIONS FOR α=0.20 and α=0.30
# ============================================================================

print("\n[6/7] Generating recommendations...")

new_recommendations = {}

for alpha in ALPHAS_TO_GENERATE:
    print(f"\n  {alpha_names[alpha]}")
    print(f"  {'─' * 60}")
    
    new_recommendations[alpha] = {}
    
    start_time = datetime.now()
    
    for i, user_id in enumerate(user_ids, 1):
        if i % 10 == 0 or i == 1:
            elapsed = (datetime.now() - start_time).total_seconds()
            if i > 1:
                rate = elapsed / (i - 1)
                remaining = rate * (len(user_ids) - i + 1) / 60
                print(f"    Progress: {i}/{len(user_ids)} users "
                      f"({elapsed/60:.1f} min elapsed, ~{remaining:.1f} min remaining)")
            else:
                print(f"    Progress: {i}/{len(user_ids)} users (starting...)")
        
        # Extract E_u
        E_u_data = user_E_u[user_id]
        E_u = E_u_data['E_u_ids'] if isinstance(E_u_data, dict) else E_u_data
        
        # Candidate tracks
        candidate_tracks = all_tracks - E_u
        
        scores = []
        
        for track_id in candidate_tracks:
            # CF prediction
            cf_pred = svd.predict(user_id, track_id).est
            
            # Distance
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
            cf_normalized = (cf_pred - 1) / 4
            
            # Weighted score
            combined_score = alpha * avg_distance + (1 - alpha) * cf_normalized
            
            scores.append({
                'track_id': track_id,
                'distance': avg_distance,
                'cf_score': cf_pred,
                'combined_score': combined_score
            })
        
        # Sort and take top-10
        scores.sort(key=lambda x: x['combined_score'], reverse=True)
        new_recommendations[alpha][user_id] = scores[:10]
    
    total_time = (datetime.now() - start_time).total_seconds() / 60
    print(f"  ✓ Completed in {total_time:.1f} minutes")

# ============================================================================
# MERGE WITH EXISTING RECOMMENDATIONS
# ============================================================================

print("\n[7/7] Merging with existing recommendations...")

# Combine all recommendations
all_recommendations = dict(existing_recs)
all_recommendations.update(new_recommendations)

# Save merged recommendations
pickle.dump(all_recommendations, open('recommendations_with_neighbors.pkl', 'wb'))
print(f"  ✓ Saved: recommendations_with_neighbors.pkl")
print(f"  ✓ Now contains {len(all_recommendations)} α values")

# ============================================================================
# CALCULATE SERENDIPITY
# ============================================================================

print("\n" + "=" * 80)
print("CALCULATING SERENDIPITY")
print("=" * 80)

DISTANCE_THRESHOLD = 0.7
CF_THRESHOLD = 1.8

def calculate_serendipity(recs_dict):
    """Calculate serendipity for a set of recommendations"""
    user_scores = []
    
    for user_id, recs in recs_dict.items():
        serendipitous = sum(1 for rec in recs 
                           if rec['distance'] > DISTANCE_THRESHOLD 
                           and rec['cf_score'] > CF_THRESHOLD)
        user_scores.append(serendipitous / len(recs) if recs else 0)
    
    return np.array(user_scores)

# Calculate for all α values including new ones
all_alphas = [0.0, 0.20, 0.25, 0.30, 0.5, 0.75, 1.0]
results = []

print()
for alpha in all_alphas:
    if alpha in all_recommendations:
        scores = calculate_serendipity(all_recommendations[alpha])
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        results.append({
            'Alpha': alpha,
            'Mean': mean_score,
            'Std': std_score,
            'Median': np.median(scores)
        })
        
        marker = "NEW!" if alpha in ALPHAS_TO_GENERATE else ""
        print(f"  α={alpha:4.2f}: mean={mean_score:.3f}, std={std_score:.3f} {marker}")

df_results = pd.DataFrame(results)

# ============================================================================
# LOCAL OPTIMUM CHECK
# ============================================================================

print("\n" + "=" * 80)
print("LOCAL OPTIMUM VERIFICATION")
print("=" * 80)
print()

neighbors = [0.20, 0.25, 0.30]
neighbor_results = df_results[df_results['Alpha'].isin(neighbors)]

print("Neighbors of α=0.25:")
for _, row in neighbor_results.iterrows():
    if row['Alpha'] == 0.25:
        print(f"  α={row['Alpha']:.2f}: {row['Mean']:.3f} ← TESTED OPTIMUM")
    else:
        diff = row['Mean'] - neighbor_results[neighbor_results['Alpha'] == 0.25]['Mean'].values[0]
        print(f"  α={row['Alpha']:.2f}: {row['Mean']:.3f} (Δ={diff:+.3f})")

print()

# Check if 0.25 is local max
alpha_020 = df_results[df_results['Alpha'] == 0.20]['Mean'].values[0]
alpha_025 = df_results[df_results['Alpha'] == 0.25]['Mean'].values[0]
alpha_030 = df_results[df_results['Alpha'] == 0.30]['Mean'].values[0]

if alpha_025 > alpha_020 and alpha_025 > alpha_030:
    print("✓✓✓ CONFIRMED: α=0.25 IS A LOCAL MAXIMUM!")
    print(f"    α=0.25 ({alpha_025:.3f}) > α=0.20 ({alpha_020:.3f})")
    print(f"    α=0.25 ({alpha_025:.3f}) > α=0.30 ({alpha_030:.3f})")
elif alpha_025 == max([alpha_020, alpha_025, alpha_030]):
    print("✓ α=0.25 ties for maximum among neighbors")
else:
    print("⚠ α=0.25 is NOT a local maximum")
    print(f"  Better alternative found!")

# Save results
df_results.to_csv('alpha_neighbors_results.csv', index=False, float_format='%.4f')
print(f"\n✓ Saved: alpha_neighbors_results.csv")

print("\n" + "=" * 80)
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print()
print("Next steps:")
print("  1. Run statistical_significance_testing.py with new recommendations")
print("  2. Create visualization with all 7 α values")
print("  3. Write LaTeX section about local optimum")
print()