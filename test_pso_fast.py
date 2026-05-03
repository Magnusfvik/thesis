"""
FAST TEST: PSO Proof of Concept
================================

Tests PSO approach with SMALL SAMPLE to validate theory before full run.

Configuration:
- 25 users (instead of 100)
- Coarse grid: α = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

Runtime: ~30-45 minutes (vs 8-10 hours for full)

If this works, we run the full version overnight.

Author: [Your Name]
Date: 2026-04-29
"""

import pandas as pd
import numpy as np
import pickle
import os
from surprise import SVD, Dataset, Reader
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FAST TEST: PSO PROOF OF CONCEPT")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# CONFIGURATION - FAST VERSION
# ============================================================================

# Coarse grid for testing
ALPHA_GRID_TEST = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Small user sample
N_TEST_USERS = 25  # Instead of 100

print(f"TEST CONFIGURATION:")
print(f"  Alpha grid: {ALPHA_GRID_TEST} ({len(ALPHA_GRID_TEST)} values)")
print(f"  Users: {N_TEST_USERS} (instead of 100)")
print(f"  Estimated runtime: ~{len(ALPHA_GRID_TEST) * 5} minutes\n")

# ============================================================================
# LOAD USER DATA
# ============================================================================

print("[1/7] Loading user data...")

try:
    user_E_u_full = pickle.load(open('user_E_u.pkl', 'rb'))
    print(f"  ✓ Loaded {len(user_E_u_full)} users from full dataset")
except FileNotFoundError:
    print("  ❌ ERROR: user_E_u.pkl not found!")
    exit(1)

# Sample random users for testing
np.random.seed(42)  # Reproducible
all_user_ids = list(user_E_u_full.keys())
test_user_ids = np.random.choice(all_user_ids, size=N_TEST_USERS, replace=False)

# Create subset
user_E_u = {uid: user_E_u_full[uid] for uid in test_user_ids}

print(f"  ✓ Sampled {len(user_E_u)} users for testing")
print(f"  ✓ User IDs: {list(test_user_ids)[:5]}... (showing first 5)")

# ============================================================================
# LOAD AMBAR DATA
# ============================================================================

print("\n[2/7] Loading AMBAR dataset...")

ratings_df = pd.read_csv('AMBAR/ratings_info.csv')
tracks_df = pd.read_csv('AMBAR/tracks_info.csv')

# Use sample
USE_SAMPLE = True
if USE_SAMPLE:
    ratings_df = ratings_df.sample(n=200000, random_state=42)

print(f"  ✓ Using {len(ratings_df):,} ratings")

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

track_metadata = {}
for _, track in tracks_df.iterrows():
    track_metadata[track['track_id']] = set(track['category_styles'])

print(f"  ✓ Pre-computed metadata for {len(track_metadata)} tracks")

all_tracks = set(tracks_df['track_id'].unique())

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

print("\n[4/7] Creating train/test split...")

train_data = []
for user_id in ratings_df['user_id'].unique():
    user_ratings = ratings_df[ratings_df['user_id'] == user_id].copy()
    
    if 'timestamp' in user_ratings.columns:
        user_ratings = user_ratings.sort_values('timestamp')
    else:
        user_ratings = user_ratings.sample(frac=1, random_state=42)
    
    n_train = int(0.8 * len(user_ratings))
    train_data.append(user_ratings.iloc[:n_train])

train_df = pd.concat(train_data, ignore_index=True)
print(f"  ✓ Train set: {len(train_df):,} ratings")

# ============================================================================
# TRAIN CF MODEL
# ============================================================================

print("\n[5/7] Training CF model (SVD)...")

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train_df[['user_id', 'track_id', 'rating']], reader)

svd = SVD(n_factors=50, n_epochs=20, random_state=42, verbose=False)
trainset = data.build_full_trainset()
svd.fit(trainset)

print(f"  ✓ CF model trained")

# ============================================================================
# DISTANCE FUNCTION
# ============================================================================

def jaccard_distance(set1, set2):
    if len(set1) == 0 or len(set2) == 0:
        return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 1 - (intersection / union) if union > 0 else 1.0

# ============================================================================
# GENERATE RECOMMENDATIONS - FAST VERSION
# ============================================================================

print("\n[6/7] Generating recommendations (FAST TEST)...")

all_recommendations = {}

overall_start = datetime.now()

for alpha_idx, alpha in enumerate(ALPHA_GRID_TEST, 1):
    print(f"\n  α={alpha:.1f} ({alpha_idx}/{len(ALPHA_GRID_TEST)})")
    
    all_recommendations[alpha] = {}
    alpha_start = datetime.now()
    
    for i, user_id in enumerate(test_user_ids, 1):
        if i % 5 == 0 or i == 1:
            elapsed = (datetime.now() - alpha_start).total_seconds()
            if i > 1:
                rate = elapsed / (i - 1)
                remaining = rate * (len(test_user_ids) - i + 1) / 60
                print(f"    Progress: {i}/{len(test_user_ids)} users (~{remaining:.1f}m remaining)")
        
        # Extract E_u
        E_u_data = user_E_u[user_id]
        E_u = E_u_data['E_u_ids'] if isinstance(E_u_data, dict) else E_u_data
        
        candidate_tracks = all_tracks - E_u
        
        scores = []
        
        for track_id in candidate_tracks:
            cf_pred = svd.predict(user_id, track_id).est
            
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
            
            combined_score = alpha * avg_distance + (1 - alpha) * cf_normalized
            
            scores.append({
                'track_id': track_id,
                'distance': avg_distance,
                'cf_score': cf_pred,
                'combined_score': combined_score
            })
        
        scores.sort(key=lambda x: x['combined_score'], reverse=True)
        all_recommendations[alpha][user_id] = scores[:10]
    
    alpha_time = (datetime.now() - alpha_start).total_seconds() / 60
    print(f"    ✓ Completed in {alpha_time:.1f} minutes")

# ============================================================================
# CALCULATE SERENDIPITY
# ============================================================================

print("\n[7/7] Calculating serendipity...")

DISTANCE_THRESHOLD = 0.7
CF_THRESHOLD = 1.8

def calculate_serendipity(recs_dict):
    user_scores = []
    for user_id, recs in recs_dict.items():
        serendipitous = sum(1 for rec in recs 
                           if rec['distance'] > DISTANCE_THRESHOLD 
                           and rec['cf_score'] > CF_THRESHOLD)
        user_scores.append(serendipitous / len(recs) if recs else 0)
    return np.array(user_scores)

results = []

print()
for alpha in ALPHA_GRID_TEST:
    scores = calculate_serendipity(all_recommendations[alpha])
    
    results.append({
        'Alpha': alpha,
        'Mean': np.mean(scores),
        'Std': np.std(scores),
        'Median': np.median(scores)
    })
    
    print(f"  α={alpha:.1f}: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")

df = pd.DataFrame(results)

# Save test results
pickle.dump(all_recommendations, open('recommendations_test_pso.pkl', 'wb'))
df.to_csv('test_pso_results.csv', index=False, float_format='%.4f')

print(f"\n✓ Saved: recommendations_test_pso.pkl")
print(f"✓ Saved: test_pso_results.csv")

# ============================================================================
# QUICK ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("QUICK ANALYSIS")
print("=" * 80)
print()

optimal_idx = df['Mean'].idxmax()
optimal_alpha = df.loc[optimal_idx, 'Alpha']
optimal_score = df.loc[optimal_idx, 'Mean']

print(f"Optimal α: {optimal_alpha:.1f} (serendipity: {optimal_score:.3f})")
print()

# Check if pattern makes sense
print("Serendipity curve:")
for _, row in df.iterrows():
    bar_length = int(row['Mean'] * 50)
    bar = '█' * bar_length
    print(f"  α={row['Alpha']:.1f}: {bar} {row['Mean']:.3f}")

print()
print("✓ Does this look like it peaks around α=0.2-0.4?")
print("✓ If yes, PSO will work! Run full version overnight.")
print("✓ If no, check data or parameters.")

total_time = (datetime.now() - overall_start).total_seconds() / 60
print(f"\n{'=' * 80}")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total runtime: {total_time:.1f} minutes")
print(f"{'=' * 80}")
print()
print("NEXT STEPS:")
print("  1. Review results above")
print("  2. If pattern looks good, run full version:")
print("     python generate_dense_alpha_grid.py")
print("  3. Then implement PSO tomorrow!")
print()