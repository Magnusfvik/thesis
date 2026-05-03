"""
Fair Generation: All Alpha Values in One Run
=============================================

Generates recommendations for ALL alpha values using the SAME:
- CF model (one training session)
- User sample (same 100 users)
- Random seed (reproducible)
- Training/test split

This ensures fair comparison with no generation artifacts.

Alpha values: [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 
               0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]

Runtime: ~16-18 hours for 17 α values × 100 users

Author: [Your Name]
Date: 2026-05-03
"""

import pandas as pd
import numpy as np
import pickle
from surprise import SVD, Dataset, Reader
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set global seed for reproducibility
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

print("=" * 80)
print("FAIR GENERATION: ALL ALPHA VALUES IN ONE RUN")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Complete alpha grid for fair comparison
# Optimized: 11 values instead of 17 (35% faster, still comprehensive)
ALPHA_VALUES = [
    0.0,        # Pure CF (baseline)
    0.15,       # Lower transition
    0.25,       # Original optimal candidate
    0.3,        # Expected peak
    0.35,       # Grid search optimal
    0.4,        # Upper optimal region
    0.5,        # Balanced
    0.65,       # Leaving optimal
    0.75,       # Distance-biased
    0.85,       # Upper transition
    1.0         # Pure Distance (baseline)
]

print(f"Will generate for {len(ALPHA_VALUES)} α values (optimized grid):")
print(f"  {ALPHA_VALUES}")
print(f"  Focus: Optimal region [0.25-0.4] + extremes + transitions")
print(f"  Estimated runtime: ~4-5 hours (vs 16-18 for full grid)")
print()

# ============================================================================
# LOAD USER DATA
# ============================================================================

print("[1/7] Loading user data...")

try:
    user_E_u = pickle.load(open('user_E_u.pkl', 'rb'))
    user_ids = list(user_E_u.keys())
    print(f"  ✓ Loaded {len(user_ids)} users")
except FileNotFoundError:
    print("  ❌ ERROR: user_E_u.pkl not found!")
    exit(1)

# ============================================================================
# LOAD AMBAR DATA
# ============================================================================

print("\n[2/7] Loading AMBAR dataset...")

ratings_df = pd.read_csv('AMBAR/ratings_info.csv')
tracks_df = pd.read_csv('AMBAR/tracks_info.csv')

# Use sample (for speed)
USE_SAMPLE = True
if USE_SAMPLE:
    ratings_df = ratings_df.sample(n=200000, random_state=GLOBAL_SEED)

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

print("\n[4/7] Creating train/test split (80/20)...")

train_data = []
for user_id in ratings_df['user_id'].unique():
    user_ratings = ratings_df[ratings_df['user_id'] == user_id].copy()
    
    if 'timestamp' in user_ratings.columns:
        user_ratings = user_ratings.sort_values('timestamp')
    else:
        user_ratings = user_ratings.sample(frac=1, random_state=GLOBAL_SEED)
    
    n_train = int(0.8 * len(user_ratings))
    train_data.append(user_ratings.iloc[:n_train])

train_df = pd.concat(train_data, ignore_index=True)
print(f"  ✓ Train set: {len(train_df):,} ratings")

# ============================================================================
# TRAIN CF MODEL - ONCE FOR ALL ALPHA VALUES
# ============================================================================

print("\n[5/7] Training CF model (SVD) - ONCE for all α values...")

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train_df[['user_id', 'track_id', 'rating']], reader)

svd = SVD(n_factors=50, n_epochs=20, random_state=GLOBAL_SEED, verbose=False)
trainset = data.build_full_trainset()
svd.fit(trainset)

print(f"  ✓ CF model trained (will be used for ALL α values)")
print(f"  ✓ This ensures fair comparison with no CF model variance")

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
# PRE-COMPUTE DISTANCES (HUGE OPTIMIZATION!)
# ============================================================================

print(f"\n[6a/8] Pre-computing distances for all users...")
print("  (Distance doesn't depend on α, so compute once and reuse)")
print()

user_track_distances = {}
user_track_cf = {}

precompute_start = datetime.now()

for i, user_id in enumerate(user_ids, 1):
    if i % 10 == 0 or i == 1:
        elapsed = (datetime.now() - precompute_start).total_seconds()
        if i > 1:
            rate = elapsed / (i - 1)
            remaining = rate * (len(user_ids) - i + 1) / 60
            print(f"  Users: {i}/{len(user_ids)} (~{remaining:.0f}m remaining)")
    
    E_u_data = user_E_u[user_id]
    E_u = E_u_data['E_u_ids'] if isinstance(E_u_data, dict) else E_u_data
    candidate_tracks = all_tracks - E_u
    
    user_track_distances[user_id] = {}
    user_track_cf[user_id] = {}
    
    for track_id in candidate_tracks:
        # Pre-compute CF prediction (same for all α)
        cf_pred = svd.predict(user_id, track_id).est
        user_track_cf[user_id][track_id] = cf_pred
        
        # Pre-compute distance (same for all α)
        if track_id not in track_metadata:
            continue
        
        track_set = track_metadata[track_id]
        distances = []
        
        for e_track in E_u:
            if e_track in track_metadata:
                e_set = track_metadata[e_track]
                dist = jaccard_distance(track_set, e_set)
                distances.append(dist)
        
        if distances:
            user_track_distances[user_id][track_id] = np.mean(distances)

precompute_time = (datetime.now() - precompute_start).total_seconds() / 60
print(f"\n  ✓ Pre-computation complete in {precompute_time:.1f} minutes")
print(f"  ✓ Computed distances and CF scores for all (user, track) pairs")
print(f"  ✓ This will save ~{len(ALPHA_VALUES)-1} × {precompute_time:.0f}m = " +
      f"{(len(ALPHA_VALUES)-1) * precompute_time / 60:.1f}h total!")

# ============================================================================
# GENERATE RECOMMENDATIONS FOR ALL ALPHA VALUES (FAST!)
# ============================================================================

print(f"\n[6b/8] Generating recommendations for ALL {len(ALPHA_VALUES)} α values...")
print("  (Using pre-computed distances - much faster!)")
print()

all_recommendations = {}
overall_start = datetime.now()

for alpha_idx, alpha in enumerate(ALPHA_VALUES, 1):
    print(f"\n{'=' * 80}")
    print(f"α = {alpha:.2f} ({alpha_idx}/{len(ALPHA_VALUES)})")
    print(f"{'=' * 80}")
    
    all_recommendations[alpha] = {}
    alpha_start = datetime.now()
    
    for i, user_id in enumerate(user_ids, 1):
        if i % 10 == 0 or i == 1:
            elapsed = (datetime.now() - alpha_start).total_seconds()
            if i > 1:
                rate = elapsed / (i - 1)
                remaining = rate * (len(user_ids) - i + 1) / 60
                
                # Overall progress
                alphas_done = alpha_idx - 1
                alphas_remaining = len(ALPHA_VALUES) - alpha_idx
                current_alpha_progress = (i - 1) / len(user_ids)
                total_progress = (alphas_done + current_alpha_progress) / len(ALPHA_VALUES)
                
                overall_elapsed = (datetime.now() - overall_start).total_seconds() / 60
                estimated_total = overall_elapsed / total_progress if total_progress > 0 else 0
                overall_remaining = estimated_total - overall_elapsed
                
                print(f"  Users: {i}/{len(user_ids)} | "
                      f"This α: ~{remaining:.0f}m | "
                      f"Total: {total_progress*100:.0f}% (~{overall_remaining:.0f}m)")
            else:
                print(f"  Users: {i}/{len(user_ids)} (starting...)")
        
        # Extract E_u
        E_u_data = user_E_u[user_id]
        E_u = E_u_data['E_u_ids'] if isinstance(E_u_data, dict) else E_u_data
        
        # Use pre-computed distances and CF scores!
        scores = []
        
        for track_id in user_track_distances[user_id]:
            avg_distance = user_track_distances[user_id][track_id]
            cf_pred = user_track_cf[user_id][track_id]
            
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
    print(f"\n  ✓ Completed α={alpha:.2f} in {alpha_time:.1f} minutes")
    
    # CHECKPOINT after each α
    pickle.dump(all_recommendations, open('recommendations_fair_complete.pkl', 'wb'))
    print(f"  ✓ Checkpoint saved")

# ============================================================================
# FINAL SAVE
# ============================================================================

print("\n[7/8] Finalizing...")

pickle.dump(all_recommendations, open('recommendations_fair_complete.pkl', 'wb'))
print(f"  ✓ Final save: recommendations_fair_complete.pkl")
print(f"  ✓ Contains {len(all_recommendations)} α values")

# Save metadata about generation
generation_info = {
    'alpha_values': ALPHA_VALUES,
    'n_users': len(user_ids),
    'global_seed': GLOBAL_SEED,
    'cf_params': {'n_factors': 50, 'n_epochs': 20},
    'sample_size': len(ratings_df),
    'generation_date': datetime.now().isoformat(),
}

pickle.dump(generation_info, open('generation_info.pkl', 'wb'))
print(f"  ✓ Saved generation metadata")

total_time = (datetime.now() - overall_start).total_seconds() / 3600
print(f"\n{'=' * 80}")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total runtime: {total_time:.1f} hours")
print(f"{'=' * 80}")
print()
print("✓ Fair generation complete!")
print("✓ All α values generated with SAME CF model")
print("→ Ready for clean PSO validation!")
print()