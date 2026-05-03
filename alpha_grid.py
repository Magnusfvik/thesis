"""
OPTIMIZED Dense Alpha Grid Generation for PSO
==============================================
 
Based on fast test results showing plateau at α ∈ [0.2, 0.6],
this version uses a SMART grid that:
- Focuses on transition regions (edges of plateau)
- Skips redundant points in plateau
- Minimizes runtime while maintaining PSO quality
 
Alpha grid: [0.0, 0.15, 0.2, 0.35, 0.5, 0.65, 0.75, 0.85, 1.0]
Need to generate: [0.15, 0.2, 0.35, 0.65, 0.85] (5 values)
 
Runtime: ~5 hours (vs 8 hours for full dense grid)
 
Usage:
    caffeinate -i python generate_optimized_alpha_grid.py
 
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
print("OPTIMIZED ALPHA GRID GENERATION FOR PSO")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
 
# ============================================================================
# CONFIGURATION - OPTIMIZED BASED ON TEST
# ============================================================================
 
# Smart grid focusing on transition regions
ALPHA_GRID_OPTIMIZED = [
    0.0,    # Pure CF (have it)
    0.15,   # Rising edge (NEED)
    0.2,    # Plateau start (NEED)  
    0.25,   # Plateau (have it)
    0.35,   # Plateau middle (NEED)
    0.5,    # Plateau (have it)
    0.65,   # Falling edge (NEED)
    0.75,   # Falling (have it)
    0.85,   # Falling edge (NEED)
    1.0     # Pure Distance (have it)
]
 
print(f"OPTIMIZED GRID based on test results:")
print(f"  Test showed plateau at α ∈ [0.2, 0.6]")
print(f"  → Focusing on transition edges")
print(f"  → Grid: {ALPHA_GRID_OPTIMIZED}")
print()
 
# ============================================================================
# CHECK EXISTING WORK
# ============================================================================
 
print("[0/7] Checking existing recommendations...")
 
existing_recs = {}
if os.path.exists('recommendations.pkl'):
    existing_recs = pickle.load(open('recommendations.pkl', 'rb'))
    print(f"  ✓ Found existing: {list(existing_recs.keys())}")
 
# Check for previous run
if os.path.exists('recommendations_dense_grid.pkl'):
    prev_run = pickle.load(open('recommendations_dense_grid.pkl', 'rb'))
    existing_recs.update(prev_run)
    print(f"  ✓ Found previous grid: {list(prev_run.keys())}")
 
alphas_to_generate = [a for a in ALPHA_GRID_OPTIMIZED if a not in existing_recs]
 
if not alphas_to_generate:
    print("\n  ✓✓✓ All α values already generated!")
    print(f"  ✓ Grid complete with {len(ALPHA_GRID_OPTIMIZED)} values")
    print("\n  → Ready for PSO!")
    exit(0)
 
print(f"\n  Need to generate: {alphas_to_generate}")
print(f"  Estimated time: ~{len(alphas_to_generate)} hours")
print(f"  (vs ~8 hours for full dense grid)")
print()
 
# ============================================================================
# LOAD USER DATA
# ============================================================================
 
print("[1/7] Loading user data...")
 
try:
    user_E_u = pickle.load(open('user_E_u.pkl', 'rb'))
    print(f"  ✓ Loaded {len(user_E_u)} users")
except FileNotFoundError:
    print("  ❌ ERROR: user_E_u.pkl not found!")
    exit(1)
 
user_ids = list(user_E_u.keys())
 
# ============================================================================
# LOAD AMBAR DATA
# ============================================================================
 
print("\n[2/7] Loading AMBAR dataset...")
 
ratings_df = pd.read_csv('AMBAR/ratings_info.csv')
tracks_df = pd.read_csv('AMBAR/tracks_info.csv')
 
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
# GENERATE RECOMMENDATIONS
# ============================================================================
 
print("\n[6/7] Generating recommendations...")
 
all_recommendations = dict(existing_recs)
overall_start = datetime.now()
 
for alpha_idx, alpha in enumerate(alphas_to_generate, 1):
    print(f"\n{'=' * 80}")
    print(f"α = {alpha:.2f} ({alpha_idx}/{len(alphas_to_generate)})")
    print(f"{'=' * 80}")
    
    all_recommendations[alpha] = {}
    alpha_start = datetime.now()
    
    for i, user_id in enumerate(user_ids, 1):
        if i % 10 == 0 or i == 1:
            elapsed = (datetime.now() - alpha_start).total_seconds()
            if i > 1:
                rate = elapsed / (i - 1)
                remaining = rate * (len(user_ids) - i + 1) / 60
                
                total_progress = (alpha_idx - 1 + (i-1)/len(user_ids)) / len(alphas_to_generate)
                overall_elapsed = (datetime.now() - overall_start).total_seconds() / 60
                estimated_total = overall_elapsed / total_progress if total_progress > 0 else 0
                overall_remaining = estimated_total - overall_elapsed
                
                print(f"  Users: {i}/{len(user_ids)} | "
                      f"This α: ~{remaining:.0f}m | "
                      f"Total: {total_progress*100:.0f}% (~{overall_remaining:.0f}m)")
        
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
    print(f"\n  ✓ Completed α={alpha:.2f} in {alpha_time:.1f} minutes")
    
    # CHECKPOINT after each α
    pickle.dump(all_recommendations, open('recommendations_optimized_grid.pkl', 'wb'))
    print(f"  ✓ Checkpoint saved")
 
# ============================================================================
# FINAL SAVE
# ============================================================================
 
print("\n[7/7] Finalizing...")
 
pickle.dump(all_recommendations, open('recommendations_optimized_grid.pkl', 'wb'))
print(f"  ✓ Final save: recommendations_optimized_grid.pkl")
print(f"  ✓ Contains {len(all_recommendations)} α values")
 
# Verify
missing = [a for a in ALPHA_GRID_OPTIMIZED if a not in all_recommendations]
if missing:
    print(f"\n  ⚠ WARNING: Missing {missing}")
else:
    print(f"\n  ✓✓✓ COMPLETE! All {len(ALPHA_GRID_OPTIMIZED)} α values ready!")
 
total_time = (datetime.now() - overall_start).total_seconds() / 3600
print(f"\n{'=' * 80}")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Runtime: {total_time:.1f} hours")
print(f"Time saved: ~{8-total_time:.1f} hours vs full dense grid")
print(f"{'=' * 80}")
print()
print("✓ Optimized grid complete!")
print("→ Ready for PSO implementation tomorrow!")
print()