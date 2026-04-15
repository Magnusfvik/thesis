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

# Set style for better plots
plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("EXPERIMENT 1: Validating the Unimodal Relationship (RQ1.1)")
print("=" * 80)

# Load the three main AMBAR files
print("\n[1/9] Loading AMBAR dataset...")

# Adjust paths to your local files
ratings_df = pd.read_csv('AMBAR/ratings_info.csv')
tracks_df = pd.read_csv('AMBAR/tracks_info.csv')
users_df = pd.read_csv('AMBAR/users_info.csv')  # If you have it

print(f"  ✓ Ratings: {len(ratings_df):,} rows")
print(f"  ✓ Tracks: {len(tracks_df):,} rows")
print(f"  ✓ Users: {ratings_df['user_id'].nunique():,} unique")

# Show first few rows to understand structure
print("\nRatings sample:")
print(ratings_df.head())

print("\nTracks sample:")
print(tracks_df.head())


print("\n[2/9] Preprocessing data...")

# 1. Parse styles from string to list
def parse_styles(style_string):
    """Convert string representation of list to actual list"""
    if pd.isna(style_string):
        return []
    if isinstance(style_string, list):
        return style_string
    try:
        # Try parsing as Python literal (handles both ['style1', 'style2'] and "['style1', 'style2']")
        return ast.literal_eval(style_string)
    except:
        # Fallback: split by comma
        return [s.strip() for s in str(style_string).split(',') if s.strip()]

tracks_df['styles'] = tracks_df['styles'].apply(parse_styles)

print(f"  ✓ Parsed styles for {len(tracks_df)} tracks")

# 2. Remove tracks without style metadata
tracks_before = len(tracks_df)
tracks_df = tracks_df[tracks_df['styles'].apply(len) > 0]
tracks_after = len(tracks_df)
print(f"  ✓ Removed {tracks_before - tracks_after:,} tracks without styles")

# 3. Remove users with too few ratings
ratings_before = len(ratings_df)
user_rating_counts = ratings_df.groupby('user_id').size()
valid_users = user_rating_counts[user_rating_counts >= 20].index
ratings_df = ratings_df[ratings_df['user_id'].isin(valid_users)]
ratings_after = len(ratings_df)
print(f"  ✓ Removed users with <20 ratings: {ratings_before - ratings_after:,} ratings dropped")

# 4. Keep only tracks that exist in tracks_df
ratings_df = ratings_df[ratings_df['track_id'].isin(tracks_df['track_id'])]
print(f"  ✓ Final dataset: {len(ratings_df):,} ratings, {ratings_df['user_id'].nunique():,} users")

# 5. Convert ratings to 1-5 scale if needed
# (AMBAR uses implicit ratings, might need normalization)
if ratings_df['rating'].max() > 5:
    print("  ⚠ Normalizing ratings to 1-5 scale...")
    ratings_df['rating'] = pd.cut(ratings_df['rating'], 
                                   bins=5, 
                                   labels=[1, 2, 3, 4, 5]).astype(float)

print("\nFinal statistics:")
print(f"  Users: {ratings_df['user_id'].nunique():,}")
print(f"  Tracks: {len(tracks_df):,}")
print(f"  Ratings: {len(ratings_df):,}")
print(f"  Sparsity: {100 * (1 - len(ratings_df) / (ratings_df['user_id'].nunique() * len(tracks_df))):.2f}%")

# After loading ratings_df, add:
print("\n=== RATING DIAGNOSTIC ===")
print(f"Rating distribution:")
print(ratings_df['rating'].value_counts().sort_index())
print(f"\nRating stats:")
print(ratings_df['rating'].describe())
print("=" * 50)

print("\n[3/9] Creating train/test split (80/20 per user)...")

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

print(f"  ✓ Train set: {len(train_df):,} ratings ({len(train_df)/len(ratings_df)*100:.1f}%)")
print(f"  ✓ Test set: {len(test_df):,} ratings ({len(test_df)/len(ratings_df)*100:.1f}%)")

# Statistics
train_per_user = train_df.groupby('user_id').size()
test_per_user = test_df.groupby('user_id').size()

print(f"\nPer-user statistics:")
print(f"  Train: mean={train_per_user.mean():.1f}, median={train_per_user.median():.0f}")
print(f"  Test: mean={test_per_user.mean():.1f}, median={test_per_user.median():.0f}")


print("\n[4/9] Sampling users for experiment...")

# Filter: users with at least 30 train ratings and 10 test ratings
train_counts = train_df.groupby('user_id').size()
test_counts = test_df.groupby('user_id').size()

eligible_users = train_counts[
    (train_counts >= 30) & 
    (test_counts >= 10)
].index

print(f"  ✓ Eligible users: {len(eligible_users):,}")

# Random sample of 100 users
np.random.seed(42)
sample_users = np.random.choice(eligible_users, size=min(100, len(eligible_users)), replace=False)

print(f"  ✓ Sampled {len(sample_users)} users for experiment")

# Filter data to sample users
train_sample = train_df[train_df['user_id'].isin(sample_users)]
test_sample = test_df[test_df['user_id'].isin(sample_users)]

print(f"  ✓ Sample train ratings: {len(train_sample):,}")
print(f"  ✓ Sample test ratings: {len(test_sample):,}")

print("\n[5/9] Building Expected Set (E_u) for each user...")

def build_E_u_moderate(user_id, train_df, tracks_df):
    """
    Build Moderate E_u: rated tracks + all tracks by same artists
    
    Returns:
        E_u_ids: set of track IDs in E_u
        E_u_styles: set of all styles in E_u (for distance computation)
    """
    # Get user's rated tracks in training set
    user_train = train_df[train_df['user_id'] == user_id]
    rated_track_ids = set(user_train['track_id'].values)
    
    # Get artists from rated tracks
    rated_tracks_info = tracks_df[tracks_df['track_id'].isin(rated_track_ids)]
    known_artists = set(rated_tracks_info['artist_id'].values)
    
    # E_u = rated tracks + all tracks by known artists
    E_u_tracks = tracks_df[
        (tracks_df['track_id'].isin(rated_track_ids)) |
        (tracks_df['artist_id'].isin(known_artists))
    ]
    
    E_u_ids = set(E_u_tracks['track_id'].values)
    
    # Collect all unique styles in E_u
    all_styles = []
    for styles_list in E_u_tracks['styles']:
        all_styles.extend(styles_list)
    E_u_styles = set(all_styles)
    
    return E_u_ids, E_u_styles

# Build E_u for all sample users
user_E_u = {}

for user_id in sample_users:
    E_u_ids, E_u_styles = build_E_u_moderate(user_id, train_sample, tracks_df)
    user_E_u[user_id] = {
        'E_u_ids': E_u_ids,
        'E_u_styles': E_u_styles
    }

# Statistics
E_u_sizes = [len(data['E_u_ids']) for data in user_E_u.values()]
print(f"  ✓ Built E_u for {len(user_E_u)} users")
print(f"  ✓ E_u size: mean={np.mean(E_u_sizes):.0f}, median={np.median(E_u_sizes):.0f}")
print(f"  ✓ E_u coverage: {np.mean(E_u_sizes) / len(tracks_df) * 100:.2f}% of catalog")

print("\n[6/9] Computing distances for test tracks...")

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

def distance_from_expected(track_styles, E_u_styles):
    """
    Compute average Jaccard distance from track to E_u
    
    For simplicity, we use aggregate E_u styles (union of all styles)
    Alternative: average distance to each track in E_u (more computationally expensive)
    """
    return jaccard_distance(track_styles, E_u_styles)

# Collect results
results = []

for user_id in sample_users:
    # Get user's E_u
    E_u_styles = user_E_u[user_id]['E_u_styles']
    
    # Get user's test ratings
    user_test = test_sample[test_sample['user_id'] == user_id]
    
    for _, row in user_test.iterrows():
        track_id = row['track_id']
        actual_rating = row['rating']
        
        # Get track styles
        track_info = tracks_df[tracks_df['track_id'] == track_id]
        if len(track_info) == 0:
            continue
        
        track_styles = set(track_info.iloc[0]['styles'])
        
        # Compute distance
        distance = distance_from_expected(track_styles, E_u_styles)
        
        results.append({
            'user_id': user_id,
            'track_id': track_id,
            'distance': distance,
            'rating': actual_rating
        })

df_results = pd.DataFrame(results)

print(f"  ✓ Computed distances for {len(df_results):,} test tracks")
print(f"  ✓ Distance range: [{df_results['distance'].min():.3f}, {df_results['distance'].max():.3f}]")
print(f"  ✓ Mean distance: {df_results['distance'].mean():.3f}")
print(f"  ✓ Rating range: [{df_results['rating'].min():.1f}, {df_results['rating'].max():.1f}]")

print("\n=== DISTANCE DIAGNOSTIC ===")
print(f"Total test tracks: {len(df_results)}")
print(f"\nDistance distribution:")
print(df_results['distance'].describe())
print(f"\nDistance histogram:")
print(df_results['distance'].value_counts(bins=10, sort=False))
print(f"\nSample of results:")
print(df_results.head(20)[['user_id', 'track_id', 'distance', 'rating']])
print("=" * 50)