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

# Set style for better plots
plt.style.use('ggplot') 
sns.set_palette("husl")

print("=" * 80)
print("EXPERIMENT 2:PSO MOO")
print("=" * 80)

# Load the three main AMBAR files
print("\n[1/9] Loading AMBAR dataset...")

# Adjust paths to your local files
ratings_df = pd.read_csv('AMBAR/ratings_info.csv')
tracks_df = pd.read_csv('AMBAR/tracks_info.csv')
users_df = pd.read_csv('AMBAR/users_info.csv')

print("\n[DEBUG] Using sample for faster testing...")
ratings_df = ratings_df.sample(n=100000, random_state=42)  # Just 100k ratings
print(f"  ✓ Sampled to {len(ratings_df):,} ratings")

print(f"  ✓ Ratings: {len(ratings_df):,} rows")
print(f"  ✓ Tracks: {len(tracks_df):,} rows")
print(f"  ✓ Users: {ratings_df['user_id'].nunique():,} unique")

# Show first few rows to understand structure
print("\nRatings sample:")
print(ratings_df.head())

print("\nTracks sample:")
print(tracks_df.head())


# Train CF model on AMBAR
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'track_id', 'rating']], reader)

# Train SVD
# Replace this line:
svd = SVD(n_factors=50, n_epochs=20)

# With this (shows progress):
svd = SVD(n_factors=50, n_epochs=20, verbose=True)
trainset = data.build_full_trainset()
svd.fit(trainset)

# For each user, predict ratings for all unrated tracks
def get_cf_predictions(user_id, unrated_tracks):
    predictions = []
    for track_id in unrated_tracks:
        pred = svd.predict(user_id, track_id).est
        predictions.append((track_id, pred))
    return predictions

print("\n[TEST] Testing CF predictions...")

# Pick a random user
test_user = ratings_df['user_id'].iloc[0]
print(f"Testing user: {test_user}")

# Get some tracks they haven't rated
user_rated = set(ratings_df[ratings_df['user_id'] == test_user]['track_id'])
all_tracks = set(tracks_df['track_id'])
unrated = list(all_tracks - user_rated)[:10]  # Get 10 unrated tracks

# Predict
predictions = get_cf_predictions(test_user, unrated)

print(f"\nTop 5 predictions for user {test_user}:")
predictions.sort(key=lambda x: x[1], reverse=True)
for track_id, pred_rating in predictions[:5]:
    track_info = tracks_df[tracks_df['track_id'] == track_id].iloc[0]
    print(f"  Track {track_id}: Predicted {pred_rating:.2f} - {track_info.get('artist_id', 'Unknown')}")

print("\n✅ CF Model trained successfully!")