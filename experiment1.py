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


# Add this to Part 2, right after the rating diagnostic:

print("\n[2b/9] Converting ratings to binary scale...")

# AMBAR ratings are skewed - treat as binary
ratings_df['liked'] = (ratings_df['rating'] >= 3).astype(int)

print(f"  Original rating distribution:")
print(f"    Rating 1-2 (not liked): {(ratings_df['rating'] < 3).sum():,} ({(ratings_df['rating'] < 3).sum()/len(ratings_df)*100:.1f}%)")
print(f"    Rating 3-5 (liked):     {(ratings_df['rating'] >= 3).sum():,} ({(ratings_df['rating'] >= 3).sum()/len(ratings_df)*100:.1f}%)")

# Map binary to 1-5 for interpretability in plots
ratings_df['rating'] = ratings_df['liked'].apply(lambda x: 5 if x == 1 else 1)

print(f"\n  ✓ New rating scale:")
print(f"    1 = Not liked (original 1-2)")
print(f"    5 = Liked (original 3-5)")
print(f"  ✓ New mean: {ratings_df['rating'].mean():.2f}")
print(f"  ✓ New distribution:")
print(ratings_df['rating'].value_counts().sort_index())


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

def distance_from_expected_proper(track_styles, E_u_ids, tracks_df, sample_size=50):
    """
    Compute average Jaccard distance from track to SAMPLED E_u tracks.
    
    This is the CORRECT way - compare track to individual E_u tracks,
    not to aggregated E_u styles.
    """
    if not track_styles:
        return 1.0
    
    # Sample E_u tracks for efficiency (otherwise too slow)
    E_u_list = list(E_u_ids)
    if len(E_u_list) > sample_size:
        E_u_sample = np.random.choice(E_u_list, size=sample_size, replace=False)
    else:
        E_u_sample = E_u_list
    
    distances = []
    for e_track_id in E_u_sample:
        e_track = tracks_df[tracks_df['track_id'] == e_track_id]
        if len(e_track) == 0:
            continue
        
        e_styles = set(e_track.iloc[0]['styles'])
        dist = jaccard_distance(track_styles, e_styles)
        distances.append(dist)
    
    return np.mean(distances) if distances else 1.0

# Collect results
results = []

print("  Processing users...")
for idx, user_id in enumerate(sample_users):
    if (idx + 1) % 10 == 0:
        print(f"    {idx + 1}/{len(sample_users)} users processed...")
    
    # Get user's E_u
    E_u_ids = user_E_u[user_id]['E_u_ids']
    
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
        
        # FIXED: Compute distance to individual E_u tracks (not aggregate styles)
        distance = distance_from_expected_proper(track_styles, E_u_ids, tracks_df, sample_size=50)
        
        results.append({
            'user_id': user_id,
            'track_id': track_id,
            'distance': distance,
            'rating': actual_rating
        })

df_results = pd.DataFrame(results)

print(f"\n  ✓ Computed distances for {len(df_results):,} test tracks")
print(f"  ✓ Distance range: [{df_results['distance'].min():.3f}, {df_results['distance'].max():.3f}]")
print(f"  ✓ Mean distance: {df_results['distance'].mean():.3f}")
print(f"  ✓ Median distance: {df_results['distance'].median():.3f}")

# Quick diagnostic
print(f"\n  Distribution check:")
for i in range(0, 10):
    low = i * 0.1
    high = (i + 1) * 0.1
    count = len(df_results[(df_results['distance'] >= low) & (df_results['distance'] < high)])
    pct = count / len(df_results) * 100
    print(f"    [{low:.1f}-{high:.1f}): {count:4d} tracks ({pct:5.1f}%)")
print("\n[7/9] Binning distances and computing statistics...")

# Create 10 distance bins
bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
bin_labels = ['[0.0-0.1)', '[0.1-0.2)', '[0.2-0.3)', '[0.3-0.4)', '[0.4-0.5)', 
              '[0.5-0.6)', '[0.6-0.7)', '[0.7-0.8)', '[0.8-0.9)', '[0.9-1.0]']

df_results['distance_bin'] = pd.cut(df_results['distance'], bins=bins, labels=bin_labels, include_lowest=True)

# Compute statistics per bin
bin_stats = df_results.groupby('distance_bin', observed=True).agg({
    'rating': ['mean', 'std', 'count'],
    'user_id': 'nunique'
}).round(2)

bin_stats.columns = ['avg_rating', 'std_rating', 'sample_size', 'num_users']

# Add percentage liked (rating >= 4)
liked_counts = df_results[df_results['rating'] >= 4].groupby('distance_bin', observed=True).size()
bin_stats['pct_liked'] = (liked_counts / bin_stats['sample_size'] * 100).fillna(0).round(0).astype(int)

print("\n" + "=" * 80)
print("DISTANCE BIN STATISTICS")
print("=" * 80)
print(bin_stats.to_string())
print("=" * 80)

# Find the bin with highest average rating
best_bin = bin_stats['avg_rating'].idxmax()
print(f"\n  ✓ Peak bin: {best_bin} with avg rating {bin_stats.loc[best_bin, 'avg_rating']:.2f}")

print("\n[8/9] Performing quadratic regression...")

# Extract bin midpoints and average ratings
# IMPORTANT: Only use bins that have data
bin_midpoints = []
avg_ratings_list = []

for i, bin_label in enumerate(bin_labels):
    if bin_label in bin_stats.index and bin_stats.loc[bin_label, 'sample_size'] > 0:
        midpoint = 0.05 + i * 0.1  # 0.05, 0.15, 0.25, ...
        avg_rating = bin_stats.loc[bin_label, 'avg_rating']
        bin_midpoints.append(midpoint)
        avg_ratings_list.append(avg_rating)

print(f"  ✓ Using {len(bin_midpoints)} bins with data (out of 10 total bins)")

if len(bin_midpoints) < 3:
    print("  ⚠ ERROR: Need at least 3 data points for quadratic regression")
    print("  → Not enough distance bins with data. Cannot proceed.")
    exit(1)

# Convert to numpy arrays
x = np.array(bin_midpoints)
y = np.array(avg_ratings_list)

# Fit quadratic model: rating = β0 + β1×distance + β2×distance²
# Create design matrix [1, x, x²]
X = np.column_stack([np.ones(len(x)), x, x**2])

# Least squares fit
coefficients, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
beta_0, beta_1, beta_2 = coefficients

print("\n" + "=" * 80)
print("QUADRATIC REGRESSION RESULTS")
print("=" * 80)
print(f"\nModel: Rating = {beta_0:.2f} + {beta_1:.2f}×distance + {beta_2:.2f}×distance²")
print(f"\nCoefficients:")
print(f"  β₀ (intercept):     {beta_0:.3f}")
print(f"  β₁ (linear):        {beta_1:.3f}")
print(f"  β₂ (quadratic):     {beta_2:.3f}")

# Compute R²
y_pred = beta_0 + beta_1 * x + beta_2 * x**2
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)

print(f"\nGoodness of fit:")
print(f"  R² = {r_squared:.3f}")
print(f"  Residual sum of squares = {ss_res:.3f}")

# Find optimal distance (peak of parabola)
if beta_2 < 0:
    delta_star = -beta_1 / (2 * beta_2)
    rating_at_peak = beta_0 + beta_1 * delta_star + beta_2 * delta_star**2
    print(f"\nOptimal distance:")
    print(f"  δ* = {delta_star:.3f}")
    print(f"  Predicted rating at δ*: {rating_at_peak:.2f}")
    
    # Check if peak is within data range
    if delta_star < min(bin_midpoints) or delta_star > max(bin_midpoints):
        print(f"  ⚠ WARNING: Peak is outside observed data range [{min(bin_midpoints):.2f}, {max(bin_midpoints):.2f}]")
else:
    print("\n⚠ WARNING: β₂ is positive - no peak found (upward parabola)")
    print("   → This suggests no unimodal relationship")
    delta_star = None
    rating_at_peak = None

print("=" * 80)

print("\n[9/9] Testing statistical significance (ANOVA)...")

# Group ratings by distance bin
groups = [df_results[df_results['distance_bin'] == bin]['rating'].dropna().values 
          for bin in bin_labels]

# Remove empty groups
groups = [g for g in groups if len(g) > 0]

# One-way ANOVA
f_stat, p_value = f_oneway(*groups)

print("\n" + "=" * 80)
print("ANOVA TEST RESULTS")
print("=" * 80)
print(f"\nNull Hypothesis (H₀): Mean ratings are equal across all distance bins")
print(f"Alternative Hypothesis (H₁): At least one bin has different mean rating")
print(f"\nF-statistic: {f_stat:.2f}")
print(f"p-value: {p_value:.6f}")

if p_value < 0.05:
    print(f"\n✅ RESULT: Reject H₀ (p < 0.05)")
    print(f"   → Distance bins have significantly different mean ratings")
else:
    print(f"\n❌ RESULT: Fail to reject H₀ (p >= 0.05)")
    print(f"   → No significant difference across distance bins")

print("=" * 80)

print("\n[10/10] Creating visualizations...")

# Extract bin midpoints and average ratings for plotting
# Use the same filtering as regression
plot_midpoints = []
plot_avg_ratings = []

for i, bin_label in enumerate(bin_labels):
    if bin_label in bin_stats.index and bin_stats.loc[bin_label, 'sample_size'] > 0:
        midpoint = 0.05 + i * 0.1
        avg_rating = bin_stats.loc[bin_label, 'avg_rating']
        plot_midpoints.append(midpoint)
        plot_avg_ratings.append(avg_rating)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Experiment 1: Validating the Unimodal Relationship (RQ1.1)', 
             fontsize=16, fontweight='bold', y=0.995)

# ===== Plot 1: Line plot with error bars =====
ax1 = axes[0, 0]

# Get std for error bars
plot_std = []
for i, bin_label in enumerate(bin_labels):
    if bin_label in bin_stats.index and bin_stats.loc[bin_label, 'sample_size'] > 0:
        plot_std.append(bin_stats.loc[bin_label, 'std_rating'])

# Plot average ratings with std as error bars
ax1.errorbar(plot_midpoints, plot_avg_ratings, 
             yerr=plot_std,
             fmt='o-', linewidth=2.5, markersize=10,
             capsize=5, capthick=2, color='#2E86AB', label='Observed data')

# Plot quadratic fit
x_smooth = np.linspace(min(plot_midpoints), max(plot_midpoints), 100)
y_smooth = beta_0 + beta_1 * x_smooth + beta_2 * x_smooth**2
ax1.plot(x_smooth, y_smooth, '--', linewidth=2, color='#A23B72', 
         label=f'Quadratic fit (R²={r_squared:.3f})')

# Mark optimal point
if delta_star is not None and rating_at_peak is not None:
    ax1.axvline(x=delta_star, color='red', linestyle=':', linewidth=2, 
                label=f'δ* = {delta_star:.3f}')
    ax1.plot(delta_star, rating_at_peak, 'r*', markersize=20, 
             label=f'Peak: {rating_at_peak:.2f}')

ax1.set_xlabel('Distance from Expected Set (E_u)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average User Rating', fontsize=12, fontweight='bold')
ax1.set_title('Distance vs. Rating', fontsize=13, fontweight='bold')
ax1.grid(alpha=0.3, linestyle='--')
ax1.legend(fontsize=10, loc='best')
y_min = min(plot_avg_ratings) - 0.5
y_max = max(plot_avg_ratings) + 0.5
ax1.set_ylim(y_min, y_max)

# ===== Plot 2: Bar chart of average ratings =====
ax2 = axes[0, 1]

# Create bar colors based on rating value
colors = []
for rating in plot_avg_ratings:
    if rating < np.percentile(plot_avg_ratings, 33):
        colors.append('#d73027')  # Low - red
    elif rating > np.percentile(plot_avg_ratings, 67):
        colors.append('#1a9850')  # High - green
    else:
        colors.append('#fee08b')  # Medium - yellow

bars = ax2.bar(range(len(plot_avg_ratings)), plot_avg_ratings, color=colors, alpha=0.7, edgecolor='black')

# Highlight peak
peak_idx = np.argmax(plot_avg_ratings)
bars[peak_idx].set_edgecolor('red')
bars[peak_idx].set_linewidth(3)

ax2.set_xlabel('Distance Bin', fontsize=12, fontweight='bold')
ax2.set_ylabel('Average Rating', fontsize=12, fontweight='bold')
ax2.set_title('Rating by Distance Bin', fontsize=13, fontweight='bold')
ax2.set_xticks(range(len(plot_avg_ratings)))
ax2.set_xticklabels([bin_labels[i] for i, _ in enumerate(bin_labels) 
                      if bin_labels[i] in bin_stats.index and bin_stats.loc[bin_labels[i], 'sample_size'] > 0], 
                     rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# ===== Plot 3: Sample size distribution =====
ax3 = axes[1, 0]

plot_sample_sizes = [bin_stats.loc[bin_labels[i], 'sample_size'] 
                     for i, _ in enumerate(bin_labels) 
                     if bin_labels[i] in bin_stats.index and bin_stats.loc[bin_labels[i], 'sample_size'] > 0]

ax3.bar(range(len(plot_sample_sizes)), plot_sample_sizes, 
        color='#4ECDC4', alpha=0.7, edgecolor='black')

ax3.set_xlabel('Distance Bin', fontsize=12, fontweight='bold')
ax3.set_ylabel('Number of Test Tracks', fontsize=12, fontweight='bold')
ax3.set_title('Sample Size Distribution', fontsize=13, fontweight='bold')
ax3.set_xticks(range(len(plot_sample_sizes)))
ax3.set_xticklabels([bin_labels[i] for i, _ in enumerate(bin_labels) 
                      if bin_labels[i] in bin_stats.index and bin_stats.loc[bin_labels[i], 'sample_size'] > 0], 
                     rotation=45, ha='right')
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# Add sample size labels on bars
for i, v in enumerate(plot_sample_sizes):
    ax3.text(i, v + max(plot_sample_sizes)*0.02, str(int(v)), ha='center', va='bottom', fontweight='bold')

# ===== Plot 4: Percentage liked (rating >= 4) =====
ax4 = axes[1, 1]

# Note: With low ratings (1-2), adjust threshold
rating_threshold = np.percentile(df_results['rating'], 75)  # Top 25%

pct_liked = []
for i, bin_label in enumerate(bin_labels):
    if bin_label in bin_stats.index and bin_stats.loc[bin_label, 'sample_size'] > 0:
        bin_data = df_results[df_results['distance_bin'] == bin_label]
        pct = (bin_data['rating'] >= rating_threshold).sum() / len(bin_data) * 100
        pct_liked.append(pct)

ax4.plot(plot_midpoints, pct_liked, 'o-', linewidth=2.5, markersize=10, color='#F18F01')

if delta_star is not None:
    ax4.axvline(x=delta_star, color='red', linestyle=':', linewidth=2, alpha=0.7)

ax4.set_xlabel('Distance from Expected Set (E_u)', fontsize=12, fontweight='bold')
ax4.set_ylabel(f'% Tracks Rated ≥ {rating_threshold:.1f} (Top 25%)', fontsize=12, fontweight='bold')
ax4.set_title('User Satisfaction by Distance', fontsize=13, fontweight='bold')
ax4.grid(alpha=0.3, linestyle='--')
ax4.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('experiment1_unimodal_relationship.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: experiment1_unimodal_relationship.png")

plt.show()


print("\n" + "=" * 80)
print("EXPERIMENT 1 SUMMARY REPORT")
print("=" * 80)

print(f"\n📊 DATA STATISTICS:")
print(f"  • Sample users: {len(sample_users)}")
print(f"  • Test tracks analyzed: {len(df_results):,}")
print(f"  • Distance range: [{df_results['distance'].min():.3f}, {df_results['distance'].max():.3f}]")
print(f"  • Rating range: [{df_results['rating'].min():.1f}, {df_results['rating'].max():.1f}]")

print(f"\n📈 QUADRATIC REGRESSION:")
print(f"  • Model: Rating = {beta_0:.2f} + {beta_1:.2f}×d + {beta_2:.2f}×d²")
print(f"  • R² = {r_squared:.3f} ({r_squared*100:.1f}% variance explained)")
print(f"  • β₂ = {beta_2:.3f} {'(NEGATIVE ✓ - confirms unimodal)' if beta_2 < 0 else '(POSITIVE ✗ - no unimodal)'}")

if delta_star is not None:
    print(f"\n🎯 OPTIMAL DISTANCE:")
    print(f"  • δ* = {delta_star:.3f}")
    print(f"  • Predicted rating at δ*: {rating_at_peak:.2f}")
    print(f"  • Interpretation: Peak {'OUTSIDE' if delta_star < 0 or delta_star > 1 else 'within'} observable range")

print(f"\n📉 DISTRIBUTION INSIGHTS:")
# Get ratings from bin_stats (already computed)
bin_avg_ratings = bin_stats['avg_rating'].values
print(f"  • Lowest rating bin: {bin_stats['avg_rating'].idxmin()} (avg: {bin_stats['avg_rating'].min():.2f})")
print(f"  • Highest rating bin: {bin_stats['avg_rating'].idxmax()} (avg: {bin_stats['avg_rating'].max():.2f})")
print(f"  • Rating range across bins: {bin_stats['avg_rating'].max() - bin_stats['avg_rating'].min():.2f} points")

print(f"\n🔬 STATISTICAL SIGNIFICANCE:")
print(f"  • ANOVA F-statistic: {f_stat:.2f}")
print(f"  • p-value: {p_value:.6f}")
print(f"  • Result: {'✅ SIGNIFICANT (p < 0.05)' if p_value < 0.05 else '❌ NOT SIGNIFICANT (p >= 0.05)'}")

print(f"\n✅ HYPOTHESIS TEST:")
if beta_2 < 0 and r_squared > 0.4 and p_value < 0.05:
    print(f"  🎉 H₁ CONFIRMED: Unimodal relationship exists!")
    print(f"  • Bell curve with peak at δ* ≈ {delta_star:.2f}")
    print(f"  • Moderate model fit (R² = {r_squared:.3f})")
    print(f"  • Statistically significant (p < 0.05)")
    print(f"\n  INTERPRETATION:")
    print(f"  This validates Adamopoulos & Tuzhilin's (2014) theory in music domain.")
    print(f"  Serendipity exists at moderate distance - not too close, not too far.")
elif beta_2 > 0 and p_value < 0.05:
    print(f"  ⚠ H₁ NOT CONFIRMED: No unimodal relationship found")
    print(f"  • β₂ is POSITIVE ({beta_2:.3f}) → U-shaped or upward trend")
    print(f"  • Highest ratings at distance {bin_stats['avg_rating'].idxmax()}")
    print(f"  • R² = {r_squared:.3f} (weak-moderate fit)")
    print(f"\n  INTERPRETATION:")
    print(f"  Unlike books/movies (Adamopoulos 2014), AMBAR users show highest")
    print(f"  engagement with tracks VERY DISTANT from expected set.")
    print(f"  This suggests music discovery benefits from maximizing unexpectedness")
    print(f"  rather than targeting an optimal zone.")
    print(f"\n  RESEARCH PIVOT:")
    print(f"  → RQ2: Can PSO-MOO maximize distance while maintaining relevance?")
    print(f"  → Compare distance-maximizing vs. relevance-maximizing approaches")
else:
    print(f"  ⚠ INCONCLUSIVE: Hypothesis not clearly supported or rejected")
    if beta_2 >= 0:
        print(f"    → No bell curve (β₂ = {beta_2:.3f} should be negative)")
    if r_squared <= 0.4:
        print(f"    → Weak model fit (R² = {r_squared:.3f} should be > 0.4)")
    if p_value >= 0.05:
        print(f"    → Not statistically significant (p = {p_value:.3f})")
    
    print(f"\n  POSSIBLE CAUSES:")
    print(f"  • Distance metric may not capture music similarity well")
    print(f"  • E_u definition may be too broad or too narrow")
    print(f"  • AMBAR's implicit ratings may not reflect true preference")
    print(f"\n  NEXT STEPS:")
    print(f"  • Try different distance metrics (cosine, category-based)")
    print(f"  • Try different E_u definitions (minimal, genre-based)")
    print(f"  • Consider using explicit feedback dataset")

print("=" * 80)

print("\n[11/11] Saving results...")

# Save detailed results
df_results.to_csv('experiment1_detailed_results.csv', index=False)
print("  ✓ Saved: experiment1_detailed_results.csv")

# Save bin statistics
bin_stats.to_csv('experiment1_bin_statistics.csv')
print("  ✓ Saved: experiment1_bin_statistics.csv")

# Save regression results
regression_results = pd.DataFrame({
    'coefficient': ['β₀ (intercept)', 'β₁ (linear)', 'β₂ (quadratic)'],
    'value': [beta_0, beta_1, beta_2],
    'interpretation': [
        'Baseline rating',
        'Linear effect of distance',
        'Curvature (negative = bell curve)'
    ]
})
regression_results.to_csv('experiment1_regression.csv', index=False)
print("  ✓ Saved: experiment1_regression.csv")

# Save summary report
with open('experiment1_summary.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("EXPERIMENT 1: VALIDATING THE UNIMODAL RELATIONSHIP\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Sample size: {len(sample_users)} users, {len(df_results)} test tracks\n\n")
    f.write(f"Quadratic model: Rating = {beta_0:.2f} + {beta_1:.2f}×d + {beta_2:.2f}×d²\n")
    f.write(f"R² = {r_squared:.3f}\n\n")
    if delta_star:
        f.write(f"Optimal distance δ* = {delta_star:.3f}\n")
        f.write(f"Peak rating = {rating_at_peak:.2f}\n\n")
    f.write(f"ANOVA: F = {f_stat:.2f}, p = {p_value:.6f}\n\n")
    f.write("Result: " + ("✅ H₁ CONFIRMED\n" if (beta_2 < 0 and r_squared > 0.7 and p_value < 0.05) else "⚠ Hypothesis not fully confirmed\n"))

print("  ✓ Saved: experiment1_summary.txt")

print("\n" + "=" * 80)
print("✅ EXPERIMENT 1 COMPLETE!")
print("=" * 80)