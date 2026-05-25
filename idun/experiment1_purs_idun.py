#!/usr/bin/env python3
"""
experiment1_purs_idun.py
========================
Experiment 1 — Unimodal Serendipity Hypothesis, PURS distance variant.

Identical structure to experiment1_idun.py but replaces the Jaccard style-tag
distance with the PURS-inspired latent-space distance:

  unexpectedness(item) = weighted average Euclidean distance from the item's
                         SVD qi embedding to the user's k-means interest cluster
                         centroids, min-max normalised per user.

This tests whether the unimodal (inverted-U) relationship between
unexpectedness and rating is stronger, weaker, or structurally different
when unexpectedness is defined in embedding space rather than tag space.

Key differences from experiment1_idun.py:
  - Trains SVD internally to obtain qi item embeddings
  - Builds per-user interest clusters via k-means on rated item embeddings
  - Distance = weighted Euclidean to cluster centroids (PURS Eq. 3)
  - No style-tag parsing needed
  - Distance computation is O(n_test_tracks × k) per user — much faster
    than the Jaccard sampling loop, so 5 000 users runs locally too

Usage:
    # Local (fast, no IDUN needed):
    python experiment1_purs_idun.py --n_users 5000 --workers 8

    # IDUN:
    python experiment1_purs_idun.py --n_users 5000 --workers 32

    # Quick sanity check:
    python experiment1_purs_idun.py --n_users 300 --workers 4

Arguments:
    --n_users       Number of users to sample          (default: 5000)
    --workers       Parallel worker processes          (default: 8)
    --data_dir      Path to AMBAR CSV directory        (default: ../AMBAR)
    --out_dir       Output directory                   (default: idun_results_purs)
    --n_clusters    k-means clusters per user (PURS)   (default: 3)
    --seed          Random seed                        (default: 42)

Output (in --out_dir):
    exp1_purs_detailed_results.csv
    exp1_purs_bin_statistics.csv
    exp1_purs_regression.csv
    exp1_purs_summary.txt
    exp1_purs_unimodal_relationship.png
"""

import argparse
import os
import warnings
from datetime import datetime
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
from surprise import SVD, Dataset, Reader

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Global state shared across worker processes
# ---------------------------------------------------------------------------
_SVD_FACTORS = None   # dict with pu, qi, bu, bi, global_mean, uid_map, iid_map
_TEST_DF     = None   # test dataframe filtered to selected users


def _init_worker(svd_factors, test_df):
    global _SVD_FACTORS, _TEST_DF
    _SVD_FACTORS = svd_factors
    _TEST_DF     = test_df


# ---------------------------------------------------------------------------
# PURS latent-space distance for a single user
# ---------------------------------------------------------------------------
def _purs_distance(track_ids, cluster_centroids, cluster_weights):
    """
    Compute PURS-style unexpectedness for a list of track IDs.

    For each track:
      - If its qi embedding is known: weighted average Euclidean distance
        to the k cluster centroids, min-max normalised over the batch.
      - If its embedding is unknown (cold item): distance = 1.0 (max).

    Parameters
    ----------
    track_ids        : list of track IDs
    cluster_centroids: np.array (k, n_factors) or None
    cluster_weights  : np.array (k,)           or None

    Returns
    -------
    distances : np.array (len(track_ids),) in [0, 1]
    """
    iid_map = _SVD_FACTORS["iid_map"]
    qi      = _SVD_FACTORS["qi"]

    distances  = np.ones(len(track_ids), dtype=np.float64)   # default: max unexpectedness

    if cluster_centroids is None or len(track_ids) == 0:
        return distances

    known_mask  = np.array([tid in iid_map for tid in track_ids])
    known_items = [tid for tid in track_ids if tid in iid_map]

    if len(known_items) == 0:
        return distances

    i_idx  = np.array([iid_map[tid] for tid in known_items])
    qi_mat = qi[i_idx]                                        # (n_known, n_factors)

    # Weighted average Euclidean distance to each cluster centroid
    weighted_dist = np.zeros(len(known_items), dtype=np.float64)
    for centroid, weight in zip(cluster_centroids, cluster_weights):
        d = np.linalg.norm(qi_mat - centroid, axis=1)         # (n_known,)
        weighted_dist += weight * d

    # Min-max normalise so distances are in [0, 1]
    d_min, d_max = weighted_dist.min(), weighted_dist.max()
    if d_max > d_min:
        weighted_dist = (weighted_dist - d_min) / (d_max - d_min)
    else:
        weighted_dist = np.zeros_like(weighted_dist)

    distances[known_mask] = weighted_dist
    return distances


# ---------------------------------------------------------------------------
# Per-user worker
# ---------------------------------------------------------------------------
def process_user(args):
    """
    For one user compute the PURS latent-space distance for every test track,
    paired with the user's actual rating.

    Returns list of dicts: {user_id, track_id, distance, rating}
    """
    user_id, cluster_centroids, cluster_weights = args

    user_test = _TEST_DF[_TEST_DF["user_id"] == user_id]
    if user_test.empty:
        return []

    track_ids = list(user_test["track_id"])
    distances = _purs_distance(track_ids, cluster_centroids, cluster_weights)

    rows = []
    for (_, row), dist in zip(user_test.iterrows(), distances):
        rows.append({
            "user_id":  user_id,
            "track_id": row["track_id"],
            "distance": float(dist),
            "rating":   float(row["rating"]),
        })
    return rows


# ---------------------------------------------------------------------------
# Build PURS interest clusters for one user
# ---------------------------------------------------------------------------
def build_user_clusters(E_u_ids, svd_factors, n_clusters):
    """
    Cluster the SVD qi embeddings of a user's rated items.

    Returns (centroids, weights) or (None, None) if too few known items.
    """
    iid_map = svd_factors["iid_map"]
    qi      = svd_factors["qi"]

    known_ids = [iid for iid in E_u_ids if iid in iid_map]
    if len(known_ids) < 2:
        return None, None

    embeddings = qi[[iid_map[iid] for iid in known_ids]]     # (n_known, n_factors)
    k          = min(n_clusters, len(known_ids))

    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
    km.fit(embeddings)

    centroids = km.cluster_centers_
    counts    = np.bincount(km.labels_, minlength=k).astype(float)
    weights   = counts / counts.sum()

    return centroids, weights


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def quadratic_regression(midpoints, avg_ratings):
    """Fit Rating = b0 + b1*d + b2*d² and return (b0, b1, b2, R²)."""
    X = np.column_stack([np.ones(len(midpoints)), midpoints, midpoints**2])
    coeffs, _, _, _ = np.linalg.lstsq(X, avg_ratings, rcond=None)
    b0, b1, b2 = coeffs
    y_pred = b0 + b1 * midpoints + b2 * midpoints**2
    ss_res = np.sum((avg_ratings - y_pred)**2)
    ss_tot = np.sum((avg_ratings - avg_ratings.mean())**2)
    r_sq   = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return b0, b1, b2, r_sq


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Exp 1 — Unimodal hypothesis with PURS latent-space distance"
    )
    parser.add_argument("--n_users",    type=int, default=5000,           help="Users to sample")
    parser.add_argument("--workers",    type=int, default=8,              help="Parallel workers")
    parser.add_argument("--data_dir",   type=str, default="../AMBAR",     help="AMBAR CSV directory")
    parser.add_argument("--out_dir",    type=str, default="idun_results_purs")
    parser.add_argument("--n_clusters", type=int, default=3,              help="k-means clusters per user")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    start_time = datetime.now()

    print("=" * 80)
    print("EXPERIMENT 1 (PURS) — UNIMODAL SERENDIPITY HYPOTHESIS")
    print("  Distance metric: latent-space Euclidean to k-means cluster centroids")
    print("=" * 80)
    print(f"Started    : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config     : {args.n_users} users | {args.workers} workers | "
          f"k={args.n_clusters} clusters | seed={args.seed}")
    print(f"Output dir : {args.out_dir}")
    print()

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    print("[1/7] Loading AMBAR dataset...")
    ratings_df = pd.read_csv(os.path.join(args.data_dir, "ratings_info.csv"))
    tracks_df  = pd.read_csv(os.path.join(args.data_dir, "tracks_info.csv"))
    print(f"  Ratings : {len(ratings_df):,}  |  Tracks : {len(tracks_df):,}  "
          f"|  Users : {ratings_df['user_id'].nunique():,}")

    # Binary rating conversion (same as experiment1_idun.py)
    ratings_df["liked"]  = (ratings_df["rating"] >= 3).astype(int)
    ratings_df["rating"] = ratings_df["liked"].apply(lambda x: 5.0 if x == 1 else 1.0)
    print(f"  Ratings binarised → 1 (disliked) / 5 (liked)")

    # -----------------------------------------------------------------------
    # 2. Train/test split (80/20 per user)
    # -----------------------------------------------------------------------
    print("\n[2/7] Train/test split (80/20 per user)...")
    has_ts = "timestamp" in ratings_df.columns
    train_parts, test_parts = [], []
    for uid, grp in ratings_df.groupby("user_id"):
        grp = grp.sort_values("timestamp") if has_ts else grp.sample(frac=1, random_state=args.seed)
        n   = max(1, int(0.8 * len(grp)))
        train_parts.append(grp.iloc[:n])
        test_parts.append(grp.iloc[n:])
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df  = pd.concat(test_parts,  ignore_index=True)
    print(f"  Train : {len(train_df):,}  |  Test : {len(test_df):,}")

    # -----------------------------------------------------------------------
    # 3. Select users
    # -----------------------------------------------------------------------
    print("\n[3/7] Selecting users...")
    train_counts = train_df.groupby("user_id").size()
    test_counts  = test_df.groupby("user_id").size()
    eligible = train_counts[
        (train_counts >= 30) &
        (train_counts.index.isin(test_counts[test_counts >= 10].index))
    ].index
    print(f"  Eligible (≥30 train, ≥10 test) : {len(eligible):,}")

    n_select = min(args.n_users, len(eligible))
    selected = np.random.choice(eligible, size=n_select, replace=False)
    print(f"  Selected : {n_select:,}")

    test_sample = test_df[test_df["user_id"].isin(selected)]

    # -----------------------------------------------------------------------
    # 4. Train SVD to obtain item embeddings (qi)
    # -----------------------------------------------------------------------
    print("\n[4/7] Training SVD (50 factors, 20 epochs)...")
    t0     = datetime.now()
    reader = Reader(rating_scale=(1, 5))
    # Train on all users' training data (not just selected) for richer embeddings
    surprise_data = Dataset.load_from_df(
        train_df[["user_id", "track_id", "rating"]], reader
    )
    trainset = surprise_data.build_full_trainset()
    svd = SVD(n_factors=50, n_epochs=20, random_state=args.seed, verbose=False)
    svd.fit(trainset)
    print(f"  SVD trained in {(datetime.now()-t0).total_seconds():.1f}s")

    svd_factors = {
        "pu":          svd.pu,
        "qi":          svd.qi,
        "bu":          svd.bu,
        "bi":          svd.bi,
        "global_mean": trainset.global_mean,
        "uid_map":     {raw: inner for raw, inner in trainset._raw2inner_id_users.items()},
        "iid_map":     {raw: inner for raw, inner in trainset._raw2inner_id_items.items()},
    }
    n_known_items = len(svd_factors["iid_map"])
    print(f"  Known item embeddings : {n_known_items:,} / {len(tracks_df):,} tracks")

    # -----------------------------------------------------------------------
    # 5. Build PURS interest clusters per user
    # -----------------------------------------------------------------------
    print(f"\n[5/7] Building k={args.n_clusters} interest clusters per user...")
    t0         = datetime.now()
    user_args  = []
    n_fallback = 0
    for uid in selected:
        E_u_ids = set(train_df.loc[train_df["user_id"] == uid, "track_id"])
        centroids, weights = build_user_clusters(E_u_ids, svd_factors, args.n_clusters)
        if centroids is None:
            n_fallback += 1
        user_args.append((uid, centroids, weights))
    print(f"  Clusters built in {(datetime.now()-t0).total_seconds():.1f}s  "
          f"({n_fallback} users without known embeddings → distance=1.0)")

    # -----------------------------------------------------------------------
    # 6. Parallel PURS distance computation
    # -----------------------------------------------------------------------
    print(f"\n[6/7] Computing PURS latent distances ({args.workers} workers)...")
    all_rows   = []
    phase_start = datetime.now()

    with Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(svd_factors, test_sample),
    ) as pool:
        for done, rows in enumerate(
            pool.imap_unordered(process_user, user_args, chunksize=50), start=1
        ):
            all_rows.extend(rows)
            if done % 500 == 0 or done == len(user_args):
                elapsed   = (datetime.now() - phase_start).total_seconds()
                rate      = done / elapsed if elapsed > 0 else 0
                remaining = (len(user_args) - done) / rate if rate > 0 else 0
                print(f"  {done:5d}/{len(user_args)} users | "
                      f"{rate:.1f} users/s | ~{remaining/60:.1f} min remaining")

    df = pd.DataFrame(all_rows)
    print(f"\n  Computed {len(df):,} (user, track) distance-rating pairs")
    print(f"  Distance range : [{df['distance'].min():.3f}, {df['distance'].max():.3f}]")
    print(f"  Mean distance  : {df['distance'].mean():.3f}")

    # -----------------------------------------------------------------------
    # 7. Analysis: binning → quadratic regression → ANOVA → plots → save
    # -----------------------------------------------------------------------
    print("\n[7/7] Running analysis...")

    bins       = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bin_labels = [f"[{bins[i]:.1f}-{bins[i+1]:.1f})" for i in range(len(bins)-1)]

    df["distance_bin"] = pd.cut(df["distance"], bins=bins, labels=bin_labels, include_lowest=True)

    bin_stats = df.groupby("distance_bin", observed=True).agg(
        avg_rating  =("rating",  "mean"),
        std_rating  =("rating",  "std"),
        sample_size =("rating",  "count"),
        num_users   =("user_id", "nunique"),
    ).round(4)

    print("\nDistance bin statistics:")
    print(bin_stats.to_string())

    # Only bins with data
    valid      = bin_stats[bin_stats["sample_size"] > 0].copy()
    midpoints  = np.array([bins[i] + 0.05
                           for i, lbl in enumerate(bin_labels) if lbl in valid.index])
    avg_ratings = valid["avg_rating"].values

    b0, b1, b2, r_sq = quadratic_regression(midpoints, avg_ratings)
    delta_star = -b1 / (2 * b2) if b2 < 0 else None

    print(f"\nQuadratic regression : Rating = {b0:.3f} + {b1:.3f}×d + {b2:.3f}×d²")
    print(f"R²                   : {r_sq:.4f}")
    if delta_star is not None:
        peak_in_range = 0.0 <= delta_star <= 1.0
        print(f"δ*                   : {delta_star:.3f}  "
              f"({'within [0,1]' if peak_in_range else 'outside [0,1]'})")
    else:
        print("β₂ > 0 — no inverted-U (hypothesis not supported by shape)")

    # ANOVA across bins
    groups = [df.loc[df["distance_bin"] == lbl, "rating"].dropna().values
              for lbl in bin_labels]
    groups = [g for g in groups if len(g) > 0]
    f_stat, p_val = f_oneway(*groups)
    print(f"ANOVA                : F={f_stat:.2f}, p={p_val:.6f} "
          f"({'significant' if p_val < 0.05 else 'not significant'})")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    unimodal_supported = (b2 < 0 and r_sq > 0.4 and p_val < 0.05)
    verdict = "SUPPORTED" if unimodal_supported else "NOT SUPPORTED"
    print(f"\nUnimodal hypothesis  : {verdict}")
    if not unimodal_supported:
        reasons = []
        if b2 >= 0:
            reasons.append(f"β₂={b2:.3f} (positive, need negative)")
        if r_sq <= 0.4:
            reasons.append(f"R²={r_sq:.3f} (need >0.40)")
        if p_val >= 0.05:
            reasons.append(f"p={p_val:.4f} (need <0.05)")
        print(f"  Reasons: {'; '.join(reasons)}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Experiment 1 (PURS): Unimodal Hypothesis — Latent-Space Distance\n"
        f"{n_select:,} users | k={args.n_clusters} clusters | R²={r_sq:.3f}",
        fontsize=14, fontweight="bold",
    )

    # Panel 1: distance vs avg rating with quadratic fit
    ax1 = axes[0, 0]
    ax1.errorbar(midpoints, avg_ratings, yerr=valid["std_rating"].values,
                 fmt="o-", linewidth=2.5, markersize=9, capsize=4,
                 color="#2E86AB", label="Observed (mean ± std)")
    x_s = np.linspace(midpoints.min(), midpoints.max(), 200)
    ax1.plot(x_s, b0 + b1*x_s + b2*x_s**2, "--", color="#A23B72", linewidth=2,
             label=f"Quadratic fit (R²={r_sq:.3f})")
    if delta_star is not None and 0.0 <= delta_star <= 1.0:
        ax1.axvline(delta_star, color="red", linestyle=":", linewidth=2,
                    label=f"δ*={delta_star:.3f}")
    ax1.set_xlabel("PURS Latent-Space Distance from E_u", fontweight="bold")
    ax1.set_ylabel("Avg Rating", fontweight="bold")
    ax1.set_title("Distance vs Rating (PURS)", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel 2: bar chart of avg rating per bin
    ax2 = axes[0, 1]
    colors = ["#1a9850" if r == avg_ratings.max() else "#4ECDC4" for r in avg_ratings]
    ax2.bar(range(len(avg_ratings)), avg_ratings, color=colors, edgecolor="black", alpha=0.8)
    ax2.set_xticks(range(len(avg_ratings)))
    ax2.set_xticklabels(list(valid.index), rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Avg Rating", fontweight="bold")
    ax2.set_title("Avg Rating per Distance Bin", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # Panel 3: sample size distribution
    ax3 = axes[1, 0]
    ax3.bar(range(len(valid)), valid["sample_size"].values,
            color="#F18F01", edgecolor="black", alpha=0.8)
    ax3.set_xticks(range(len(valid)))
    ax3.set_xticklabels(list(valid.index), rotation=45, ha="right", fontsize=8)
    ax3.set_ylabel("(user, track) pairs", fontweight="bold")
    ax3.set_title("Sample Size per Distance Bin", fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    # Panel 4: % liked (top 25%)
    threshold = np.percentile(df["rating"], 75)
    pct_liked = []
    for lbl in valid.index:
        bd = df[df["distance_bin"] == lbl]
        pct_liked.append((bd["rating"] >= threshold).mean() * 100)
    ax4 = axes[1, 1]
    ax4.plot(midpoints, pct_liked, "o-", linewidth=2.5, markersize=9, color="#E84855")
    if delta_star is not None and 0.0 <= delta_star <= 1.0:
        ax4.axvline(delta_star, color="red", linestyle=":", linewidth=2, alpha=0.7)
    ax4.set_xlabel("PURS Latent-Space Distance from E_u", fontweight="bold")
    ax4.set_ylabel(f"% Rated ≥ {threshold:.1f} (top 25%)", fontweight="bold")
    ax4.set_title("User Satisfaction vs Distance (PURS)", fontweight="bold")
    ax4.set_ylim(0, 100)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "exp1_purs_unimodal_relationship.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved plot : {fig_path}")

    # ------------------------------------------------------------------
    # Save CSVs and summary
    # ------------------------------------------------------------------
    df.to_csv(os.path.join(args.out_dir, "exp1_purs_detailed_results.csv"), index=False)
    bin_stats.to_csv(os.path.join(args.out_dir, "exp1_purs_bin_statistics.csv"))

    reg_df = pd.DataFrame({
        "coefficient": ["β0 (intercept)", "β1 (linear)", "β2 (quadratic)", "R2", "F_stat", "p_value"],
        "value":       [b0, b1, b2, r_sq, f_stat, p_val],
    })
    reg_df.to_csv(os.path.join(args.out_dir, "exp1_purs_regression.csv"), index=False)

    summary_path = os.path.join(args.out_dir, "exp1_purs_summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("EXPERIMENT 1 (PURS) — UNIMODAL HYPOTHESIS\n")
        fh.write(f"Distance metric : PURS latent-space Euclidean (k={args.n_clusters} clusters)\n")
        fh.write(f"Users : {n_select:,}  |  Pairs : {len(df):,}\n")
        fh.write(f"Rating = {b0:.3f} + {b1:.3f}×d + {b2:.3f}×d²\n")
        fh.write(f"R² = {r_sq:.4f}\n")
        fh.write(f"δ* = {delta_star:.3f}\n" if delta_star is not None else "No peak (β₂ > 0)\n")
        fh.write(f"ANOVA: F={f_stat:.2f}, p={p_val:.6f}\n")
        fh.write(f"Unimodal hypothesis: {verdict}\n")
    print(f"  Saved summary: {summary_path}")

    total_min = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 1 (PURS) COMPLETE — {total_min:.1f} minutes")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
