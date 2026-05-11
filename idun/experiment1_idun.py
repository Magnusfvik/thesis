#!/usr/bin/env python3
"""
experiment1_idun.py
===================
Parallelised Experiment 1 for IDUN: Validating the Unimodal Serendipity Hypothesis.

Tests whether serendipity (measured as user rating) follows an inverted-U
relationship with unexpectedness (Jaccard distance from E_u), or whether
the relationship is flat — requiring multi-objective optimisation.

Parallelises over users: each worker processes one user's test tracks.

Usage:
    python experiment1_idun.py \\
        --n_users 5000 \\
        --workers 32 \\
        --data_dir ../AMBAR \\
        --out_dir idun_results

Output (in --out_dir):
    exp1_detailed_results.csv
    exp1_bin_statistics.csv
    exp1_regression.csv
    exp1_summary.txt
    exp1_unimodal_relationship.png
"""

import argparse
import os
import warnings
from datetime import datetime
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")   # no display needed on IDUN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Global shared state (set via Pool initializer)
# ---------------------------------------------------------------------------
_TRACK_STYLES = None   # dict: track_id -> frozenset of style tags
_TEST_DF      = None   # full test dataframe (filtered to sample users)


def _init_worker(track_styles, test_df):
    global _TRACK_STYLES, _TEST_DF
    _TRACK_STYLES = track_styles
    _TEST_DF      = test_df


# ---------------------------------------------------------------------------
# Jaccard distance between two sets
# ---------------------------------------------------------------------------
def _jaccard(a, b):
    if not a or not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return 1.0 - inter / union if union else 1.0


# ---------------------------------------------------------------------------
# Per-user worker: compute (distance, rating) for each test track
# ---------------------------------------------------------------------------
def process_user(args):
    """
    For one user: compute the Jaccard distance of every test track to E_u,
    using the average distance over a sample of E_u tracks.
    Returns list of dicts with keys: user_id, track_id, distance, rating.
    """
    user_id, E_u_ids, E_u_styles = args
    user_test = _TEST_DF[_TEST_DF["user_id"] == user_id]

    E_u_list = list(E_u_ids)
    SAMPLE_SIZE = 50
    if len(E_u_list) > SAMPLE_SIZE:
        rng = np.random.default_rng(int(user_id) if str(user_id).isdigit() else 0)
        E_u_sample = rng.choice(E_u_list, size=SAMPLE_SIZE, replace=False)
    else:
        E_u_sample = E_u_list

    rows = []
    for _, row in user_test.iterrows():
        tid = row["track_id"]
        track_styles = _TRACK_STYLES.get(tid)
        if track_styles is None:
            continue

        dists = []
        for e_tid in E_u_sample:
            e_styles = _TRACK_STYLES.get(e_tid)
            if e_styles is not None:
                dists.append(_jaccard(track_styles, e_styles))

        if not dists:
            continue

        rows.append({
            "user_id":  user_id,
            "track_id": tid,
            "distance": float(np.mean(dists)),
            "rating":   float(row["rating"]),
        })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Exp 1 — Unimodal hypothesis (IDUN)")
    parser.add_argument("--n_users",  type=int, default=5000,   help="Number of users")
    parser.add_argument("--workers",  type=int, default=32,     help="Parallel workers")
    parser.add_argument("--data_dir", type=str, default="../AMBAR")
    parser.add_argument("--out_dir",  type=str, default="idun_results")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    start_time = datetime.now()

    print("=" * 80)
    print("EXPERIMENT 1 (IDUN) — UNIMODAL SERENDIPITY HYPOTHESIS")
    print("=" * 80)
    print(f"Started : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config  : {args.n_users} users | {args.workers} workers | seed={args.seed}")
    print()

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    print("[1/7] Loading AMBAR dataset...")
    ratings_df = pd.read_csv(os.path.join(args.data_dir, "ratings_info.csv"))
    tracks_df  = pd.read_csv(os.path.join(args.data_dir, "tracks_info.csv"))
    print(f"  Ratings: {len(ratings_df):,}  |  Tracks: {len(tracks_df):,}")

    # Binary rating conversion (same as original experiment1.py)
    ratings_df["liked"]  = (ratings_df["rating"] >= 3).astype(int)
    ratings_df["rating"] = ratings_df["liked"].apply(lambda x: 5 if x == 1 else 1)

    # -----------------------------------------------------------------------
    # 2. Parse styles (experiment 1 uses 'styles', not 'category_styles')
    # -----------------------------------------------------------------------
    print("\n[2/7] Parsing track styles...")

    def parse_styles(val):
        if pd.isna(val):
            return frozenset()
        if isinstance(val, list):
            return frozenset(str(s).strip() for s in val)
        return frozenset(s.strip() for s in str(val).split("|") if s.strip())

    tracks_df["_styles"] = tracks_df["styles"].apply(parse_styles)
    track_styles = dict(zip(tracks_df["track_id"], tracks_df["_styles"]))
    print(f"  Style metadata for {len(track_styles):,} tracks")

    # Also pre-compute category_styles for E_u (artist-based expansion not used here;
    # we use rated tracks only to keep consistent with generate_complete approach)
    tracks_df["_cats"] = tracks_df["category_styles"].apply(parse_styles)
    track_cats = dict(zip(tracks_df["track_id"], tracks_df["_cats"]))

    # -----------------------------------------------------------------------
    # 3. Train/test split
    # -----------------------------------------------------------------------
    print("\n[3/7] Train/test split (80/20)...")
    has_ts = "timestamp" in ratings_df.columns
    train_parts, test_parts = [], []
    for uid, grp in ratings_df.groupby("user_id"):
        grp = grp.sort_values("timestamp") if has_ts else grp.sample(frac=1, random_state=args.seed)
        n = max(1, int(0.8 * len(grp)))
        train_parts.append(grp.iloc[:n])
        test_parts.append(grp.iloc[n:])
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df  = pd.concat(test_parts,  ignore_index=True)
    print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    # -----------------------------------------------------------------------
    # 4. Select users
    # -----------------------------------------------------------------------
    print("\n[4/7] Selecting users...")
    train_counts = train_df.groupby("user_id").size()
    test_counts  = test_df.groupby("user_id").size()
    eligible = train_counts[(train_counts >= 30)].index
    eligible = eligible[eligible.isin(test_counts[test_counts >= 10].index)]
    print(f"  Eligible (>=30 train, >=10 test): {len(eligible):,}")

    n_select = min(args.n_users, len(eligible))
    selected = np.random.choice(eligible, size=n_select, replace=False)
    print(f"  Selected: {n_select:,}")

    # Filter test set to selected users only
    test_sample = test_df[test_df["user_id"].isin(selected)]

    # -----------------------------------------------------------------------
    # 5. Build E_u for each user
    #    E_u = rated track IDs (used to sample from for distance)
    #    E_u also includes tracks by the same artists (moderate definition)
    # -----------------------------------------------------------------------
    print("\n[5/7] Building E_u for each user...")
    artist_col = "artist_id" if "artist_id" in tracks_df.columns else None
    artist_tracks = {}  # artist_id -> set of track_ids
    if artist_col:
        for _, row in tracks_df.iterrows():
            artist_tracks.setdefault(row[artist_col], set()).add(row["track_id"])

    user_args = []
    for uid in selected:
        rated = set(train_df.loc[train_df["user_id"] == uid, "track_id"])
        # Expand to artist tracks (moderate E_u)
        E_u_ids = set(rated)
        if artist_col:
            rated_info = tracks_df[tracks_df["track_id"].isin(rated)]
            known_artists = set(rated_info[artist_col])
            for a in known_artists:
                E_u_ids.update(artist_tracks.get(a, set()))

        # E_u styles = union of styles across E_u tracks
        E_u_styles = frozenset().union(*(track_styles.get(tid, frozenset()) for tid in E_u_ids))
        user_args.append((uid, E_u_ids, E_u_styles))

    print(f"  E_u built for {len(user_args):,} users")

    # -----------------------------------------------------------------------
    # 6. Parallel distance computation
    # -----------------------------------------------------------------------
    print(f"\n[6/7] Computing distances ({args.workers} workers)...")
    all_rows = []
    phase_start = datetime.now()

    with Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(track_styles, test_sample),
    ) as pool:
        for done, rows in enumerate(
            pool.imap_unordered(process_user, user_args, chunksize=20), start=1
        ):
            all_rows.extend(rows)
            if done % 500 == 0 or done == len(user_args):
                elapsed   = (datetime.now() - phase_start).total_seconds()
                rate      = done / elapsed
                remaining = (len(user_args) - done) / rate if rate > 0 else 0
                print(f"  {done:5d}/{len(user_args)} users | "
                      f"{rate:.1f} users/s | ~{remaining/60:.1f} min remaining")

    df = pd.DataFrame(all_rows)
    print(f"\n  Computed distances for {len(df):,} (user, track) pairs")
    print(f"  Distance range: [{df['distance'].min():.3f}, {df['distance'].max():.3f}]")

    # -----------------------------------------------------------------------
    # 7. Analysis: binning, regression, ANOVA
    # -----------------------------------------------------------------------
    print("\n[7/7] Running analysis...")

    bins       = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bin_labels = [f"[{bins[i]:.1f}-{bins[i+1]:.1f})" for i in range(len(bins)-1)]

    df["distance_bin"] = pd.cut(df["distance"], bins=bins, labels=bin_labels, include_lowest=True)

    bin_stats = df.groupby("distance_bin", observed=True).agg(
        avg_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        sample_size=("rating", "count"),
        num_users=("user_id", "nunique"),
    ).round(4)

    print("\nDistance bin statistics:")
    print(bin_stats.to_string())

    # Quadratic regression
    valid = bin_stats[bin_stats["sample_size"] > 0].copy()
    midpoints = np.array([bins[i] + 0.05 for i, lbl in enumerate(bin_labels) if lbl in valid.index])
    avg_ratings = valid["avg_rating"].values

    X = np.column_stack([np.ones(len(midpoints)), midpoints, midpoints**2])
    coeffs, _, _, _ = np.linalg.lstsq(X, avg_ratings, rcond=None)
    b0, b1, b2 = coeffs

    y_pred   = b0 + b1 * midpoints + b2 * midpoints**2
    ss_res   = np.sum((avg_ratings - y_pred)**2)
    ss_tot   = np.sum((avg_ratings - avg_ratings.mean())**2)
    r_sq     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    delta_star = -b1 / (2 * b2) if b2 < 0 else None

    print(f"\nQuadratic regression: Rating = {b0:.3f} + {b1:.3f}×d + {b2:.3f}×d²")
    print(f"R² = {r_sq:.4f}")
    if delta_star is not None:
        print(f"δ* = {delta_star:.3f}  (peak within [0,1]: {0 <= delta_star <= 1})")
    else:
        print("β₂ > 0 — no inverted-U (hypothesis not supported)")

    # ANOVA
    groups = [df.loc[df["distance_bin"] == lbl, "rating"].dropna().values for lbl in bin_labels]
    groups = [g for g in groups if len(g) > 0]
    f_stat, p_val = f_oneway(*groups)
    print(f"ANOVA: F={f_stat:.2f}, p={p_val:.6f} ({'significant' if p_val < 0.05 else 'not significant'})")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Experiment 1: Unimodal Hypothesis — {n_select:,} Users (IDUN)",
        fontsize=15, fontweight="bold",
    )

    # Panel 1: distance vs avg rating + fit
    ax1 = axes[0, 0]
    ax1.errorbar(midpoints, avg_ratings, yerr=valid["std_rating"].values,
                 fmt="o-", linewidth=2.5, markersize=9, capsize=4, color="#2E86AB")
    x_s = np.linspace(midpoints.min(), midpoints.max(), 200)
    ax1.plot(x_s, b0 + b1*x_s + b2*x_s**2, "--", color="#A23B72", linewidth=2,
             label=f"Quadratic fit (R²={r_sq:.3f})")
    if delta_star is not None and 0 <= delta_star <= 1:
        ax1.axvline(delta_star, color="red", linestyle=":", linewidth=2,
                    label=f"δ*={delta_star:.3f}")
    ax1.set_xlabel("Distance from E_u", fontweight="bold")
    ax1.set_ylabel("Avg Rating", fontweight="bold")
    ax1.set_title("Distance vs Rating", fontweight="bold")
    ax1.legend(); ax1.grid(alpha=0.3)

    # Panel 2: bar chart of avg rating per bin
    ax2 = axes[0, 1]
    ax2.bar(range(len(avg_ratings)), avg_ratings, color="#4ECDC4", edgecolor="black", alpha=0.75)
    ax2.set_xticks(range(len(avg_ratings)))
    ax2.set_xticklabels(list(valid.index), rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Avg Rating", fontweight="bold")
    ax2.set_title("Avg Rating per Distance Bin", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # Panel 3: sample sizes
    ax3 = axes[1, 0]
    ax3.bar(range(len(valid)), valid["sample_size"].values, color="#F18F01", edgecolor="black", alpha=0.75)
    ax3.set_xticks(range(len(valid)))
    ax3.set_xticklabels(list(valid.index), rotation=45, ha="right", fontsize=8)
    ax3.set_ylabel("Number of (user, track) pairs", fontweight="bold")
    ax3.set_title("Sample Size per Bin", fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    # Panel 4: % liked (top 25% rating)
    threshold = np.percentile(df["rating"], 75)
    pct_liked = []
    for lbl in valid.index:
        bd = df[df["distance_bin"] == lbl]
        pct_liked.append((bd["rating"] >= threshold).mean() * 100)
    ax4 = axes[1, 1]
    ax4.plot(midpoints, pct_liked, "o-", linewidth=2.5, markersize=9, color="#E84855")
    ax4.set_xlabel("Distance from E_u", fontweight="bold")
    ax4.set_ylabel(f"% Rated ≥ {threshold:.1f} (top 25%)", fontweight="bold")
    ax4.set_title("User Satisfaction vs Distance", fontweight="bold")
    ax4.set_ylim(0, 100); ax4.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "exp1_unimodal_relationship.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {fig_path}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    df.to_csv(os.path.join(args.out_dir, "exp1_detailed_results.csv"), index=False)
    bin_stats.to_csv(os.path.join(args.out_dir, "exp1_bin_statistics.csv"))

    reg_df = pd.DataFrame({
        "coefficient": ["β0 (intercept)", "β1 (linear)", "β2 (quadratic)", "R2", "F_stat", "p_value"],
        "value":       [b0, b1, b2, r_sq, f_stat, p_val],
    })
    reg_df.to_csv(os.path.join(args.out_dir, "exp1_regression.csv"), index=False)

    summary_path = os.path.join(args.out_dir, "exp1_summary.txt")
    with open(summary_path, "w") as f:
        f.write("EXPERIMENT 1 — UNIMODAL HYPOTHESIS (IDUN)\n")
        f.write(f"Users: {n_select:,}  |  Pairs: {len(df):,}\n")
        f.write(f"Rating = {b0:.3f} + {b1:.3f}×d + {b2:.3f}×d²\n")
        f.write(f"R² = {r_sq:.4f}\n")
        f.write(f"δ* = {delta_star:.3f}\n" if delta_star else "No peak (β2 > 0)\n")
        f.write(f"ANOVA: F={f_stat:.2f}, p={p_val:.6f}\n")
        verdict = "SUPPORTED" if (b2 < 0 and r_sq > 0.4 and p_val < 0.05) else "NOT SUPPORTED"
        f.write(f"Unimodal hypothesis: {verdict}\n")
    print(f"  Saved: {summary_path}")

    total_min = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n{'='*80}")
    print(f"Completed in {total_min:.1f} minutes")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
