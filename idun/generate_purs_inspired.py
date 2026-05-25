#!/usr/bin/env python3
"""
generate_purs_inspired.py
=========================
PURS-inspired recommendation generation.

Key differences from generate_complete_idun.py:
  - Unexpectedness in LATENT SPACE (Euclidean distance on SVD qi embeddings)
    instead of Jaccard distance on style tags. Directly addresses the PURS
    critique: "the distance function is typically difficult to define in the
    feature space."
  - Multi-cluster user interest modeling (k-means on rated item embeddings)
    instead of one single E_u blob. A user who likes jazz AND metal gets two
    clusters, not one blended expected set that compresses distance ranges.
  - Unexpectedness = weighted average Euclidean distance from candidate
    embedding to each user interest cluster centroid (Equation 3 in PURS).
  - Min-max normalised per user so scores stay in [0, 1].

Scoring formula (same structure as original):
    score = α * unexp_latent + (1 - α) * cf_norm

Usage:
    python generate_purs_inspired.py \\
        --data_dir movielens/data \\
        --out_dir  movielens/results_purs \\
        --n_users  50 \\
        --workers  4 \\
        --n_clusters 3

Output:
    recommendations_fair_complete.pkl  (same format as generate_complete_idun.py)
    generation_info.pkl
"""

import argparse
import os
import pickle
import warnings
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from surprise import SVD, Dataset, Reader

warnings.filterwarnings("ignore")

ALPHA_VALUES = [0.0, 0.15, 0.25, 0.30, 0.35, 0.40, 0.50, 0.65, 0.75, 0.85, 1.0]

# ---------------------------------------------------------------------------
# Global state — set once per worker via Pool initializer
# ---------------------------------------------------------------------------
_SVD_FACTORS = None
_ALL_TRACKS  = None


def _init_worker(svd_factors, all_tracks):
    global _SVD_FACTORS, _ALL_TRACKS
    _SVD_FACTORS = svd_factors
    _ALL_TRACKS  = all_tracks


# ---------------------------------------------------------------------------
# Vectorised CF predictions (same as generate_complete_idun.py)
# ---------------------------------------------------------------------------
def _predict_batch(user_id, item_ids):
    f       = _SVD_FACTORS
    gm      = f["global_mean"]
    uid_map = f["uid_map"]
    iid_map = f["iid_map"]

    if user_id in uid_map:
        u_idx = uid_map[user_id]
        pu    = f["pu"][u_idx]
        bu    = f["bu"][u_idx]
    else:
        pu = np.zeros(f["pu"].shape[1])
        bu = 0.0

    known_mask  = np.array([iid in iid_map for iid in item_ids])
    known_items = [iid for iid in item_ids if iid in iid_map]
    preds       = np.full(len(item_ids), gm + bu, dtype=np.float32)

    if known_items:
        i_idx       = np.array([iid_map[iid] for iid in known_items])
        qi_mat      = f["qi"][i_idx]
        bi_arr      = f["bi"][i_idx]
        known_preds = gm + bu + bi_arr + qi_mat @ pu
        preds[known_mask] = known_preds

    return np.clip(preds, 1.0, 5.0)


# ---------------------------------------------------------------------------
# Build user interest clusters in SVD latent space (PURS Section 3.1)
# ---------------------------------------------------------------------------
def build_user_clusters(E_u_ids, svd_factors, n_clusters=3):
    """
    Cluster the user's rated item embeddings (qi vectors) using k-means.

    Returns:
        centroids : np.array (k, n_factors) — cluster centroids
        weights   : np.array (k,)           — cluster size weights (sum=1)

    Falls back to (None, None) if the user has too few known items.
    """
    iid_map = svd_factors["iid_map"]
    qi      = svd_factors["qi"]

    known_ids = [iid for iid in E_u_ids if iid in iid_map]
    if len(known_ids) < 2:
        return None, None

    embeddings = qi[[iid_map[iid] for iid in known_ids]]   # (n_known, n_factors)
    k          = min(n_clusters, len(known_ids))

    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
    km.fit(embeddings)

    centroids = km.cluster_centers_                          # (k, n_factors)
    counts    = np.bincount(km.labels_, minlength=k).astype(float)
    weights   = counts / counts.sum()

    return centroids, weights


# ---------------------------------------------------------------------------
# Per-user worker — PURS-inspired latent distance
# ---------------------------------------------------------------------------
def process_user(args):
    """
    Computes top-10 recommendations for one user across ALL alpha values
    using PURS-inspired latent-space unexpectedness.
    """
    user_id, E_u_ids, cluster_centroids, cluster_weights = args

    candidate_ids = list(_ALL_TRACKS - E_u_ids)
    if not candidate_ids:
        return user_id, {a: [] for a in ALPHA_VALUES}

    f       = _SVD_FACTORS
    iid_map = f["iid_map"]
    qi      = f["qi"]

    # --- Vectorised CF predictions ---
    cf_preds = _predict_batch(user_id, candidate_ids)
    cf_norm  = (cf_preds - 1.0) / 4.0

    # --- Latent-space unexpectedness (PURS Eq. 3) ---
    known_mask  = np.array([iid in iid_map for iid in candidate_ids])
    known_items = [iid for iid in candidate_ids if iid in iid_map]

    # Default: unknown items get maximum distance (1.0)
    distances = np.ones(len(candidate_ids), dtype=np.float32)

    if cluster_centroids is not None and len(known_items) > 0:
        i_idx  = np.array([iid_map[iid] for iid in known_items])
        qi_mat = qi[i_idx]                                   # (n_known, n_factors)

        # Weighted average Euclidean distance to cluster centroids
        weighted_dist = np.zeros(len(known_items), dtype=np.float64)
        for centroid, weight in zip(cluster_centroids, cluster_weights):
            d = np.linalg.norm(qi_mat - centroid, axis=1)   # (n_known,)
            weighted_dist += weight * d

        # Min-max normalise to [0, 1] within this user's candidate set
        d_min = weighted_dist.min()
        d_max = weighted_dist.max()
        if d_max > d_min:
            weighted_dist = (weighted_dist - d_min) / (d_max - d_min)
        else:
            weighted_dist = np.zeros_like(weighted_dist)

        distances[known_mask] = weighted_dist.astype(np.float32)

    # --- Score and rank for each alpha ---
    user_recs = {}
    for alpha in ALPHA_VALUES:
        scores    = alpha * distances + (1.0 - alpha) * cf_norm
        top10_idx = np.argpartition(scores, -10)[-10:]
        top10_idx = top10_idx[np.argsort(scores[top10_idx])[::-1]]

        recs = []
        for idx in top10_idx:
            recs.append({
                "track_id":       candidate_ids[idx],
                "distance":       float(distances[idx]),
                "cf_score":       float(cf_preds[idx]),
                "combined_score": float(scores[idx]),
            })
        user_recs[alpha] = recs

    return user_id, user_recs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PURS-inspired recommendation generation")
    parser.add_argument("--n_users",          type=int, default=50)
    parser.add_argument("--workers",          type=int, default=4)
    parser.add_argument("--data_dir",         type=str, default="movielens/data")
    parser.add_argument("--out_dir",          type=str, default="movielens/results_purs")
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--n_clusters",       type=int, default=3,
                        help="Number of interest clusters per user (PURS Section 3.1)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    start_time = datetime.now()

    print("=" * 80)
    print("PURS-INSPIRED RECOMMENDATION GENERATION")
    print("  Unexpectedness: latent-space Euclidean distance to k-means clusters")
    print("=" * 80)
    print(f"Started  : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config   : {args.n_users} users | {args.workers} workers | "
          f"{args.n_clusters} clusters | seed={args.seed}")
    print(f"Data     : {args.data_dir}")
    print(f"Output   : {args.out_dir}")
    print()

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    print("[1/5] Loading dataset...")
    ratings_df = pd.read_csv(os.path.join(args.data_dir, "ratings_info.csv"))
    tracks_df  = pd.read_csv(os.path.join(args.data_dir, "tracks_info.csv"))
    all_tracks_set = set(tracks_df["track_id"].unique())
    print(f"  Ratings : {len(ratings_df):,}  |  Items : {len(all_tracks_set):,}")

    # -----------------------------------------------------------------------
    # 2. Train/test split
    # -----------------------------------------------------------------------
    print("\n[2/5] Creating 80/20 train/test split...")
    train_parts = []
    has_ts = "timestamp" in ratings_df.columns
    for uid, grp in ratings_df.groupby("user_id"):
        grp = grp.sort_values("timestamp") if has_ts else grp.sample(frac=1, random_state=args.seed)
        n_train = max(1, int(0.8 * len(grp)))
        train_parts.append(grp.iloc[:n_train])
    train_df = pd.concat(train_parts, ignore_index=True)
    print(f"  Train ratings: {len(train_df):,}")

    # -----------------------------------------------------------------------
    # 3. Select users
    # -----------------------------------------------------------------------
    print("\n[3/5] Selecting users...")
    train_counts = train_df.groupby("user_id").size()
    test_df_temp = ratings_df[~ratings_df.index.isin(train_df.index)]
    test_counts  = test_df_temp.groupby("user_id").size() if len(test_df_temp) else pd.Series(dtype=int)

    eligible = train_counts[train_counts >= 10].index
    if len(test_counts):
        eligible = eligible[eligible.isin(test_counts[test_counts >= 3].index)]

    n_select       = min(args.n_users, len(eligible))
    selected_users = np.random.choice(eligible, size=n_select, replace=False)
    print(f"  Eligible : {len(eligible):,}  |  Selected : {n_select:,}")

    # -----------------------------------------------------------------------
    # 4. Train SVD
    # -----------------------------------------------------------------------
    print("\n[4/5] Training SVD (50 factors, 20 epochs)...")
    reader        = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(train_df[["user_id", "track_id", "rating"]], reader)
    trainset      = surprise_data.build_full_trainset()
    svd           = SVD(n_factors=50, n_epochs=20, random_state=args.seed, verbose=False)
    svd.fit(trainset)
    print("  SVD trained.")

    svd_factors = {
        "pu":          svd.pu,
        "qi":          svd.qi,
        "bu":          svd.bu,
        "bi":          svd.bi,
        "global_mean": trainset.global_mean,
        "uid_map":     {raw: inner for raw, inner in trainset._raw2inner_id_users.items()},
        "iid_map":     {raw: inner for raw, inner in trainset._raw2inner_id_items.items()},
    }

    # Build E_u ids and PURS interest clusters per user
    print(f"  Building interest clusters (k={args.n_clusters}) per user...")
    user_args = []
    n_fallback = 0
    for uid in selected_users:
        E_u_ids = set(train_df.loc[train_df["user_id"] == uid, "track_id"])
        centroids, weights = build_user_clusters(E_u_ids, svd_factors, args.n_clusters)
        if centroids is None:
            n_fallback += 1
        user_args.append((uid, E_u_ids, centroids, weights))

    print(f"  Clusters built for {len(user_args):,} users "
          f"({n_fallback} fallback to uniform distance)")

    # -----------------------------------------------------------------------
    # 5. Parallel generation
    # -----------------------------------------------------------------------
    print(f"\n[5/5] Generating recommendations ({args.workers} workers)...")
    print(f"  {len(user_args):,} users × {len(ALPHA_VALUES)} alphas\n")

    all_recommendations = {a: {} for a in ALPHA_VALUES}
    out_pkl   = os.path.join(args.out_dir, "recommendations_fair_complete.pkl")
    phase_start = datetime.now()

    with Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(svd_factors, all_tracks_set),
    ) as pool:
        for done, (user_id, user_recs) in enumerate(
            pool.imap_unordered(process_user, user_args, chunksize=10), start=1
        ):
            for alpha in ALPHA_VALUES:
                all_recommendations[alpha][user_id] = user_recs[alpha]

            if done % 10 == 0 or done == len(user_args):
                elapsed   = (datetime.now() - phase_start).total_seconds()
                rate      = done / elapsed
                remaining = (len(user_args) - done) / rate if rate > 0 else 0
                print(f"  {done:4d}/{len(user_args)} users | "
                      f"{rate:.1f} users/s | ~{remaining/60:.1f} min remaining")

            if done % args.checkpoint_every == 0:
                pickle.dump(all_recommendations, open(out_pkl, "wb"), protocol=4)
                print(f"  [checkpoint at {done} users]")

    pickle.dump(all_recommendations, open(out_pkl, "wb"), protocol=4)
    print(f"\n  Saved: {out_pkl}")

    info = {
        "alpha_values":    ALPHA_VALUES,
        "n_users":         len(user_args),
        "selected_users":  list(selected_users),
        "global_seed":     args.seed,
        "cf_params":       {"n_factors": 50, "n_epochs": 20},
        "unexpectedness":  "PURS-inspired latent-space k-means cluster distance",
        "n_clusters":      args.n_clusters,
        "generation_date": datetime.now().isoformat(),
    }
    pickle.dump(info, open(os.path.join(args.out_dir, "generation_info.pkl"), "wb"), protocol=4)

    total_min = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n{'='*80}")
    print(f"Completed in {total_min:.1f} minutes | {len(user_args):,} users processed")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
