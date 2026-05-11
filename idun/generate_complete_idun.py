#!/usr/bin/env python3
"""
generate_complete_idun.py
=========================
Parallelised recommendation generation for IDUN supercomputer.

Generates top-10 recommendations for N_USERS users across 11 alpha values
using a single shared SVD model. Parallelises the heavy per-user computation
across CPU cores with multiprocessing (fork-based, Linux default on IDUN).

Key improvements over generate_complete.py:
  - Vectorised CF prediction (numpy matmul instead of per-call svd.predict)
  - Parallelised across users via multiprocessing.Pool
  - Configurable N_USERS and worker count via command-line arguments
  - Uses full 3.3M ratings for SVD (no 200K sub-sample)
  - Frequent checkpointing — safe against SLURM wall-time kills
  - All paths relative to configurable --data_dir and --out_dir

Usage:
    python generate_complete_idun.py \\
        --n_users 5000 \\
        --workers 32 \\
        --data_dir ../AMBAR \\
        --out_dir idun_results

Typical IDUN runtime:
    5 000 users, 32 workers:  ~50 minutes
    5 000 users, 64 workers:  ~25 minutes
"""

import argparse
import os
import pickle
import warnings
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Alpha grid — same as generate_complete.py for compatibility
# ---------------------------------------------------------------------------
ALPHA_VALUES = [0.0, 0.15, 0.25, 0.30, 0.35, 0.40, 0.50, 0.65, 0.75, 0.85, 1.0]

# ---------------------------------------------------------------------------
# Global state — populated in each worker via Pool initializer.
# Using an initializer works on both Linux (IDUN, fork) and macOS (spawn).
# ---------------------------------------------------------------------------
_SVD_FACTORS = None      # dict of numpy arrays extracted from trained SVD
_TRACK_META  = None      # dict: track_id -> frozenset of category strings
_ALL_TRACKS  = None      # set of all track ids


def _init_worker(svd_factors, track_meta, all_tracks):
    """Initialiser run once per worker process to set shared read-only state."""
    global _SVD_FACTORS, _TRACK_META, _ALL_TRACKS
    _SVD_FACTORS = svd_factors
    _TRACK_META  = track_meta
    _ALL_TRACKS  = all_tracks


# ---------------------------------------------------------------------------
# Helper: build E_u as a frozenset of category style strings for one user
# ---------------------------------------------------------------------------
def build_E_u_categories(user_id, train_df, track_meta):
    """
    Returns:
        E_u_ids        : set of track IDs the user has rated (to exclude from candidates)
        E_u_categories : frozenset of all category style tags in those tracks
    """
    rated_ids = set(train_df.loc[train_df["user_id"] == user_id, "track_id"])
    cats = set()
    for tid in rated_ids:
        if tid in track_meta:
            cats.update(track_meta[tid])
    return rated_ids, frozenset(cats)


# ---------------------------------------------------------------------------
# Helper: vectorised CF predictions for one user against many items
# ---------------------------------------------------------------------------
def _predict_batch(user_id, item_ids):
    """
    Returns a numpy array of predicted ratings (clipped to [1, 5]) for
    each item_id in item_ids, using the global _SVD_FACTORS.
    Unknown users/items fall back to global_mean + available biases.
    """
    f = _SVD_FACTORS
    gm = f["global_mean"]
    uid_map = f["uid_map"]
    iid_map = f["iid_map"]

    # User vector and bias
    if user_id in uid_map:
        u_idx = uid_map[user_id]
        pu = f["pu"][u_idx]       # shape (n_factors,)
        bu = f["bu"][u_idx]
    else:
        pu = np.zeros(f["pu"].shape[1])
        bu = 0.0

    # Build item index arrays (separate known vs unknown)
    known_mask  = np.array([iid in iid_map for iid in item_ids])
    known_items = [iid for iid in item_ids if iid in iid_map]
    unknown_count = int((~known_mask).sum())

    preds = np.full(len(item_ids), gm + bu, dtype=np.float32)

    if known_items:
        i_idx   = np.array([iid_map[iid] for iid in known_items])
        qi_mat  = f["qi"][i_idx]          # shape (n_known, n_factors)
        bi_arr  = f["bi"][i_idx]          # shape (n_known,)
        known_preds = gm + bu + bi_arr + qi_mat @ pu   # vectorised
        preds[known_mask] = known_preds

    return np.clip(preds, 1.0, 5.0)


# ---------------------------------------------------------------------------
# Per-user worker function — called in parallel by Pool
# ---------------------------------------------------------------------------
def process_user(args):
    """
    Computes top-10 recommendations for one user across ALL alpha values.

    Returns:
        (user_id, {alpha: [list of 10 rec dicts]})
    """
    user_id, E_u_ids, E_u_categories = args

    candidate_ids = list(_ALL_TRACKS - E_u_ids)
    if not candidate_ids:
        return user_id, {a: [] for a in ALPHA_VALUES}

    # --- Vectorised CF predictions for all candidates ---
    cf_preds = _predict_batch(user_id, candidate_ids)          # shape (N_cand,)
    cf_norm  = (cf_preds - 1.0) / 4.0                         # normalise to [0,1]

    # --- Jaccard distances for all candidates ---
    distances = np.empty(len(candidate_ids), dtype=np.float32)
    for k, tid in enumerate(candidate_ids):
        track_cats = _TRACK_META.get(tid)
        if track_cats is None or len(track_cats) == 0 or len(E_u_categories) == 0:
            distances[k] = 1.0
        else:
            inter = len(track_cats & E_u_categories)
            union = len(track_cats | E_u_categories)
            distances[k] = 1.0 - inter / union if union > 0 else 1.0

    # --- Apply all alpha values in one pass ---
    user_recs = {}
    for alpha in ALPHA_VALUES:
        scores = alpha * distances + (1.0 - alpha) * cf_norm
        top10_idx = np.argpartition(scores, -10)[-10:]
        top10_idx = top10_idx[np.argsort(scores[top10_idx])[::-1]]

        recs = []
        for idx in top10_idx:
            recs.append({
                "track_id":      candidate_ids[idx],
                "distance":      float(distances[idx]),
                "cf_score":      float(cf_preds[idx]),
                "combined_score": float(scores[idx]),
            })
        user_recs[alpha] = recs

    return user_id, user_recs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Parallelised recommendation generation for IDUN")
    parser.add_argument("--n_users",  type=int, default=5000,   help="Number of users to process (default 5000)")
    parser.add_argument("--workers",  type=int, default=32,     help="Number of parallel worker processes (default 32)")
    parser.add_argument("--data_dir", type=str, default="../AMBAR", help="Path to AMBAR directory")
    parser.add_argument("--out_dir",  type=str, default="idun_results", help="Output directory for pkl files")
    parser.add_argument("--seed",     type=int, default=42,     help="Random seed")
    parser.add_argument("--checkpoint_every", type=int, default=500, help="Save checkpoint every N users")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    start_time = datetime.now()
    print("=" * 80)
    print("IDUN — PARALLELISED RECOMMENDATION GENERATION")
    print("=" * 80)
    print(f"Started : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config  : {args.n_users} users | {args.workers} workers | seed={args.seed}")
    print(f"Data    : {args.data_dir}")
    print(f"Output  : {args.out_dir}")
    print(f"Alphas  : {ALPHA_VALUES}")
    print()

    # -----------------------------------------------------------------------
    # 1. Load AMBAR data
    # -----------------------------------------------------------------------
    print("[1/6] Loading AMBAR dataset...")
    ratings_df = pd.read_csv(os.path.join(args.data_dir, "ratings_info.csv"))
    tracks_df  = pd.read_csv(os.path.join(args.data_dir, "tracks_info.csv"))
    print(f"  Ratings : {len(ratings_df):,}")
    print(f"  Tracks  : {len(tracks_df):,} unique tracks")

    # -----------------------------------------------------------------------
    # 2. Parse track metadata → frozenset of category style tags per track
    # -----------------------------------------------------------------------
    print("\n[2/6] Parsing track metadata...")

    def parse_cats(val):
        if pd.isna(val):
            return frozenset()
        if isinstance(val, list):
            tags = val
        else:
            tags = [s.strip() for s in str(val).split("|")]
        # split compound tags like "Rock|Pop" → ["Rock", "Pop"]
        expanded = set()
        for t in tags:
            for part in t.split("|"):
                part = part.strip()
                if part:
                    expanded.add(part)
        return frozenset(expanded)

    tracks_df["_cats"] = tracks_df["category_styles"].apply(parse_cats)
    track_meta = dict(zip(tracks_df["track_id"], tracks_df["_cats"]))
    all_tracks_set = set(tracks_df["track_id"].unique())
    print(f"  Pre-computed metadata for {len(track_meta):,} tracks")

    # -----------------------------------------------------------------------
    # 3. Train/test split (80/20 per user, temporal if timestamps present)
    # -----------------------------------------------------------------------
    print("\n[3/6] Creating 80/20 train/test split...")
    train_parts = []
    has_ts = "timestamp" in ratings_df.columns
    for uid, grp in ratings_df.groupby("user_id"):
        grp = grp.sort_values("timestamp") if has_ts else grp.sample(frac=1, random_state=args.seed)
        n_train = max(1, int(0.8 * len(grp)))
        train_parts.append(grp.iloc[:n_train])
    train_df = pd.concat(train_parts, ignore_index=True)
    print(f"  Train ratings: {len(train_df):,}")

    # -----------------------------------------------------------------------
    # 4. Select users — those with >= 10 train ratings and >= 3 test ratings
    # -----------------------------------------------------------------------
    print("\n[4/6] Selecting users...")
    train_counts = train_df.groupby("user_id").size()
    test_df_temp = ratings_df[~ratings_df.index.isin(train_df.index)]
    test_counts  = test_df_temp.groupby("user_id").size() if len(test_df_temp) else pd.Series(dtype=int)

    eligible = train_counts[train_counts >= 10].index
    if len(test_counts):
        eligible = eligible[eligible.isin(test_counts[test_counts >= 3].index)]

    print(f"  Eligible users : {len(eligible):,}")
    n_select = min(args.n_users, len(eligible))
    selected_users = np.random.choice(eligible, size=n_select, replace=False)
    print(f"  Selected       : {n_select:,} users")

    # Build E_u for each selected user
    print("  Building E_u ...")
    user_args = []
    for uid in selected_users:
        E_u_ids, E_u_cats = build_E_u_categories(uid, train_df, track_meta)
        user_args.append((uid, E_u_ids, E_u_cats))
    print(f"  E_u built for {len(user_args):,} users")

    # -----------------------------------------------------------------------
    # 5. Train SVD on full training set — once for all alpha values
    # -----------------------------------------------------------------------
    print("\n[5/6] Training SVD (50 factors, 20 epochs) on full train set...")
    reader   = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(train_df[["user_id", "track_id", "rating"]], reader)
    svd      = SVD(n_factors=50, n_epochs=20, random_state=args.seed, verbose=False)
    trainset = surprise_data.build_full_trainset()
    svd.fit(trainset)
    print("  SVD trained.")

    # Build the shared data bundle passed to each worker via initializer
    svd_factors = {
        "pu":          svd.pu,
        "qi":          svd.qi,
        "bu":          svd.bu,
        "bi":          svd.bi,
        "global_mean": trainset.global_mean,
        "uid_map":     {raw: inner for raw, inner in trainset._raw2inner_id_users.items()},
        "iid_map":     {raw: inner for raw, inner in trainset._raw2inner_id_items.items()},
    }

    # -----------------------------------------------------------------------
    # 6. Parallel recommendation generation
    # -----------------------------------------------------------------------
    print(f"\n[6/6] Generating recommendations ({args.workers} workers)...")
    print(f"  {len(user_args):,} users × {len(ALPHA_VALUES)} alphas")
    print(f"  Checkpoint every {args.checkpoint_every} users\n")

    all_recommendations = {a: {} for a in ALPHA_VALUES}
    out_pkl = os.path.join(args.out_dir, "recommendations_fair_complete.pkl")

    phase_start = datetime.now()

    with Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(svd_factors, track_meta, all_tracks_set),
    ) as pool:
        for done, (user_id, user_recs) in enumerate(
            pool.imap_unordered(process_user, user_args, chunksize=10), start=1
        ):
            for alpha in ALPHA_VALUES:
                all_recommendations[alpha][user_id] = user_recs[alpha]

            if done % 100 == 0 or done == len(user_args):
                elapsed  = (datetime.now() - phase_start).total_seconds()
                rate     = done / elapsed
                remaining = (len(user_args) - done) / rate if rate > 0 else 0
                print(f"  {done:5d}/{len(user_args)} users | "
                      f"{rate:.1f} users/s | "
                      f"~{remaining/60:.1f} min remaining")

            if done % args.checkpoint_every == 0:
                pickle.dump(all_recommendations, open(out_pkl, "wb"), protocol=4)
                print(f"  [checkpoint saved at {done} users]")

    # -----------------------------------------------------------------------
    # Final save
    # -----------------------------------------------------------------------
    pickle.dump(all_recommendations, open(out_pkl, "wb"), protocol=4)
    print(f"\n  Saved: {out_pkl}")

    info = {
        "alpha_values":   ALPHA_VALUES,
        "n_users":        len(user_args),
        "selected_users": list(selected_users),
        "global_seed":    args.seed,
        "cf_params":      {"n_factors": 50, "n_epochs": 20},
        "n_ratings_train": len(train_df),
        "generation_date": datetime.now().isoformat(),
    }
    info_pkl = os.path.join(args.out_dir, "generation_info.pkl")
    pickle.dump(info, open(info_pkl, "wb"), protocol=4)
    print(f"  Saved: {info_pkl}")

    total_min = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n{'=' * 80}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {total_min:.1f} minutes")
    print(f"Users processed: {len(user_args):,}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
