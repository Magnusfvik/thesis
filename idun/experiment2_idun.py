#!/usr/bin/env python3
"""
experiment2_idun.py
===================
Experiment 2 for IDUN: Multi-Objective Weighted Scalarization Analysis.

Loads the pre-computed recommendations from generate_complete_idun.py
and evaluates serendipity, diversity, unexpectedness and relevance across
all alpha values. The heavy recommendation generation is already done —
this script is fast (minutes, not hours).

Must be run AFTER generate_complete_idun.py has completed.

Usage:
    python experiment2_idun.py \\
        --recs_pkl  idun_results/recommendations_fair_complete.pkl \\
        --data_dir  ../AMBAR \\
        --out_dir   idun_results

Output (in --out_dir):
    exp2_results.csv
    exp2_summary.txt
    exp2_serendipity_curve.png
    exp2_tradeoff_frontier.png
"""

import argparse
import os
import pickle
import warnings
from datetime import datetime
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Harmonic mean serendipity — no arbitrary thresholds.
# U(r) = distance (already in [0,1])
# R(r) = (cf_score - 1) / 4  →  maps [1,5] to [0,1]
# S(r) = 2*U*R / (U+R)  (harmonic mean; 0 if either is 0)

# Global track metadata for workers
_TRACK_META = None


def _init_worker(track_meta):
    global _TRACK_META
    _TRACK_META = track_meta


def _jaccard(a, b):
    if not a or not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return 1.0 - inter / union if union else 1.0


# ---------------------------------------------------------------------------
# Per-alpha metric computation (parallelised over alpha values is optional;
# here we parallelise over users within each alpha for ILD computation)
# ---------------------------------------------------------------------------

def compute_ild_for_user(recs):
    """Intra-list diversity for one user's top-10 list."""
    cats = []
    for rec in recs:
        tc = _TRACK_META.get(rec["track_id"])
        if tc:
            cats.append(tc)
    if len(cats) < 2:
        return None
    dists = [_jaccard(cats[i], cats[j]) for i in range(len(cats)) for j in range(i+1, len(cats))]
    return float(np.mean(dists)) if dists else None


def evaluate_alpha(alpha, user_recs, workers):
    """
    Compute all metrics for one alpha value.
    user_recs: dict {user_id: [list of rec dicts]}
    """
    serendipity_scores = []
    avg_distances      = []
    avg_cf_scores      = []
    ild_scores         = []

    all_recs_lists = list(user_recs.values())

    # ILD in parallel
    with Pool(processes=workers, initializer=_init_worker, initargs=(_TRACK_META,)) as pool:
        ilds = pool.map(compute_ild_for_user, all_recs_lists, chunksize=50)

    for user_id, recs in user_recs.items():
        if not recs:
            continue

        # Harmonic mean serendipity per recommendation
        s_scores = []
        for r in recs:
            U = float(r["distance"])
            R = (float(r["cf_score"]) - 1.0) / 4.0
            denom = U + R
            s_scores.append(2 * U * R / denom if denom > 0 else 0.0)
        serendipity_scores.append(float(np.mean(s_scores)))

        avg_distances.append(np.mean([r["distance"] for r in recs]))
        avg_cf_scores.append(np.mean([r["cf_score"] for r in recs]))

    ild_values = [v for v in ilds if v is not None]

    return {
        "alpha":             alpha,
        "serendipity":       float(np.mean(serendipity_scores)) if serendipity_scores else 0.0,
        "avg_distance":      float(np.mean(avg_distances))      if avg_distances      else 0.0,
        "avg_cf_score":      float(np.mean(avg_cf_scores))      if avg_cf_scores      else 0.0,
        "diversity_ild":     float(np.mean(ild_values))         if ild_values         else 0.0,
        "n_users":           len(serendipity_scores),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Exp 2 — Multi-objective analysis (IDUN)")
    parser.add_argument("--recs_pkl",  type=str,
                        default="idun_results/recommendations_fair_complete.pkl",
                        help="Path to recommendations pkl from generate_complete_idun.py")
    parser.add_argument("--data_dir",  type=str, default="../AMBAR")
    parser.add_argument("--out_dir",   type=str, default="idun_results")
    parser.add_argument("--workers",   type=int, default=16,
                        help="Workers for ILD computation (default 16)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    start_time = datetime.now()

    print("=" * 80)
    print("EXPERIMENT 2 (IDUN) — MULTI-OBJECTIVE WEIGHTED SCALARIZATION")
    print("=" * 80)
    print(f"Started : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Recs pkl: {args.recs_pkl}")
    print()

    # -----------------------------------------------------------------------
    # 1. Load recommendations pkl
    # -----------------------------------------------------------------------
    print("[1/4] Loading recommendations...")
    if not os.path.exists(args.recs_pkl):
        print(f"  ERROR: {args.recs_pkl} not found.")
        print("  Run generate_complete_idun.py first.")
        return

    all_recs = pickle.load(open(args.recs_pkl, "rb"))
    alpha_values = sorted(all_recs.keys())
    n_users = len(all_recs[alpha_values[0]])
    print(f"  {len(alpha_values)} alpha values | {n_users:,} users")
    print(f"  Alpha values: {alpha_values}")

    # -----------------------------------------------------------------------
    # 2. Load track metadata for ILD
    # -----------------------------------------------------------------------
    print("\n[2/4] Loading track metadata...")
    tracks_df = pd.read_csv(os.path.join(args.data_dir, "tracks_info.csv"))

    def parse_cats(val):
        if pd.isna(val):
            return frozenset()
        tags = val if isinstance(val, list) else [s.strip() for s in str(val).split("|")]
        expanded = set()
        for t in tags:
            for part in t.split("|"):
                part = part.strip()
                if part:
                    expanded.add(part)
        return frozenset(expanded)

    tracks_df["_cats"] = tracks_df["category_styles"].apply(parse_cats)
    global _TRACK_META
    _TRACK_META = dict(zip(tracks_df["track_id"], tracks_df["_cats"]))
    print(f"  Metadata for {len(_TRACK_META):,} tracks")

    # -----------------------------------------------------------------------
    # 3. Evaluate each alpha value
    # -----------------------------------------------------------------------
    print(f"\n[3/4] Evaluating {len(alpha_values)} alpha values...")
    results = []

    for alpha in alpha_values:
        print(f"  α={alpha:.2f} ...", end=" ", flush=True)
        t0 = datetime.now()
        metrics = evaluate_alpha(alpha, all_recs[alpha], args.workers)
        elapsed = (datetime.now() - t0).total_seconds()
        results.append(metrics)
        print(f"serendipity={metrics['serendipity']:.3f}  "
              f"dist={metrics['avg_distance']:.3f}  "
              f"cf={metrics['avg_cf_score']:.3f}  "
              f"({elapsed:.1f}s)")

    results_df = pd.DataFrame(results)
    print("\nResults table:")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Key findings
    best_seren_alpha = results_df.loc[results_df["serendipity"].idxmax(), "alpha"]
    best_seren_val   = results_df["serendipity"].max()
    print(f"\nBest serendipity: α={best_seren_alpha:.2f} → {best_seren_val:.4f}")

    # -----------------------------------------------------------------------
    # 4. Plots and outputs
    # -----------------------------------------------------------------------
    print("\n[4/4] Saving plots and results...")

    # --- Serendipity curve ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Experiment 2: Multi-Objective Analysis — {n_users:,} Users (IDUN)",
        fontsize=14, fontweight="bold",
    )

    ax1 = axes[0, 0]
    ax1.plot(results_df["alpha"], results_df["serendipity"],
             "o-", linewidth=2.5, markersize=9, color="#2ecc71")
    ax1.axvline(best_seren_alpha, color="red", linestyle="--", linewidth=1.5,
                label=f"Optimal α={best_seren_alpha:.2f}")
    ax1.set_xlabel("α (weight on unexpectedness)", fontweight="bold")
    ax1.set_ylabel("Mean Serendipity", fontweight="bold")
    ax1.set_title("Serendipity vs α", fontweight="bold")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(results_df["alpha"], results_df["avg_distance"],
             "o-", linewidth=2.5, markersize=9, color="#9b59b6")
    ax2.set_xlabel("α", fontweight="bold")
    ax2.set_ylabel("Mean Distance from E_u", fontweight="bold")
    ax2.set_title("Unexpectedness vs α", fontweight="bold")
    ax2.grid(alpha=0.3)

    ax3 = axes[1, 0]
    ax3.plot(results_df["alpha"], results_df["avg_cf_score"],
             "o-", linewidth=2.5, markersize=9, color="#e74c3c")
    ax3.set_xlabel("α", fontweight="bold")
    ax3.set_ylabel("Mean CF Score", fontweight="bold")
    ax3.set_title("Relevance vs α", fontweight="bold")
    ax3.grid(alpha=0.3)

    ax4 = axes[1, 1]
    ax4.plot(results_df["alpha"], results_df["diversity_ild"],
             "o-", linewidth=2.5, markersize=9, color="#3498db")
    ax4.set_xlabel("α", fontweight="bold")
    ax4.set_ylabel("Intra-List Distance", fontweight="bold")
    ax4.set_title("Diversity (ILD) vs α", fontweight="bold")
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    p1 = os.path.join(args.out_dir, "exp2_serendipity_curve.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    print(f"  Saved: {p1}")

    # --- Trade-off frontier ---
    fig2, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(results_df["avg_distance"], results_df["avg_cf_score"],
                    c=results_df["serendipity"], cmap="RdYlGn",
                    s=200, edgecolors="black", linewidths=1.5, zorder=5)
    plt.colorbar(sc, ax=ax, label="Serendipity")
    ax.plot(results_df["avg_distance"], results_df["avg_cf_score"],
            "--", color="gray", linewidth=1.5, alpha=0.5, zorder=3)
    for _, row in results_df.iterrows():
        ax.annotate(f"α={row['alpha']:.2f}",
                    xy=(row["avg_distance"], row["avg_cf_score"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Unexpectedness (Mean Distance from E_u)", fontweight="bold")
    ax.set_ylabel("Relevance (Mean CF Score)", fontweight="bold")
    ax.set_title(f"Trade-off Frontier: Unexpectedness vs Relevance — {n_users:,} Users",
                 fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(args.out_dir, "exp2_tradeoff_frontier.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"  Saved: {p2}")

    # --- CSV ---
    csv_path = os.path.join(args.out_dir, "exp2_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # --- Summary text ---
    summary_path = os.path.join(args.out_dir, "exp2_summary.txt")
    with open(summary_path, "w") as f:
        f.write("EXPERIMENT 2 — MULTI-OBJECTIVE WEIGHTED SCALARIZATION (IDUN)\n")
        f.write(f"Users: {n_users:,}  |  Alpha values: {alpha_values}\n")
        f.write(f"Serendipity metric: harmonic mean of U=distance and R=(cf_score-1)/4\n\n")
        f.write(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        f.write(f"\n\nBest serendipity: α={best_seren_alpha:.2f} → {best_seren_val:.4f}\n")
    print(f"  Saved: {summary_path}")

    total_min = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n{'='*80}")
    print(f"Completed in {total_min:.1f} minutes")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
