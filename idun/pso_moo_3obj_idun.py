#!/usr/bin/env python3
"""
pso_moo_3obj_idun.py
====================
3-Objective MOPSO: Unexpectedness × Relevance × Diversity (ILD).

Objectives (all maximized):
  f1 = mean unexpectedness (latent-space distance from user clusters)
  f2 = mean relevance      (SVD CF score)
  f3 = mean ILD            (intra-list Jaccard diversity of top-10)

Knee point: solution closest to the ideal point (1,1,1) in
normalised objective space.

Usage:
    python pso_moo_3obj_idun.py \\
        --recs_pkl idun_results_purs/recommendations_fair_complete.pkl \\
        --data_dir ../AMBAR \\
        --out_dir  idun_results_purs

Output:
    pso_moo_3obj_front.csv
    pso_moo_3obj_pareto.png
    pso_moo_3obj_summary.txt
"""

import argparse
import os
import pickle
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

warnings.filterwarnings("ignore")

# ============================================================================
# MOPSO hyperparameters
# ============================================================================
N_PARTICLES  = 30
MAX_ITER     = 150
ARCHIVE_SIZE = 40
W  = 0.729
C1 = 1.49445
C2 = 1.49445

N_OBJ = 3   # unexpectedness, relevance, ILD


# ============================================================================
# Helpers
# ============================================================================

def dominates(a, b):
    """True if a dominates b (all 3 objectives, maximise)."""
    return (all(a[i] >= b[i] for i in range(N_OBJ)) and
            any(a[i] >  b[i] for i in range(N_OBJ)))


def crowding_distance(archive):
    n = len(archive)
    if n <= 2:
        return [np.inf] * n
    distances = np.zeros(n)
    for obj_idx in range(N_OBJ):
        values = np.array([s["objectives"][obj_idx] for s in archive])
        order  = np.argsort(values)
        obj_range = values[order[-1]] - values[order[0]]
        if obj_range == 0:
            continue
        distances[order[0]]  = np.inf
        distances[order[-1]] = np.inf
        for i in range(1, n - 1):
            distances[order[i]] += (values[order[i+1]] - values[order[i-1]]) / obj_range
    return distances.tolist()


def update_archive(archive, candidate, max_size):
    obj = candidate["objectives"]
    for member in archive:
        if dominates(member["objectives"], obj):
            return archive
    archive = [m for m in archive if not dominates(obj, m["objectives"])]
    archive.append(candidate)
    if len(archive) > max_size:
        cd = crowding_distance(archive)
        cd_finite = [d if np.isfinite(d) else 1e9 for d in cd]
        archive.pop(int(np.argmin(cd_finite)))
    return archive


def select_leader(archive):
    if len(archive) == 1:
        return archive[0]
    cd   = crowding_distance(archive)
    i, j = np.random.choice(len(archive), size=2, replace=False)
    return archive[i] if cd[i] >= cd[j] else archive[j]


# ============================================================================
# ILD helpers
# ============================================================================

def _jaccard(a, b):
    if not a or not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return 1.0 - inter / union if union else 1.0


def compute_ild_grid(recs, alpha_grid, track_meta):
    """Compute mean ILD for each alpha in the grid."""
    ild_vals = []
    for alpha in alpha_grid:
        user_ilds = []
        for user_recs in recs[alpha].values():
            if not user_recs:
                continue
            cats = [track_meta.get(r["track_id"]) for r in user_recs]
            cats = [c for c in cats if c]
            if len(cats) < 2:
                continue
            dists = [_jaccard(cats[i], cats[j])
                     for i in range(len(cats))
                     for j in range(i + 1, len(cats))]
            user_ilds.append(float(np.mean(dists)))
        ild_vals.append(float(np.mean(user_ilds)) if user_ilds else 0.0)
    return np.array(ild_vals)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="3-Objective MOPSO (IDUN)")
    parser.add_argument("--recs_pkl", type=str,
                        default="idun_results_purs/recommendations_fair_complete.pkl")
    parser.add_argument("--data_dir", type=str, default="../AMBAR",
                        help="Directory containing tracks_info.csv")
    parser.add_argument("--out_dir",  type=str, default="idun_results_purs")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    start_time = datetime.now()

    print("=" * 80)
    print("3-OBJECTIVE MOPSO: UNEXPECTEDNESS × RELEVANCE × DIVERSITY (IDUN)")
    print("=" * 80)
    print(f"Started : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Recs pkl: {args.recs_pkl}")
    print()

    # -----------------------------------------------------------------------
    # 1. Load recommendations + track metadata
    # -----------------------------------------------------------------------
    print("[1/4] Loading data...")

    if not os.path.exists(args.recs_pkl):
        print(f"  ERROR: {args.recs_pkl} not found.")
        return

    recs = pickle.load(open(args.recs_pkl, "rb"))
    alpha_grid = sorted(recs.keys())
    alpha_arr  = np.array(alpha_grid)

    # Track metadata for ILD
    tracks_path = os.path.join(args.data_dir, "tracks_info.csv")
    tracks_df   = pd.read_csv(tracks_path)

    def parse_cats(val):
        if pd.isna(val):
            return frozenset()
        tags = [s.strip() for s in str(val).split("|")]
        return frozenset(t for t in tags if t)

    col = "category_styles" if "category_styles" in tracks_df.columns else "styles"
    tracks_df["_cats"] = tracks_df[col].apply(parse_cats)
    track_meta = dict(zip(tracks_df["track_id"], tracks_df["_cats"]))
    print(f"  {len(alpha_grid)} alpha values | {len(track_meta):,} tracks in metadata")

    # -----------------------------------------------------------------------
    # 2. Build interpolators for all 3 objectives
    # -----------------------------------------------------------------------
    print("[2/4] Building interpolators...")

    grid_unexp, grid_rel, grid_seren = [], [], []

    for alpha in alpha_grid:
        all_unexp, all_rel, all_seren = [], [], []
        for user_recs in recs[alpha].values():
            if not user_recs:
                continue
            all_unexp.append(np.mean([r["distance"] for r in user_recs]))
            all_rel.append(np.mean([r["cf_score"]  for r in user_recs]))
            s_vals = []
            for r in user_recs:
                U = float(r["distance"])
                R = (float(r["cf_score"]) - 1.0) / 4.0
                denom = U + R
                s_vals.append(2 * U * R / denom if denom > 0 else 0.0)
            all_seren.append(np.mean(s_vals))
        grid_unexp.append(np.mean(all_unexp))
        grid_rel.append(np.mean(all_rel))
        grid_seren.append(np.mean(all_seren))

    unexp_arr = np.array(grid_unexp)
    rel_arr   = np.array(grid_rel)
    seren_arr = np.array(grid_seren)

    print("  Computing ILD across alpha grid (may take a moment)...")
    ild_arr = compute_ild_grid(recs, alpha_grid, track_meta)

    interp_unexp = PchipInterpolator(alpha_arr, unexp_arr)
    interp_rel   = PchipInterpolator(alpha_arr, rel_arr)
    interp_seren = PchipInterpolator(alpha_arr, seren_arr)
    interp_ild   = PchipInterpolator(alpha_arr, ild_arr)

    def evaluate(alpha):
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return (float(interp_unexp(alpha)),
                float(interp_rel(alpha)),
                float(interp_ild(alpha)))

    print(f"  Unexpectedness : [{unexp_arr.min():.3f}, {unexp_arr.max():.3f}]")
    print(f"  Relevance      : [{rel_arr.min():.3f},   {rel_arr.max():.3f}]")
    print(f"  ILD            : [{ild_arr.min():.3f},   {ild_arr.max():.3f}]")
    print(f"  Serendipity    : [{seren_arr.min():.3f},  {seren_arr.max():.3f}]")

    # -----------------------------------------------------------------------
    # 3. Run MOPSO
    # -----------------------------------------------------------------------
    print(f"\n[3/4] Running 3-objective MOPSO "
          f"({N_PARTICLES} particles, {MAX_ITER} iterations)...")

    particles = []
    for _ in range(N_PARTICLES):
        pos = np.random.random()
        obj = evaluate(pos)
        particles.append({
            "position":         pos,
            "velocity":         np.random.uniform(-0.1, 0.1),
            "objectives":       obj,
            "pbest_position":   pos,
            "pbest_objectives": obj,
        })

    archive = []
    for p in particles:
        archive = update_archive(
            archive, {"alpha": p["position"], "objectives": p["objectives"]}, ARCHIVE_SIZE
        )

    for iteration in range(MAX_ITER):
        for p in particles:
            leader = select_leader(archive)
            r1 = np.random.random()
            r2 = np.random.random()
            p["velocity"] = (W * p["velocity"]
                             + C1 * r1 * (p["pbest_position"] - p["position"])
                             + C2 * r2 * (leader["alpha"]     - p["position"]))
            p["position"]   = float(np.clip(p["position"] + p["velocity"], 0.0, 1.0))
            p["objectives"] = evaluate(p["position"])

            if dominates(p["objectives"], p["pbest_objectives"]):
                p["pbest_position"]   = p["position"]
                p["pbest_objectives"] = p["objectives"]
            elif not dominates(p["pbest_objectives"], p["objectives"]):
                if np.random.random() < 0.5:
                    p["pbest_position"]   = p["position"]
                    p["pbest_objectives"] = p["objectives"]

            archive = update_archive(
                archive, {"alpha": p["position"], "objectives": p["objectives"]}, ARCHIVE_SIZE
            )

        if (iteration + 1) % 30 == 0 or iteration == MAX_ITER - 1:
            print(f"  Iteration {iteration+1:3d}/{MAX_ITER} | archive: {len(archive)} solutions")

    print(f"\n  Done — {len(archive)} non-dominated solutions")

    # -----------------------------------------------------------------------
    # 4. Post-process
    # -----------------------------------------------------------------------
    print("\n[4/4] Post-processing and plotting...")

    archive.sort(key=lambda s: s["objectives"][0])

    unexp_front = np.array([s["objectives"][0] for s in archive])
    rel_front   = np.array([s["objectives"][1] for s in archive])
    ild_front   = np.array([s["objectives"][2] for s in archive])
    alpha_front = np.array([s["alpha"]         for s in archive])
    seren_front = np.array([float(np.clip(interp_seren(a), 0, 1)) for a in alpha_front])

    # Knee: normalise each objective to [0,1], find solution closest to ideal (1,1,1)
    def norm(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-9)

    u_n = norm(unexp_front)
    r_n = norm(rel_front)
    d_n = norm(ild_front)
    dist_to_ideal = np.sqrt((1 - u_n)**2 + (1 - r_n)**2 + (1 - d_n)**2)
    knee_idx = int(np.argmin(dist_to_ideal))
    knee = archive[knee_idx]

    print(f"  Knee point: α={knee['alpha']:.3f} | "
          f"unexp={unexp_front[knee_idx]:.3f} | "
          f"rel={rel_front[knee_idx]:.3f} | "
          f"ild={ild_front[knee_idx]:.3f} | "
          f"seren={seren_front[knee_idx]:.3f}")

    cd_front = crowding_distance(archive)
    df_front = pd.DataFrame({
        "alpha":          alpha_front,
        "unexpectedness": unexp_front,
        "relevance":      rel_front,
        "ild":            ild_front,
        "serendipity":    seren_front,
        "is_knee":        [i == knee_idx for i in range(len(archive))],
    })
    csv_path = os.path.join(args.out_dir, "pso_moo_3obj_front.csv")
    df_front.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # --- Plots ---
    alpha_dense = np.linspace(0.0, 1.0, 500)
    seren_dense = np.clip(interp_seren(alpha_dense), 0, 1)
    ild_dense   = interp_ild(alpha_dense)
    unexp_dense = interp_unexp(alpha_dense)
    rel_dense   = interp_rel(alpha_dense)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("3-Objective MOPSO Pareto Front: Unexpectedness × Relevance × Diversity",
                 fontsize=14, fontweight="bold")

    # Panel 1: 3D scatter
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    sc = ax1.scatter(unexp_front, rel_front, ild_front,
                     c=seren_front, cmap="RdYlGn", s=80,
                     edgecolors="black", linewidths=0.5, zorder=3)
    ax1.scatter(unexp_front[knee_idx], rel_front[knee_idx], ild_front[knee_idx],
                s=400, marker="*", color="red", edgecolors="black",
                linewidths=1.2, zorder=5, label=f"Knee (α={knee['alpha']:.2f})")
    fig.colorbar(sc, ax=ax1, label="Serendipity", pad=0.1)
    ax1.set_xlabel("Unexpectedness", fontsize=9)
    ax1.set_ylabel("Relevance",      fontsize=9)
    ax1.set_zlabel("ILD",            fontsize=9)
    ax1.set_title(f"3D Pareto Front — {len(archive)} solutions", fontsize=11)
    ax1.legend(fontsize=9)

    # Panel 2: Serendipity curve
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(alpha_dense, seren_dense, color="steelblue", linewidth=2.5,
             label="Serendipity (interpolated)")
    ax2.scatter(alpha_front, seren_front, s=50, color="steelblue",
                edgecolors="black", linewidths=0.5, zorder=3)
    ax2.scatter(knee["alpha"], seren_front[knee_idx], s=400, marker="*",
                color="red", edgecolors="black", linewidths=1.2, zorder=5,
                label=f"Knee (α={knee['alpha']:.2f})")
    ax2.axvline(knee["alpha"], color="red", linestyle=":", linewidth=1.5, alpha=0.6)
    ax2.set_xlabel("α (unexpectedness weight)", fontsize=10)
    ax2.set_ylabel("Serendipity (harmonic mean)", fontsize=10)
    ax2.set_title("Serendipity vs α", fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # Panel 3: Unexpectedness vs Relevance (2D projection)
    ax3 = fig.add_subplot(2, 2, 3)
    sc3 = ax3.scatter(unexp_front, rel_front, c=ild_front, cmap="viridis",
                      s=80, edgecolors="black", linewidths=0.5)
    ax3.plot(unexp_dense, rel_dense, "--", color="gray", linewidth=1, alpha=0.5)
    ax3.scatter(unexp_front[knee_idx], rel_front[knee_idx], s=400, marker="*",
                color="red", edgecolors="black", linewidths=1.2, zorder=5)
    fig.colorbar(sc3, ax=ax3, label="ILD (diversity)")
    ax3.set_xlabel("Unexpectedness", fontsize=10)
    ax3.set_ylabel("Relevance",      fontsize=10)
    ax3.set_title("Projection: Unexpectedness vs Relevance\n(colour = ILD)", fontsize=11)
    ax3.grid(alpha=0.3)

    # Panel 4: Unexpectedness vs ILD (2D projection)
    ax4 = fig.add_subplot(2, 2, 4)
    sc4 = ax4.scatter(unexp_front, ild_front, c=rel_front, cmap="plasma",
                      s=80, edgecolors="black", linewidths=0.5)
    ax4.plot(unexp_dense, ild_dense, "--", color="gray", linewidth=1, alpha=0.5)
    ax4.scatter(unexp_front[knee_idx], ild_front[knee_idx], s=400, marker="*",
                color="red", edgecolors="black", linewidths=1.2, zorder=5)
    fig.colorbar(sc4, ax=ax4, label="Relevance (CF score)")
    ax4.set_xlabel("Unexpectedness", fontsize=10)
    ax4.set_ylabel("ILD (diversity)", fontsize=10)
    ax4.set_title("Projection: Unexpectedness vs Diversity\n(colour = relevance)", fontsize=11)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(args.out_dir, "pso_moo_3obj_pareto.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {png_path}")

    # Summary text
    summary_path = os.path.join(args.out_dir, "pso_moo_3obj_summary.txt")
    with open(summary_path, "w") as f:
        f.write("3-OBJECTIVE MOPSO — PARETO FRONT (IDUN)\n")
        f.write(f"Objectives  : unexpectedness, relevance, ILD (diversity)\n")
        f.write(f"Archive size: {len(archive)} non-dominated solutions\n")
        f.write(f"Knee point  : α={knee['alpha']:.3f}\n")
        f.write(f"  Unexpectedness : {unexp_front[knee_idx]:.4f}\n")
        f.write(f"  Relevance      : {rel_front[knee_idx]:.4f}\n")
        f.write(f"  ILD            : {ild_front[knee_idx]:.4f}\n")
        f.write(f"  Serendipity    : {seren_front[knee_idx]:.4f}\n\n")
        f.write(df_front.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"  Saved: {summary_path}")

    total_min = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n{'='*80}")
    print(f"Completed in {total_min:.1f} minutes")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
