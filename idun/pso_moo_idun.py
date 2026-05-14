#!/usr/bin/env python3
"""
pso_moo_idun.py
===============
MOPSO: Multi-Objective PSO for Serendipity Pareto Front (IDUN version).

Discovers the Pareto front between unexpectedness and relevance using
MOPSO with crowding distance archive. Runs on pre-computed recommendations
from generate_complete_idun.py — no SVD retraining needed.

Objectives (both maximized):
  f1 = mean unexpectedness (Jaccard distance from E_u)
  f2 = mean relevance (SVD CF score, normalized)

Serendipity metric (harmonic mean, for reporting):
  S(r) = 2 * U * R / (U + R)
  where U = distance, R = (cf_score - 1) / 4

Usage:
    python pso_moo_idun.py \\
        --recs_pkl idun_results/recommendations_fair_complete.pkl \\
        --out_dir  idun_results

Output:
    pso_moo_front.csv
    pso_moo_pareto.png
    pso_moo_summary.txt
"""

import argparse
import os
import pickle
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


# ============================================================================
# Helpers: dominance, crowding distance, archive
# ============================================================================

def dominates(obj_a, obj_b):
    return (obj_a[0] >= obj_b[0] and obj_a[1] >= obj_b[1] and
            (obj_a[0] > obj_b[0] or obj_a[1] > obj_b[1]))


def crowding_distance(archive):
    n = len(archive)
    if n <= 2:
        return [np.inf] * n
    distances = np.zeros(n)
    for obj_idx in range(2):
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
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MOPSO Pareto front (IDUN)")
    parser.add_argument("--recs_pkl", type=str,
                        default="idun_results/recommendations_fair_complete.pkl")
    parser.add_argument("--out_dir",  type=str, default="idun_results")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    start_time = datetime.now()

    print("=" * 80)
    print("MOPSO: MULTI-OBJECTIVE PSO — UNEXPECTEDNESS vs RELEVANCE (IDUN)")
    print("=" * 80)
    print(f"Started : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Recs pkl: {args.recs_pkl}")
    print()

    # -----------------------------------------------------------------------
    # 1. Build objective interpolators from pre-computed grid
    # -----------------------------------------------------------------------
    print("[1/4] Building objective interpolators from pre-computed grid...")

    if not os.path.exists(args.recs_pkl):
        print(f"  ERROR: {args.recs_pkl} not found. Run generate_complete_idun.py first.")
        return

    recs = pickle.load(open(args.recs_pkl, "rb"))
    alpha_grid = sorted(recs.keys())
    alpha_arr  = np.array(alpha_grid)

    grid_unexp  = []
    grid_rel    = []
    grid_seren  = []

    for alpha in alpha_grid:
        all_unexp, all_rel, all_seren = [], [], []
        for user_id, user_recs in recs[alpha].items():
            if not user_recs:
                continue
            distances  = [r["distance"]  for r in user_recs]
            cf_scores  = [r["cf_score"]  for r in user_recs]
            all_unexp.append(np.mean(distances))
            all_rel.append(np.mean(cf_scores))
            # Harmonic mean serendipity per user
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

    interp_unexp = PchipInterpolator(alpha_arr, unexp_arr)
    interp_rel   = PchipInterpolator(alpha_arr, rel_arr)
    interp_seren = PchipInterpolator(alpha_arr, seren_arr)

    def evaluate(alpha):
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return float(interp_unexp(alpha)), float(interp_rel(alpha))

    print(f"  Grid points : {len(alpha_grid)} alpha values "
          f"({alpha_grid[0]:.2f} to {alpha_grid[-1]:.2f})")
    print(f"  Unexpectedness range : [{unexp_arr.min():.3f}, {unexp_arr.max():.3f}]")
    print(f"  Relevance range      : [{rel_arr.min():.3f},   {rel_arr.max():.3f}]")
    print(f"  Serendipity range    : [{seren_arr.min():.3f},  {seren_arr.max():.3f}]")

    # -----------------------------------------------------------------------
    # 2. Run MOPSO
    # -----------------------------------------------------------------------
    print(f"\n[2/4] Running MOPSO "
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

            cognitive   = C1 * r1 * (p["pbest_position"] - p["position"])
            social      = C2 * r2 * (leader["alpha"]     - p["position"])
            p["velocity"]   = W * p["velocity"] + cognitive + social
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

    print(f"\n  MOPSO complete — {len(archive)} non-dominated solutions")

    # -----------------------------------------------------------------------
    # 3. Post-process: sort, knee point, save CSV
    # -----------------------------------------------------------------------
    print("\n[3/4] Post-processing archive...")

    archive.sort(key=lambda s: s["objectives"][0])

    unexp_front = np.array([s["objectives"][0] for s in archive])
    rel_front   = np.array([s["objectives"][1] for s in archive])
    alpha_front = np.array([s["alpha"]         for s in archive])
    seren_front = np.array([float(np.clip(interp_seren(a), 0, 1)) for a in alpha_front])

    # Knee point: max perpendicular distance from the line between extremes
    u_norm = (unexp_front - unexp_front.min()) / (unexp_front.max() - unexp_front.min() + 1e-9)
    r_norm = (rel_front   - rel_front.min())   / (rel_front.max()   - rel_front.min()   + 1e-9)
    p1 = np.array([u_norm[0],  r_norm[0]])
    p2 = np.array([u_norm[-1], r_norm[-1]])
    line_len = np.linalg.norm(p2 - p1)
    perp = []
    for i in range(len(archive)):
        pt    = np.array([u_norm[i], r_norm[i]])
        cross = abs((p2[0]-p1[0])*(p1[1]-pt[1]) - (p1[0]-pt[0])*(p2[1]-p1[1]))
        perp.append(cross / (line_len + 1e-9))
    knee_idx = int(np.argmax(perp))
    knee = archive[knee_idx]

    print(f"  Knee point: α={knee['alpha']:.3f}  "
          f"unexp={knee['objectives'][0]:.3f}  "
          f"rel={knee['objectives'][1]:.3f}  "
          f"seren={seren_front[knee_idx]:.3f}")

    cd_front  = crowding_distance(archive)
    df_front  = pd.DataFrame({
        "alpha":             alpha_front,
        "unexpectedness":    unexp_front,
        "relevance":         rel_front,
        "serendipity":       seren_front,
        "crowding_distance": cd_front,
        "is_knee":           [i == knee_idx for i in range(len(archive))],
    })
    csv_path = os.path.join(args.out_dir, "pso_moo_front.csv")
    df_front.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # -----------------------------------------------------------------------
    # 4. Plots
    # -----------------------------------------------------------------------
    print("\n[4/4] Generating plots...")

    alpha_dense = np.linspace(0.0, 1.0, 500)
    unexp_dense = interp_unexp(alpha_dense)
    rel_dense   = interp_rel(alpha_dense)
    seren_dense = np.clip(interp_seren(alpha_dense), 0, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("MOPSO Pareto Front: Unexpectedness vs Relevance (IDUN, 5000 users)",
                 fontsize=14, fontweight="bold")

    # Panel 1: Pareto front
    ax1.plot(unexp_dense, rel_dense, color="lightsteelblue", linewidth=1.5,
             linestyle="--", label="Trade-off curve (interpolated)", zorder=1)

    cd_finite = np.array([d if np.isfinite(d) else 1e9 for d in cd_front])
    sc = ax1.scatter(unexp_front, rel_front, c=cd_finite, cmap="plasma",
                     s=120, edgecolors="black", linewidths=0.8, zorder=3,
                     label="MOPSO archive solutions")
    plt.colorbar(sc, ax=ax1, label="Crowding Distance")

    ax1.scatter(knee["objectives"][0], knee["objectives"][1],
                s=500, marker="*", color="red", edgecolors="black",
                linewidths=1.5, zorder=5,
                label=f"Knee point (α={knee['alpha']:.2f})")
    ax1.annotate(f"Knee\n(α={knee['alpha']:.2f})",
                 (knee["objectives"][0], knee["objectives"][1]),
                 xytext=(25, 20), textcoords="offset points", fontsize=10,
                 fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                           edgecolor="red", linewidth=1.5),
                 arrowprops=dict(arrowstyle="->", color="red", lw=1.8))

    ax1.set_xlabel("Mean Unexpectedness (Jaccard distance from E_u)",
                   fontsize=12, fontweight="bold")
    ax1.set_ylabel("Mean Relevance (SVD CF prediction)",
                   fontsize=12, fontweight="bold")
    ax1.set_title(f"Pareto Front — {len(archive)} Non-Dominated Solutions", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Panel 2: Serendipity curve
    ax2.plot(alpha_dense, seren_dense, color="steelblue", linewidth=2.5,
             label="Serendipity (harmonic mean, interpolated)")
    ax2.scatter(alpha_front, seren_front, s=60, color="steelblue",
                edgecolors="black", linewidths=0.6, zorder=3,
                label="MOPSO archive solutions")
    ax2.scatter(knee["alpha"], seren_front[knee_idx], s=500, marker="*",
                color="red", edgecolors="black", linewidths=1.5, zorder=5,
                label=f"Knee (α={knee['alpha']:.2f})")
    ax2.axvline(knee["alpha"], color="red", linestyle=":", linewidth=1.8, alpha=0.6)

    ax2.set_xlabel("α (weight on unexpectedness)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Serendipity (harmonic mean)", fontsize=12, fontweight="bold")
    ax2.set_title("Serendipity Across the Pareto Front", fontsize=13)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(0, max(seren_dense.max(), seren_front.max()) * 1.1)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(args.out_dir, "pso_moo_pareto.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {png_path}")

    # Summary text
    summary_path = os.path.join(args.out_dir, "pso_moo_summary.txt")
    with open(summary_path, "w") as f:
        f.write("MOPSO — PARETO FRONT (IDUN)\n")
        f.write(f"Archive size : {len(archive)} non-dominated solutions\n")
        f.write(f"Knee point   : α={knee['alpha']:.3f}\n")
        f.write(f"  Unexpectedness : {knee['objectives'][0]:.4f}\n")
        f.write(f"  Relevance      : {knee['objectives'][1]:.4f}\n")
        f.write(f"  Serendipity    : {seren_front[knee_idx]:.4f}\n\n")
        f.write(df_front.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"  Saved: {summary_path}")

    total_min = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n{'='*80}")
    print(f"Completed in {total_min:.1f} minutes")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
