#!/usr/bin/env python3
"""
pso_moo_personalized_v2_idun.py
================================
Bi-Objective MOPSO with Personalized α — improved convergence.

Changes from v1 (pso_moo_personalized_idun.py):
  - Warm-start: particles initialised near global optimum (α≈0.35 ± 0.15)
    instead of uniform random [0,1].  This addresses the v1 failure where
    the swarm converged to α≈0.18 — far from the known good region —
    because the 500D search space is too large for random init to find the
    right neighbourhood without a hint.
  - N_PARTICLES  : 50 → 100
  - MAX_ITER     : 200 → 500
  - ARCHIVE_SIZE : 40  → 80
  - Data source  : PURS recommendations (idun_results_purs/)

Everything else (MOPSO mechanics, dominance, crowding distance, mutation,
adaptive inertia) is identical to v1.

Usage:
    python pso_moo_personalized_v2_idun.py \\
        --recs_pkl     idun_results_purs/recommendations_fair_complete.pkl \\
        --single_front idun_results_purs/pso_moo_front.csv \\
        --out_dir      idun_results_purs

Runtime (IDUN, 16 CPUs): ~60 min
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
# Hyperparameters
# ============================================================================
N_PARTICLES    = 100
MAX_ITER       = 500
ARCHIVE_SIZE   = 80
W_START        = 0.9       # adaptive inertia: explore early, exploit late
W_END          = 0.4
C1             = 1.49445
C2             = 1.49445
MUTATION_PROB  = 0.1
MUTATION_SCALE = 0.05      # tighter than v1 (0.10) to stay near good region

N_USERS_SAMPLE = 500

# Warm-start: centre particles around known global optimum
WARM_START_MEAN = 0.35     # global optimum from Experiment 2
WARM_START_STD  = 0.15     # spread wide enough to explore, narrow enough to be useful


# ============================================================================
# MOPSO core  (identical to v1)
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
        values    = np.array([s["objectives"][obj_idx] for s in archive])
        order     = np.argsort(values)
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
        cd       = crowding_distance(archive)
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
    parser = argparse.ArgumentParser(description="Personalized MOPSO v2 (warm-start)")
    parser.add_argument("--recs_pkl",     type=str,
                        default="idun_results_purs/recommendations_fair_complete.pkl")
    parser.add_argument("--single_front", type=str,
                        default="idun_results_purs/pso_moo_front.csv")
    parser.add_argument("--out_dir",      type=str, default="idun_results_purs")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    start_time = datetime.now()

    print("=" * 80)
    print("BI-OBJECTIVE MOPSO — PERSONALIZED α v2 (WARM-START, IDUN)")
    print(f"  Particles: {N_PARTICLES}  |  Iterations: {MAX_ITER}  |  "
          f"Archive: {ARCHIVE_SIZE}  |  Users: {N_USERS_SAMPLE}")
    print(f"  Init: N({WARM_START_MEAN}, {WARM_START_STD}²) per dimension "
          f"[warm-start near global optimum]")
    print("=" * 80)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # -----------------------------------------------------------------------
    # 1. Build per-user PCHIP interpolators
    # -----------------------------------------------------------------------
    print("[1/4] Building per-user objective interpolators...")

    recs           = pickle.load(open(args.recs_pkl, "rb"))
    alpha_values   = sorted(recs.keys())
    all_users_full = list(recs[alpha_values[0]].keys())

    N_USERS = min(N_USERS_SAMPLE, len(all_users_full))
    rng     = np.random.default_rng(args.seed)
    all_users = list(rng.choice(all_users_full, size=N_USERS, replace=False))
    alpha_arr = np.array(alpha_values)

    user_seren_interp = {}
    user_rel_interp   = {}

    for user_id in all_users:
        seren_vals, rel_vals = [], []
        for a in alpha_values:
            rec_list = recs[a][user_id]
            if not rec_list:
                seren_vals.append(0.0)
                rel_vals.append(0.0)
                continue
            s_vals = []
            for r in rec_list:
                U = float(r["distance"])
                R = (float(r["cf_score"]) - 1.0) / 4.0
                denom = U + R
                s_vals.append(2 * U * R / denom if denom > 0 else 0.0)
            seren_vals.append(float(np.mean(s_vals)))
            rel_vals.append(float(np.mean([r["cf_score"] for r in rec_list])))

        user_seren_interp[user_id] = PchipInterpolator(alpha_arr, seren_vals)
        user_rel_interp[user_id]   = PchipInterpolator(alpha_arr, rel_vals)

    print(f"  Interpolators built for {N_USERS:,} users over {len(alpha_values)} α values")
    print(f"  Decision space: {N_USERS}-dimensional")
    print(f"  Grid infeasibility: {len(alpha_values)}^{N_USERS} ≈ "
          f"10^{int(N_USERS * np.log10(len(alpha_values)))} combinations\n")

    def evaluate(alpha_vector):
        serens, rels = [], []
        for i, uid in enumerate(all_users):
            a = float(np.clip(alpha_vector[i], 0.0, 1.0))
            serens.append(float(np.clip(user_seren_interp[uid](a), 0.0, 1.0)))
            rels.append(float(user_rel_interp[uid](a)))
        return float(np.mean(serens)), float(np.mean(rels))

    # -----------------------------------------------------------------------
    # 2. Initialise particles — warm-start near global optimum
    # -----------------------------------------------------------------------
    print("[2/4] Initialising swarm (warm-start)...")

    particles = []
    for _ in range(N_PARTICLES):
        # Draw from N(0.35, 0.15²), clip to [0,1]
        pos = np.clip(
            np.random.normal(WARM_START_MEAN, WARM_START_STD, N_USERS),
            0.0, 1.0
        )
        obj = evaluate(pos)
        particles.append({
            "position":         pos.copy(),
            "velocity":         np.random.uniform(-0.05, 0.05, N_USERS),
            "objectives":       obj,
            "pbest_position":   pos.copy(),
            "pbest_objectives": obj,
        })

    archive = []
    for p in particles:
        archive = update_archive(
            archive,
            {"alpha_vector": p["position"].copy(), "objectives": p["objectives"]},
            ARCHIVE_SIZE,
        )

    print(f"  Initial archive: {len(archive)} non-dominated solutions")
    init_seren = np.mean([p["objectives"][0] for p in particles])
    print(f"  Mean initial serendipity: {init_seren:.4f}  "
          f"(global optimum: 0.4944)\n")

    # -----------------------------------------------------------------------
    # 3. Run MOPSO
    # -----------------------------------------------------------------------
    print(f"[3/4] Running {N_USERS}D MOPSO "
          f"({N_PARTICLES} particles × {MAX_ITER} iterations)...")

    iter_start = datetime.now()
    for iteration in range(MAX_ITER):
        W = W_START - (W_START - W_END) * (iteration / MAX_ITER)

        for p in particles:
            leader = select_leader(archive)

            r1 = np.random.random(N_USERS)
            r2 = np.random.random(N_USERS)

            cognitive = C1 * r1 * (p["pbest_position"]   - p["position"])
            social    = C2 * r2 * (leader["alpha_vector"] - p["position"])
            p["velocity"] = W * p["velocity"] + cognitive + social

            new_pos = p["position"] + p["velocity"]
            if np.random.random() < MUTATION_PROB:
                new_pos += np.random.normal(0, MUTATION_SCALE, N_USERS)
            p["position"]   = np.clip(new_pos, 0.0, 1.0)
            p["objectives"] = evaluate(p["position"])

            if dominates(p["objectives"], p["pbest_objectives"]):
                p["pbest_position"]   = p["position"].copy()
                p["pbest_objectives"] = p["objectives"]
            elif not dominates(p["pbest_objectives"], p["objectives"]):
                if np.random.random() < 0.5:
                    p["pbest_position"]   = p["position"].copy()
                    p["pbest_objectives"] = p["objectives"]

            archive = update_archive(
                archive,
                {"alpha_vector": p["position"].copy(), "objectives": p["objectives"]},
                ARCHIVE_SIZE,
            )

        if (iteration + 1) % 50 == 0 or iteration == MAX_ITER - 1:
            elapsed = (datetime.now() - iter_start).total_seconds()
            rate    = (iteration + 1) / elapsed
            remain  = (MAX_ITER - iteration - 1) / rate if rate > 0 else 0
            best_s  = max(s["objectives"][0] for s in archive)
            print(f"  Iter {iteration+1:4d}/{MAX_ITER} | "
                  f"archive: {len(archive):3d} | "
                  f"best seren: {best_s:.4f} | "
                  f"~{remain/60:.1f} min left")

    print(f"\n  MOPSO complete — {len(archive)} non-dominated solutions")

    # -----------------------------------------------------------------------
    # 4. Post-process, compare, save
    # -----------------------------------------------------------------------
    print("\n[4/4] Post-processing...")

    archive.sort(key=lambda s: s["objectives"][0])
    seren_per   = np.array([s["objectives"][0] for s in archive])
    rel_per     = np.array([s["objectives"][1] for s in archive])
    alpha_means = np.array([s["alpha_vector"].mean() for s in archive])
    alpha_stds  = np.array([s["alpha_vector"].std()  for s in archive])

    df_front = pd.DataFrame({
        "mean_serendipity": seren_per,
        "mean_relevance":   rel_per,
        "mean_alpha":       alpha_means,
        "std_alpha":        alpha_stds,
    })
    csv_path = os.path.join(args.out_dir, "pso_moo_personalized_v2_front.csv")
    df_front.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Compare with single-α front
    single_alpha_front = None
    n_dominated = 0
    if os.path.exists(args.single_front):
        sa = pd.read_csv(args.single_front)
        single_alpha_front = sa.sort_values("unexpectedness")
        n_dominated = sum(
            1 for _, row in single_alpha_front.iterrows()
            if any(dominates(s["objectives"], (row["serendipity"], row["relevance"]))
                   for s in archive)
        )
        pct = n_dominated / len(single_alpha_front) * 100
        print(f"  Per-user front dominates {n_dominated}/{len(single_alpha_front)} "
              f"single-α solutions ({pct:.0f}%)")

    # Representative points
    idx_top = len(archive) - 1
    idx_mid = max(len(archive) // 2, 0)
    idx_bot = 0
    rep_points = [
        ("Serendipity-focused", archive[idx_top], "tab:blue"),
        ("Balanced",            archive[idx_mid], "tab:orange"),
        ("Relevance-focused",   archive[idx_bot], "tab:green"),
    ]

    # Plot
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Personalized α MOPSO v2 — Warm-Start ({N_USERS:,} users, "
        f"{N_PARTICLES} particles × {MAX_ITER} iter)",
        fontsize=13, fontweight="bold",
    )

    ax1 = fig.add_subplot(2, 2, (1, 2))
    if single_alpha_front is not None:
        ax1.plot(single_alpha_front["relevance"], single_alpha_front["serendipity"],
                 "o--", color="steelblue", linewidth=2, markersize=8,
                 label=f"Single global α ({len(single_alpha_front)} solutions)",
                 zorder=2)
    ax1.plot(rel_per, seren_per, "s-", color="crimson", linewidth=2.5, markersize=9,
             label=f"Personalized α — {N_USERS}D MOPSO ({len(archive)} solutions)",
             zorder=3)
    for label, sol, color in rep_points:
        ax1.scatter(sol["objectives"][1], sol["objectives"][0],
                    s=250, marker="*", color=color, edgecolors="black",
                    linewidths=1.2, zorder=5,
                    label=f"{label} (mean α={sol['alpha_vector'].mean():.2f})")
    ax1.set_xlabel("Mean Relevance (CF score)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Mean Serendipity (harmonic mean)", fontsize=12, fontweight="bold")
    ax1.set_title("Pareto Front: Personalized MOPSO v2 vs Single-α MOPSO")
    ax1.legend(fontsize=9, loc="lower left")
    ax1.grid(alpha=0.3)

    for col_idx, (label, sol, color) in enumerate(rep_points):
        ax = fig.add_subplot(2, 3, 4 + col_idx)
        alpha_dist = sol["alpha_vector"]
        ax.hist(alpha_dist, bins=20, color=color, edgecolor="black",
                alpha=0.75, range=(0, 1))
        ax.axvline(alpha_dist.mean(), color="black", linewidth=2,
                   linestyle="--", label=f"Mean={alpha_dist.mean():.2f}")
        ax.axvline(alpha_dist.mean() - alpha_dist.std(), color="black",
                   linewidth=1, linestyle=":")
        ax.axvline(alpha_dist.mean() + alpha_dist.std(), color="black",
                   linewidth=1, linestyle=":",
                   label=f"Std={alpha_dist.std():.2f}")
        ax.set_xlabel("α_u", fontsize=10)
        ax.set_ylabel("Users", fontsize=10)
        ax.set_title(f"{label}\nSeren={sol['objectives'][0]:.3f}  "
                     f"Rel={sol['objectives'][1]:.3f}",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    png_path = os.path.join(args.out_dir, "pso_moo_personalized_v2.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {png_path}")

    # Summary
    summary_path = os.path.join(args.out_dir, "pso_moo_personalized_v2_summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("PERSONALIZED MOPSO v2 — WARM-START (IDUN)\n")
        fh.write(f"Particles: {N_PARTICLES}  |  Iterations: {MAX_ITER}  |  "
                 f"Users: {N_USERS}  |  Archive: {ARCHIVE_SIZE}\n")
        fh.write(f"Init: N({WARM_START_MEAN}, {WARM_START_STD}²)\n\n")
        fh.write(f"Archive size     : {len(archive)} non-dominated solutions\n")
        fh.write(f"Serendipity range: [{seren_per.min():.4f}, {seren_per.max():.4f}]\n")
        fh.write(f"Relevance range  : [{rel_per.min():.4f}, {rel_per.max():.4f}]\n")
        fh.write(f"Mean std(α)      : {alpha_stds.mean():.4f}\n")
        fh.write(f"Global optimum (Exp 2): seren=0.4944 at α=0.35\n")
        fh.write(f"Dominates single-α solutions: "
                 f"{n_dominated}/{len(single_alpha_front) if single_alpha_front is not None else '?'}\n\n")
        fh.write(df_front.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"  Saved: {summary_path}")

    total_min = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n{'='*80}")
    print(f"Completed in {total_min:.1f} minutes")
    print(f"Archive: {len(archive)} solutions  |  "
          f"Seren range: [{seren_per.min():.4f}, {seren_per.max():.4f}]")
    print(f"vs global optimum (Exp 2): seren=0.4944 at α=0.35")
    if seren_per.max() >= 0.4944:
        print("✓ Personalized MOPSO matches or exceeds global optimum")
    else:
        print(f"⚠ Still below global optimum by {0.4944 - seren_per.max():.4f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
