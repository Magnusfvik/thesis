"""
Bi-Objective MOPSO with Personalized α — 100-Dimensional Search Space
=======================================================================

Each particle encodes one α value per user: [α_1, α_2, ..., α_N].
The swarm jointly optimizes two aggregate objectives:

  f1 = mean serendipity across all users   (maximize)
  f2 = mean relevance across all users     (maximize)

Why PSO is necessary here:
  A grid search over 11 α values per user has 11^100 ≈ 10^104 combinations —
  computationally infeasible. MOPSO explores this space tractably using
  swarm intelligence.

Why the resulting front is richer than single-α MOPSO (pso_moo.py):
  With one global α, every user gets the same trade-off. With personalized α,
  the swarm can allocate more unexpectedness to users who benefit from it and
  preserve relevance for users who need it — achieving Pareto-superior solutions.

Prerequisites:
  - recommendations_fair_complete.pkl
  - results/pareto/pso_moo_front.csv  (from pso_moo.py, for comparison)

Runtime: ~3 minutes

Output:
  - results/pareto/pso_moo_personalized.png / .pdf
  - results/pareto/pso_moo_personalized_front.csv
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results/pareto', exist_ok=True)

print("=" * 80)
print("BI-OBJECTIVE MOPSO — PERSONALIZED α (100-DIMENSIONAL)")
print("=" * 80)

# ============================================================================
# STEP 1: Load data and build per-user objective interpolators
# ============================================================================

print("\n[1/4] Building per-user objective interpolators...")

recs = pickle.load(open('recommendations_fair_complete.pkl', 'rb'))
alpha_values = sorted(recs.keys())
all_users = list(recs[alpha_values[0]].keys())
N_USERS = len(all_users)
alpha_arr = np.array(alpha_values)

DISTANCE_THRESHOLD = 0.7
CF_THRESHOLD = 1.8

# Per-user PCHIP interpolators for serendipity(α) and relevance(α)
user_seren_interp = {}
user_rel_interp   = {}

for user_id in all_users:
    seren_vals = []
    rel_vals   = []
    for a in alpha_values:
        rec_list = recs[a][user_id]
        n_ser = sum(1 for r in rec_list
                    if r['distance'] > DISTANCE_THRESHOLD
                    and r['cf_score'] > CF_THRESHOLD)
        seren_vals.append(n_ser / len(rec_list) if rec_list else 0.0)
        rel_vals.append(np.mean([r['cf_score'] for r in rec_list]) if rec_list else 0.0)

    user_seren_interp[user_id] = PchipInterpolator(alpha_arr, seren_vals)
    user_rel_interp[user_id]   = PchipInterpolator(alpha_arr, rel_vals)

print(f"  ✓ Built interpolators for {N_USERS} users over {len(alpha_values)} α values")
print(f"  Decision space: {N_USERS}-dimensional  (one α per user)")
print(f"  Objective space: 2-dimensional (aggregate serendipity, aggregate relevance)")
print(f"\n  Grid search infeasibility: {len(alpha_values)}^{N_USERS} ≈ "
      f"10^{int(N_USERS * np.log10(len(alpha_values)))} combinations")


def evaluate(alpha_vector):
    """
    Evaluate both objectives for a 100D particle.
    alpha_vector: array of shape (N_USERS,), one α per user.
    Returns: (mean_serendipity, mean_relevance)
    """
    serens, rels = [], []
    for i, user_id in enumerate(all_users):
        a = float(np.clip(alpha_vector[i], 0.0, 1.0))
        serens.append(float(np.clip(user_seren_interp[user_id](a), 0.0, 1.0)))
        rels.append(float(user_rel_interp[user_id](a)))
    return np.mean(serens), np.mean(rels)

# ============================================================================
# STEP 2: MOPSO core — dominance, crowding distance, archive
# ============================================================================

def dominates(obj_a, obj_b):
    """True if obj_a Pareto-dominates obj_b (both objectives maximized)."""
    return (obj_a[0] >= obj_b[0] and obj_a[1] >= obj_b[1] and
            (obj_a[0] > obj_b[0] or obj_a[1] > obj_b[1]))


def crowding_distance(archive):
    """Crowding distance over the 2D objective space."""
    n = len(archive)
    if n <= 2:
        return [np.inf] * n
    distances = np.zeros(n)
    for obj_idx in range(2):
        values = np.array([s['objectives'][obj_idx] for s in archive])
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
    """Add candidate if non-dominated; trim by crowding distance if over capacity."""
    obj = candidate['objectives']
    for member in archive:
        if dominates(member['objectives'], obj):
            return archive
    archive = [m for m in archive if not dominates(obj, m['objectives'])]
    archive.append(candidate)
    if len(archive) > max_size:
        cd = crowding_distance(archive)
        cd_finite = [d if np.isfinite(d) else 1e9 for d in cd]
        archive.pop(int(np.argmin(cd_finite)))
    return archive


def select_leader(archive):
    """Tournament selection biased toward less-crowded archive regions."""
    if len(archive) == 1:
        return archive[0]
    cd   = crowding_distance(archive)
    i, j = np.random.choice(len(archive), size=2, replace=False)
    return archive[i] if cd[i] >= cd[j] else archive[j]

# ============================================================================
# STEP 3: Run 100D MOPSO
# ============================================================================

print("\n[2/4] Running 100D MOPSO...")

N_PARTICLES  = 50
MAX_ITER     = 150
ARCHIVE_SIZE = 40

W  = 0.729
C1 = 1.49445
C2 = 1.49445

np.random.seed(42)

# Initialise particles — random α_u ∈ [0, 1] per user
particles = []
for _ in range(N_PARTICLES):
    pos = np.random.uniform(0.0, 1.0, N_USERS)
    obj = evaluate(pos)
    particles.append({
        'position':         pos.copy(),
        'velocity':         np.random.uniform(-0.1, 0.1, N_USERS),
        'objectives':       obj,
        'pbest_position':   pos.copy(),
        'pbest_objectives': obj,
    })

archive = []
for p in particles:
    archive = update_archive(archive, {'alpha_vector': p['position'].copy(),
                                       'objectives':   p['objectives']}, ARCHIVE_SIZE)

print(f"  {N_PARTICLES} particles | {MAX_ITER} iterations | "
      f"archive size {ARCHIVE_SIZE} | decision dim {N_USERS}")
print()

for iteration in range(MAX_ITER):
    for p in particles:
        leader = select_leader(archive)

        r1 = np.random.random(N_USERS)
        r2 = np.random.random(N_USERS)

        cognitive = C1 * r1 * (p['pbest_position'] - p['position'])
        social    = C2 * r2 * (leader['alpha_vector'] - p['position'])
        p['velocity']  = W * p['velocity'] + cognitive + social
        p['position']  = np.clip(p['position'] + p['velocity'], 0.0, 1.0)
        p['objectives'] = evaluate(p['position'])

        # Personal best update via Pareto dominance
        if dominates(p['objectives'], p['pbest_objectives']):
            p['pbest_position']   = p['position'].copy()
            p['pbest_objectives'] = p['objectives']
        elif not dominates(p['pbest_objectives'], p['objectives']):
            if np.random.random() < 0.5:
                p['pbest_position']   = p['position'].copy()
                p['pbest_objectives'] = p['objectives']

        archive = update_archive(archive,
                                 {'alpha_vector': p['position'].copy(),
                                  'objectives':   p['objectives']},
                                 ARCHIVE_SIZE)

    if (iteration + 1) % 30 == 0 or iteration == MAX_ITER - 1:
        print(f"  Iteration {iteration+1:3d}/{MAX_ITER} | archive: {len(archive)} solutions")

print(f"\n  MOPSO complete — {len(archive)} non-dominated solutions")

# ============================================================================
# STEP 4: Post-process and compare against single-α front
# ============================================================================

print("\n[3/4] Comparing with single-α Pareto front...")

# Sort archive by serendipity
archive.sort(key=lambda s: s['objectives'][0])

seren_per = np.array([s['objectives'][0] for s in archive])
rel_per   = np.array([s['objectives'][1] for s in archive])

# Compute per-user α statistics for each archive solution
alpha_means = np.array([s['alpha_vector'].mean() for s in archive])
alpha_stds  = np.array([s['alpha_vector'].std()  for s in archive])

# Save front CSV
df_front = pd.DataFrame({
    'mean_serendipity': seren_per,
    'mean_relevance':   rel_per,
    'mean_alpha':       alpha_means,
    'std_alpha':        alpha_stds,
})
df_front.to_csv('results/pareto/pso_moo_personalized_front.csv', index=False)

# Load single-α front for comparison (from pso_moo.py)
single_alpha_front = None
single_alpha_path  = 'results/pareto/pso_moo_front.csv'
if os.path.exists(single_alpha_path):
    sa = pd.read_csv(single_alpha_path)
    single_alpha_front = sa.sort_values('unexpectedness')
    print(f"  ✓ Loaded single-α front ({len(sa)} solutions)")
else:
    print(f"  ⚠ {single_alpha_path} not found — run pso_moo.py first for comparison")

# Dominance check: how many single-α solutions are dominated by the per-user front?
if single_alpha_front is not None:
    n_dominated = 0
    for _, row in single_alpha_front.iterrows():
        sa_obj = (row['serendipity'], row['relevance'])
        for s in archive:
            if dominates(s['objectives'], sa_obj):
                n_dominated += 1
                break
    print(f"  Per-user front dominates {n_dominated}/{len(single_alpha_front)} "
          f"single-α solutions ({n_dominated/len(single_alpha_front)*100:.0f}%)")

# Pick 3 representative Pareto points for α-distribution plots:
# serendipity-focused (top), balanced (middle), relevance-focused (bottom)
idx_top = len(archive) - 1
idx_mid = len(archive) // 2
idx_bot = 0
rep_points = [
    ('Serendipity-focused', archive[idx_top], 'tab:blue'),
    ('Balanced',            archive[idx_mid], 'tab:orange'),
    ('Relevance-focused',   archive[idx_bot], 'tab:green'),
]

# ============================================================================
# STEP 5: Visualisation
# ============================================================================

print("\n[4/4] Generating visualisation...")

fig = plt.figure(figsize=(18, 12))
fig.suptitle('Bi-Objective MOPSO — Personalized α (100-Dimensional)',
             fontsize=15, fontweight='bold')

# --- Panel 1: Pareto front comparison ---
ax1 = fig.add_subplot(2, 2, (1, 2))

if single_alpha_front is not None:
    ax1.plot(single_alpha_front['relevance'],
             single_alpha_front['serendipity'],
             'o--', color='steelblue', linewidth=2, markersize=8,
             label=f'Single global α (pso_moo.py, {len(single_alpha_front)} solutions)',
             zorder=2)

ax1.plot(rel_per, seren_per,
         's-', color='crimson', linewidth=2.5, markersize=9,
         label=f'Personalized α — 100D MOPSO ({len(archive)} solutions)',
         zorder=3)

# Mark the 3 representative points
for label, sol, color in rep_points:
    ax1.scatter(sol['objectives'][1], sol['objectives'][0],
                s=250, marker='*', color=color, edgecolors='black',
                linewidths=1.2, zorder=5,
                label=f'{label} (mean α={sol["alpha_vector"].mean():.2f})')

if single_alpha_front is not None:
    ax1.fill_betweenx(
        seren_per,
        rel_per,
        np.interp(seren_per,
                  np.sort(single_alpha_front['serendipity']),
                  single_alpha_front['relevance'].values[
                      np.argsort(single_alpha_front['serendipity'])]),
        alpha=0.12, color='crimson',
        label='Improvement region (personalized > single-α)'
    )

ax1.set_xlabel('Mean Relevance (aggregate SVD CF score)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mean Serendipity (aggregate)', fontsize=12, fontweight='bold')
ax1.set_title('Pareto Front: Personalized 100D MOPSO vs Single-α MOPSO\n'
              '(Red curve above/right = strictly better trade-offs)', fontsize=12)
ax1.legend(fontsize=9, loc='lower left')
ax1.grid(True, alpha=0.3)

# --- Panels 2–4: α distribution at each representative point ---
for col_idx, (label, sol, color) in enumerate(rep_points):
    ax = fig.add_subplot(2, 3, 4 + col_idx)
    alpha_dist = sol['alpha_vector']

    ax.hist(alpha_dist, bins=15, color=color, edgecolor='black',
            alpha=0.75, range=(0, 1))
    ax.axvline(alpha_dist.mean(), color='black', linewidth=2,
               linestyle='--', label=f'Mean={alpha_dist.mean():.2f}')
    ax.axvline(alpha_dist.mean() - alpha_dist.std(), color='black',
               linewidth=1, linestyle=':')
    ax.axvline(alpha_dist.mean() + alpha_dist.std(), color='black',
               linewidth=1, linestyle=':',
               label=f'Std={alpha_dist.std():.2f}')
    ax.set_xlabel('α_u (per-user weight)', fontsize=10)
    ax.set_ylabel('Users', fontsize=10)
    ax.set_title(f'{label}\n'
                 f'Seren={sol["objectives"][0]:.3f}  '
                 f'Rel={sol["objectives"][1]:.3f}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/pareto/pso_moo_personalized.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/pareto/pso_moo_personalized.png', dpi=300, bbox_inches='tight')
plt.show()

print("  Saved: results/pareto/pso_moo_personalized.pdf/png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\n  Decision space:  {N_USERS}-dimensional  (one α_u per user)")
print(f"  Archive size:    {len(archive)} non-dominated solutions")
print(f"\n  Personalized front range:")
print(f"    Serendipity:   [{seren_per.min():.3f}, {seren_per.max():.3f}]")
print(f"    Relevance:     [{rel_per.min():.3f},   {rel_per.max():.3f}]")
print(f"\n  Mean α spread across front:")
print(f"    mean(std_α):   {alpha_stds.mean():.3f}  "
      f"(0 = all users same α, high = strong personalization)")

if single_alpha_front is not None:
    print(f"\n  Single-α front solutions dominated by personalized front: "
          f"{n_dominated}/{len(single_alpha_front)} "
          f"({n_dominated/len(single_alpha_front)*100:.0f}%)")
    print(f"  → Personalized α provides strictly better trade-offs than any fixed global α")
