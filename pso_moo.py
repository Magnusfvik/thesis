"""
MOPSO: Multi-Objective PSO for Serendipity Pareto Front
========================================================

Discovers the Pareto front between unexpectedness and relevance using
Multi-Objective Particle Swarm Optimization (MOPSO) with crowding distance.

Key improvements over pareto_analysis.py:
  - Particles explore alpha space continuously (not fixed grid)
  - Pareto archive with crowding distance:
      * Removes the most-crowded solution when archive is full
      * Selects leaders from less-crowded archive regions
  - Produces a well-spread, smooth Pareto front

Prerequisites:
  - recommendations_fair_complete.pkl
  - user_E_u.pkl

Runtime: ~2 minutes

Output:
  - results/pareto/pso_moo_pareto.png / .pdf
  - results/pareto/pso_moo_front.csv
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
print("MOPSO: MULTI-OBJECTIVE PSO — UNEXPECTEDNESS vs RELEVANCE")
print("=" * 80)

# ============================================================================
# STEP 1: Build objective functions from pre-computed recommendations
# ============================================================================

print("\n[1/4] Building objective interpolators from pre-computed grid...")

recs = pickle.load(open('recommendations_fair_complete.pkl', 'rb'))
alpha_grid = sorted(recs.keys())

grid_unexp = []
grid_rel = []
grid_seren = []

for alpha in alpha_grid:
    all_unexp, all_rel, all_seren = [], [], []
    for user_id, user_recs in recs[alpha].items():
        all_unexp.append(np.mean([r['distance'] for r in user_recs]))
        all_rel.append(np.mean([r['cf_score'] for r in user_recs]))
        n_ser = sum(1 for r in user_recs if r['distance'] > 0.7 and r['cf_score'] > 1.8)
        all_seren.append(n_ser / len(user_recs))
    grid_unexp.append(np.mean(all_unexp))
    grid_rel.append(np.mean(all_rel))
    grid_seren.append(np.mean(all_seren))

alpha_arr = np.array(alpha_grid)
unexp_arr = np.array(grid_unexp)
rel_arr   = np.array(grid_rel)
seren_arr = np.array(grid_seren)

# PchipInterpolator preserves monotonicity — avoids ringing artifacts
interp_unexp = PchipInterpolator(alpha_arr, unexp_arr)
interp_rel   = PchipInterpolator(alpha_arr, rel_arr)
interp_seren = PchipInterpolator(alpha_arr, seren_arr)

def evaluate(alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return float(interp_unexp(alpha)), float(interp_rel(alpha))

print(f"  Grid points: {len(alpha_grid)} alpha values ({alpha_grid[0]:.2f} to {alpha_grid[-1]:.2f})")
print(f"  Unexpectedness range: [{unexp_arr.min():.3f}, {unexp_arr.max():.3f}]")
print(f"  Relevance range:      [{rel_arr.min():.3f}, {rel_arr.max():.3f}]")

# ============================================================================
# STEP 2: MOPSO core — dominance, crowding distance, archive
# ============================================================================

def dominates(obj_a, obj_b):
    """True if obj_a dominates obj_b (both objectives are maximized)."""
    return (obj_a[0] >= obj_b[0] and obj_a[1] >= obj_b[1] and
            (obj_a[0] > obj_b[0] or obj_a[1] > obj_b[1]))


def crowding_distance(archive):
    """
    Compute crowding distance for each solution in the archive.
    Boundary solutions receive infinite distance.
    Interior solutions: sum of normalized objective range spans.
    """
    n = len(archive)
    if n <= 2:
        return [np.inf] * n

    distances = np.zeros(n)

    for obj_idx in range(2):
        values = np.array([s['objectives'][obj_idx] for s in archive])
        order = np.argsort(values)
        sorted_vals = values[order]

        obj_range = sorted_vals[-1] - sorted_vals[0]
        if obj_range == 0:
            continue

        distances[order[0]]  = np.inf
        distances[order[-1]] = np.inf

        for i in range(1, n - 1):
            distances[order[i]] += (sorted_vals[i + 1] - sorted_vals[i - 1]) / obj_range

    return distances.tolist()


def update_archive(archive, candidate, max_size):
    """
    Add candidate to archive if non-dominated.
    Remove solutions dominated by candidate.
    If archive exceeds max_size, evict the most-crowded solution.
    """
    obj = candidate['objectives']

    # Drop if dominated by any existing member
    for member in archive:
        if dominates(member['objectives'], obj):
            return archive

    # Remove members that the candidate dominates
    archive = [m for m in archive if not dominates(obj, m['objectives'])]

    archive.append(candidate)

    # Trim to max_size by evicting the solution with smallest crowding distance
    if len(archive) > max_size:
        cd = crowding_distance(archive)
        # Replace inf with a large finite number for argmin
        cd_finite = [d if np.isfinite(d) else 1e9 for d in cd]
        evict_idx = int(np.argmin(cd_finite))
        archive.pop(evict_idx)

    return archive


def select_leader(archive):
    """
    Tournament selection from archive biased toward less-crowded regions:
    sample two candidates, return the one with higher crowding distance.
    """
    if len(archive) == 1:
        return archive[0]

    cd = crowding_distance(archive)
    idx1, idx2 = np.random.choice(len(archive), size=2, replace=False)

    return archive[idx1] if cd[idx1] >= cd[idx2] else archive[idx2]


# ============================================================================
# STEP 3: Run MOPSO
# ============================================================================

print("\n[2/4] Running MOPSO...")

N_PARTICLES   = 30
MAX_ITER      = 150
ARCHIVE_SIZE  = 40

# Standard constriction coefficient settings
W  = 0.729
C1 = 1.49445   # cognitive (attraction to personal best)
C2 = 1.49445   # social    (attraction to archive leader)

np.random.seed(42)

# Initialise particles
particles = []
for _ in range(N_PARTICLES):
    pos = np.random.random()
    obj = evaluate(pos)
    particles.append({
        'position':        pos,
        'velocity':        np.random.uniform(-0.1, 0.1),
        'objectives':      obj,
        'pbest_position':  pos,
        'pbest_objectives': obj,
    })

# Initialise archive from starting positions
archive = []
for p in particles:
    archive = update_archive(archive, {'alpha': p['position'], 'objectives': p['objectives']}, ARCHIVE_SIZE)

print(f"  {N_PARTICLES} particles | {MAX_ITER} iterations | archive size {ARCHIVE_SIZE}")
print()

for iteration in range(MAX_ITER):
    for p in particles:
        leader = select_leader(archive)

        r1 = np.random.random()
        r2 = np.random.random()

        # Velocity update (standard PSO constriction)
        cognitive = C1 * r1 * (p['pbest_position'] - p['position'])
        social    = C2 * r2 * (leader['alpha']     - p['position'])
        p['velocity'] = W * p['velocity'] + cognitive + social

        # Position update — clamp to [0, 1]
        p['position'] = float(np.clip(p['position'] + p['velocity'], 0.0, 1.0))
        p['objectives'] = evaluate(p['position'])

        # Update personal best using Pareto dominance
        if dominates(p['objectives'], p['pbest_objectives']):
            p['pbest_position']  = p['position']
            p['pbest_objectives'] = p['objectives']
        elif not dominates(p['pbest_objectives'], p['objectives']):
            # Neither dominates — accept randomly (maintains diversity)
            if np.random.random() < 0.5:
                p['pbest_position']  = p['position']
                p['pbest_objectives'] = p['objectives']

        # Update archive
        archive = update_archive(
            archive,
            {'alpha': p['position'], 'objectives': p['objectives']},
            ARCHIVE_SIZE
        )

    if (iteration + 1) % 30 == 0 or iteration == MAX_ITER - 1:
        print(f"  Iteration {iteration + 1:3d}/{MAX_ITER} | archive size: {len(archive)}")

print(f"\n  MOPSO complete. {len(archive)} non-dominated solutions in archive.")

# ============================================================================
# STEP 4: Post-process archive
# ============================================================================

# Sort archive by unexpectedness (x-axis)
archive.sort(key=lambda s: s['objectives'][0])

unexp_front = np.array([s['objectives'][0] for s in archive])
rel_front   = np.array([s['objectives'][1] for s in archive])
alpha_front = np.array([s['alpha']         for s in archive])

# Crowding distances for colouring
cd_front = crowding_distance(archive)
cd_finite = np.array([d if np.isfinite(d) else max(
    [x for x in cd_front if np.isfinite(x)], default=1.0
) for d in cd_front])

# Knee point: solution furthest from the line connecting the two extremes
# Normalise objectives to [0, 1] before computing distance
u_norm = (unexp_front - unexp_front.min()) / (unexp_front.max() - unexp_front.min() + 1e-9)
r_norm = (rel_front   - rel_front.min())   / (rel_front.max()   - rel_front.min()   + 1e-9)

# Line from extreme 0 (low unexp, high rel) to extreme 1 (high unexp, low rel)
# Vector along line: (1, -1) normalised
p1 = np.array([u_norm[0],  r_norm[0]])
p2 = np.array([u_norm[-1], r_norm[-1]])
line_vec = p2 - p1
line_len = np.linalg.norm(line_vec)

perp_distances = []
for i in range(len(archive)):
    pt = np.array([u_norm[i], r_norm[i]])
    cross = abs((p2[0] - p1[0]) * (p1[1] - pt[1]) - (p1[0] - pt[0]) * (p2[1] - p1[1]))
    perp_distances.append(cross / (line_len + 1e-9))

knee_idx = int(np.argmax(perp_distances))
knee = archive[knee_idx]

print(f"\n  Knee point:")
print(f"    α = {knee['alpha']:.3f}")
print(f"    Unexpectedness = {knee['objectives'][0]:.3f}")
print(f"    Relevance      = {knee['objectives'][1]:.3f}")
print(f"    Serendipity    = {float(interp_seren(knee['alpha'])):.3f}")

# Save archive to CSV
df_front = pd.DataFrame({
    'alpha':             alpha_front,
    'unexpectedness':    unexp_front,
    'relevance':         rel_front,
    'serendipity':       [float(interp_seren(a)) for a in alpha_front],
    'crowding_distance': cd_front,
    'is_knee':           [i == knee_idx for i in range(len(archive))]
})
df_front.to_csv('results/pareto/pso_moo_front.csv', index=False)
print(f"\n  Saved: results/pareto/pso_moo_front.csv")

# ============================================================================
# STEP 5: Visualisation
# ============================================================================

print("\n[4/4] Generating visualisation...")

# Dense alpha range for smooth background curve
alpha_dense = np.linspace(0.0, 1.0, 500)
unexp_dense = interp_unexp(alpha_dense)
rel_dense   = interp_rel(alpha_dense)
seren_dense = interp_seren(alpha_dense)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('MOPSO Pareto Front: Unexpectedness vs Relevance', fontsize=15, fontweight='bold')

# --- Panel 1: Pareto Front ---

# Background: interpolated trade-off curve (faint)
ax1.plot(unexp_dense, rel_dense,
         color='lightsteelblue', linewidth=1.5, linestyle='--',
         label='Trade-off curve (interpolated)', zorder=1)

# Pareto archive points coloured by crowding distance
sc = ax1.scatter(unexp_front, rel_front,
                 c=cd_finite, cmap='plasma',
                 s=120, edgecolors='black', linewidths=0.8, zorder=3,
                 label='MOPSO archive solutions')

cbar = plt.colorbar(sc, ax=ax1)
cbar.set_label('Crowding Distance\n(higher = better spread)', fontsize=10)

# Knee point
ax1.scatter(knee['objectives'][0], knee['objectives'][1],
            s=500, marker='*', color='red', edgecolors='black', linewidths=1.5,
            zorder=5, label=f"Knee point (α={knee['alpha']:.2f})")

ax1.annotate(f"Knee\n(α={knee['alpha']:.2f})",
             (knee['objectives'][0], knee['objectives'][1]),
             xytext=(25, 20), textcoords='offset points',
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='red', linewidth=1.5),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.8))

# Annotate extremes
ax1.annotate(f'Pure CF\n(α=0.0)',
             (unexp_front[0], rel_front[0]),
             xytext=(-65, 10), textcoords='offset points', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
             arrowprops=dict(arrowstyle='->', lw=1.5))

ax1.annotate(f'Pure Distance\n(α=1.0)',
             (unexp_front[-1], rel_front[-1]),
             xytext=(10, -35), textcoords='offset points', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
             arrowprops=dict(arrowstyle='->', lw=1.5))

ax1.set_xlabel('Mean Unexpectedness (Jaccard distance from E_u)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mean Relevance (SVD CF prediction)', fontsize=12, fontweight='bold')
ax1.set_title(f'Pareto Front — {len(archive)} Non-Dominated Solutions', fontsize=13)
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)

# --- Panel 2: Serendipity curve with knee ---

ax2.plot(alpha_dense, seren_dense,
         color='steelblue', linewidth=2.5, label='Serendipity (interpolated)')

# Mark all archive solutions on serendipity curve
ax2.scatter(alpha_front,
            [float(interp_seren(a)) for a in alpha_front],
            s=60, color='steelblue', edgecolors='black', linewidths=0.6, zorder=3,
            label='MOPSO archive solutions')

# Mark knee
knee_seren = float(interp_seren(knee['alpha']))
ax2.scatter(knee['alpha'], knee_seren,
            s=500, marker='*', color='red', edgecolors='black', linewidths=1.5,
            zorder=5, label=f"Knee point (α={knee['alpha']:.2f})")

ax2.axvline(knee['alpha'], color='red', linestyle=':', linewidth=1.8, alpha=0.6)

ax2.set_xlabel('α (weight on unexpectedness)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Serendipity', fontsize=12, fontweight='bold')
ax2.set_title('Serendipity Across the Pareto Front', fontsize=13)
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(0, 1.08)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/pareto/pso_moo_pareto.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/pareto/pso_moo_pareto.png', dpi=300, bbox_inches='tight')
plt.show()

print("  Saved: results/pareto/pso_moo_pareto.pdf/png")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"  Archive size:     {len(archive)} non-dominated solutions")
print(f"  Knee point:       α={knee['alpha']:.3f}")
print(f"    Unexpectedness: {knee['objectives'][0]:.3f}")
print(f"    Relevance:      {knee['objectives'][1]:.3f}")
print(f"    Serendipity:    {knee_seren:.3f}")
