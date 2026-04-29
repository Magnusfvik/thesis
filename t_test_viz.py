import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(12, 7))

alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
means = [0.887, 0.946, 0.812, 0.751, 0.184]
stds = [0.185, 0.166, 0.294, 0.368, 0.289]
labels = ['Pure CF\n(α=0.0)', 'CF-Biased\n(α=0.25)', 'Balanced\n(α=0.5)', 
          'Distance-Biased\n(α=0.75)', 'Pure Distance\n(α=1.0)']

colors = ['lightgray', 'gold', 'skyblue', 'lightgreen', 'salmon']

bars = ax.bar(range(len(alphas)), means, yerr=stds, 
              capsize=8, alpha=0.8, color=colors,
              edgecolor='black', linewidth=1.5)

# Highlight winner
bars[1].set_edgecolor('darkgreen')
bars[1].set_linewidth(3)

# Add significance stars above α=0.25
y_pos = means[1] + stds[1] + 0.08
ax.text(1, y_pos, '***', ha='center', fontsize=20, fontweight='bold', color='darkgreen')
ax.text(1, y_pos + 0.05, 'p<0.001 vs all others', ha='center', fontsize=10, style='italic')

# Add mean values on bars
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.02, f'{m:.3f}', ha='center', fontsize=11, fontweight='bold')

ax.set_xticks(range(len(alphas)))
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('Mean Serendipity Score', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.2)
ax.set_title('Serendipity Comparison with Statistical Significance\n' + 
             '(Error bars = ±1 SD, N=100 users, paired t-tests)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Perfect serendipity')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('serendipity_with_significance.png', dpi=300, bbox_inches='tight')
plt.savefig('serendipity_with_significance.pdf', bbox_inches='tight')
print("✓ Saved: serendipity_with_significance.png/pdf")