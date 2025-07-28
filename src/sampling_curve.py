import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.size'] = 11

samples = np.array([100, 200, 300, 400, 500, 600, 740, 'full'])
samples_pct = np.array([10, 20, 30, 40, 50, 60, 75, 100])
x_labels = [f"{n}\n({p}%)" for n, p in zip(samples, samples_pct)]

f1_scores = np.array([0.108, 0.364, 0.537, 0.546, 0.587, 0.591, 0.591, 0.61])
ci_lowers = np.array([0.043, 0.321, 0.384, 0.381, 0.439, 0.442, 0.453, 0.426])
ci_uppers = np.array([0.127, 0.496, 0.548, 0.549, 0.604, 0.602, 0.620, 0.621])

x_range = np.arange(len(samples))
ci_lower_bound = ci_lowers
ci_upper_bound = ci_uppers

plt.figure(figsize=(8, 5))

# Shaded 95% CI
plt.fill_between(
    x_range,
    ci_lower_bound,
    ci_upper_bound,
    color='lightblue',
    alpha=0.4,
    label='95% Confidence Interval'
)

# Mean F1 line & markers
plt.plot(
    x_range, f1_scores,
    '-o', color='royalblue', linewidth=2, markersize=7,
    label='F1-Score'
)
plt.scatter(
    x_range, f1_scores,
    color='white', edgecolor='royalblue', s=70, zorder=3
)

# Labels & Title
plt.title('F1-Score with 95% CI vs. Training Data Size', fontsize=13, weight='bold')
plt.xlabel('Number of Training Samples', fontsize=11)
plt.ylabel('F1-Score', fontsize=11)
plt.ylim(0, 0.7)

plt.xticks(ticks=x_range, labels=x_labels, rotation=0, ha='center', fontsize=10)
plt.yticks(fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig("./figs/learning_curve_f1_vs_samples.png", dpi=300, bbox_inches='tight')
plt.show()