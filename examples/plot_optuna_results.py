"""Plot Optuna study results across all completed studies."""

import optuna
import matplotlib.pyplot as plt
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Load studies ---
study_1agent = optuna.load_study(
    study_name='mppi_weights',
    storage='sqlite:///examples/optuna_study_1.db',
)
study_2agent = optuna.load_study(
    study_name='mppi_goals_2_agents',
    storage='sqlite:///examples/optuna_study.db',
)
study_3agent = optuna.load_study(
    study_name='mppi_goals_3_agents',
    storage='sqlite:///examples/optuna_study.db',
)

studies = [
    ("1 Agent (100 trials)", study_1agent),
    ("2 Agents (39 trials)", study_2agent),
    ("3 Agents (32 trials)", study_3agent),
]

# --- Style ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
})

COLORS = ["#3B82F6", "#22C55E", "#F97316"]

# ===================================================================
# Figure 1: Optimization History + Best Params Comparison
# ===================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Optuna MPPI Weight Optimization — Goals Objective",
             fontsize=16, fontweight="bold", y=0.98)

# --- 1. Optimization history (goal ratio vs trial) ---
ax = axes[0, 0]
for (label, study), color in zip(studies, COLORS):
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trial_nums = [t.number for t in trials]
    values = [t.value for t in trials]
    ax.scatter(trial_nums, values, alpha=0.5, s=20, color=color, label=label)
    # Running best
    running_best = np.maximum.accumulate(values)
    ax.plot(trial_nums, running_best, color=color, linewidth=2)
ax.set_xlabel("Trial #")
ax.set_ylabel("Goal Ratio (higher = better)")
ax.set_title("Optimization History")
ax.legend(fontsize=10)
ax.set_ylim(-0.05, 1.1)
ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

# --- 2. Best trial params comparison (grouped bar chart) ---
ax = axes[0, 1]
param_names = ['w_comm', 'w_bs', 'w_goal', 'w_energy', 'bs_decay_rate']
short_labels = ['w_comm', 'w_bs', 'w_goal', 'w_energy', 'bs_decay']
x = np.arange(len(param_names))
width = 0.25

for i, ((label, study), color) in enumerate(zip(studies, COLORS)):
    best = study.best_trial
    vals = [best.params[p] for p in param_names]
    bars = ax.bar(x + i * width, vals, width, label=label.split(" (")[0],
                  color=color, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, rotation=45)

ax.set_xticks(x + width)
ax.set_xticklabels(short_labels, fontsize=10)
ax.set_ylabel("Parameter Value")
ax.set_title("Best Trial Parameters")
ax.legend(fontsize=9)

# --- 3. Goal ratio distribution (box plot) ---
ax = axes[1, 0]
data = []
labels_box = []
for (label, study), color in zip(studies, COLORS):
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    values = [t.value for t in trials]
    data.append(values)
    labels_box.append(label.split(" (")[0])

bp = ax.boxplot(data, labels=labels_box, patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel("Goal Ratio")
ax.set_title("Goal Ratio Distribution Across Trials")
ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

# --- 4. Summary table ---
ax = axes[1, 1]
ax.axis('off')
table_data = []
headers = ["Metric", "1 Agent", "2 Agents", "3 Agents"]
best_trials = [s.best_trial for _, s in studies]

table_data.append(["Best Goal Ratio"] + [f"{b.value:.4f}" for b in best_trials])
table_data.append(["Avg Delivery Ratio"] + [
    f"{b.user_attrs.get('avg_delivery_ratio', 0):.4f}" for b in best_trials
])
table_data.append(["w_comm"] + [f"{b.params['w_comm']:.3f}" for b in best_trials])
table_data.append(["w_bs"] + [f"{b.params['w_bs']:.3f}" for b in best_trials])
table_data.append(["w_goal"] + [f"{b.params['w_goal']:.3f}" for b in best_trials])
table_data.append(["w_energy"] + [f"{b.params['w_energy']:.3f}" for b in best_trials])
table_data.append(["bs_decay_rate"] + [f"{b.params['bs_decay_rate']:.3f}" for b in best_trials])

n_complete = []
for _, study in studies:
    n = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_complete.append(str(n))
table_data.append(["Completed Trials"] + n_complete)

table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                 cellLoc='center', colColours=['#E5E7EB'] * 4)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.5)
ax.set_title("Best Trial Summary", pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.94])
out_path = "examples/optuna_results.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved to {out_path}")
