"""Generate presentation bar charts from bs_decay_rate experiment results."""

import matplotlib.pyplot as plt
import numpy as np

# --- Data from results_bs_decay_rate.md (deduplicated) ---
decay_rates = [1.0, 0.9, 0.8, 0.7, 0.5]
labels = ["1.0\n(no decay)", "0.9", "0.8", "0.7", "0.5"]

goals_completed = [2, 5, 5, 5, 5]
goals_total = 5
unique_pkts_received = [162, 53, 50, 54, 61]
pkts_generated = [163, 84, 87, 69, 89]
distance_m = [1200.0, 620.0, 652.0, 528.0, 680.0]
buffer_stored = [1, 29, 36, 15, 28]
delivery_ratio = [r / g * 100 for r, g in zip(unique_pkts_received, pkts_generated)]

# --- Style ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
})

BLUE = "#3B82F6"
GREEN = "#22C55E"
ORANGE = "#F97316"
RED = "#EF4444"
PURPLE = "#A855F7"

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Effect of BS Decay Rate on Rover Performance\n"
             r"($w_{bs}=0.25$, $w_{goal}=1.0$, seed=1886, 300 steps)",
             fontsize=16, fontweight="bold", y=0.98)

x = np.arange(len(labels))
bar_width = 0.55

# --- 1. Goals Completed ---
ax = axes[0, 0]
colors = [RED if g < goals_total else GREEN for g in goals_completed]
bars = ax.bar(x, goals_completed, bar_width, color=colors, edgecolor="white", linewidth=1.2)
ax.set_ylabel("Goals Completed")
ax.set_title("Mission Completion")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("BS Decay Rate")
ax.set_ylim(0, goals_total + 0.8)
ax.axhline(y=goals_total, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.text(len(labels) - 0.55, goals_total + 0.1, f"target = {goals_total}", color="gray", fontsize=10, ha="right")
for bar, val in zip(bars, goals_completed):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
            f"{val}/{goals_total}", ha="center", va="bottom", fontweight="bold", fontsize=12)

# --- 2. Unique Packets Received by BS ---
ax = axes[0, 1]
bars = ax.bar(x, unique_pkts_received, bar_width, color=BLUE, edgecolor="white", linewidth=1.2)
ax.set_ylabel("Unique Packets")
ax.set_title("BS Packets Received")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("BS Decay Rate")
for bar, val in zip(bars, unique_pkts_received):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            str(val), ha="center", va="bottom", fontweight="bold", fontsize=12)

# --- 3. Distance Traveled ---
ax = axes[1, 0]
bars = ax.bar(x, distance_m, bar_width, color=ORANGE, edgecolor="white", linewidth=1.2)
ax.set_ylabel("Distance (m)")
ax.set_title("Total Distance Traveled")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("BS Decay Rate")
for bar, val in zip(bars, distance_m):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            f"{val:.0f}", ha="center", va="bottom", fontweight="bold", fontsize=12)

# --- 4. Packet Delivery Ratio ---
ax = axes[1, 1]
colors4 = [PURPLE] * len(delivery_ratio)
bars = ax.bar(x, delivery_ratio, bar_width, color=colors4, edgecolor="white", linewidth=1.2)
ax.set_ylabel("Delivery Ratio (%)")
ax.set_title("Packet Delivery Ratio")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("BS Decay Rate")
ax.set_ylim(0, 110)
for bar, val in zip(bars, delivery_ratio):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.93])
out_path = "examples/bs_decay_rate_results.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved to {out_path}")
# plt.show()  # uncomment for interactive viewing
