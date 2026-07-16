"""
Plot evaluation metrics from metrics.db, comparing algorithms averaged over episodes.

Groups episodes by run_name, then plots mean ± std band at each step.

Usage:
    python examples/plot_eval_metrics.py
    python examples/plot_eval_metrics.py --db examples/metrics.db --out comparison.png
    python examples/plot_eval_metrics.py --include lozano_ppo epidemic
"""

import argparse
import os
import sqlite3
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

DEFAULT_DB  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics.db")
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_comparison.png")

# Friendly display order and colours
_RUN_ORDER  = ["epidemic_vahdat", "lozano_ppo", "lozano_ddqn"]
_RUN_LABELS = {
    "epidemic_vahdat": "Vahdat Epidemic (BW-constrained)",
    "lozano_ppo":      "Lozano PPO",
    "lozano_ddqn":     "Lozano DDQN",
}
_COLOURS    = ["#e15759", "#4e79a7", "#59a14f"]   # red, blue, green

# Runs whose names contain any of these substrings get dashed lines (RL nav runs)
_DASHED_SUBSTRINGS = ["rl_nav"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run_names(conn: sqlite3.Connection, include: list[str] | None) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT run_name FROM episodes WHERE run_name IS NOT NULL ORDER BY run_name"
    ).fetchall()
    names = [r[0] for r in rows]
    if include:
        names = [n for n in names if any(s in n for s in include)]
    # Sort by preferred order, then alphabetically for unknowns
    ordered = [n for n in _RUN_ORDER if n in names]
    ordered += sorted(n for n in names if n not in _RUN_ORDER)
    return ordered


def load_buffer_metrics(conn: sqlite3.Connection, run_name: str) -> dict:
    """Mean DTN buffer packets per step, averaged across agents and episodes."""
    rows = conn.execute("""
        SELECT s.step, AVG(s.num_packets) as mean_pkts
        FROM step_metrics s
        JOIN episodes e ON s.episode_id = e.id
        WHERE e.run_name = ?
        GROUP BY s.step
        ORDER BY s.step
    """, (run_name,)).fetchall()

    by_step: dict = {}
    for step, mean_pkts in rows:
        by_step[step] = mean_pkts or 0.0

    # Also compute std across episodes for the same steps
    std_rows = conn.execute("""
        SELECT s.step, s.episode_id, AVG(s.num_packets) as ep_mean
        FROM step_metrics s
        JOIN episodes e ON s.episode_id = e.id
        WHERE e.run_name = ?
        GROUP BY s.step, s.episode_id
        ORDER BY s.step
    """, (run_name,)).fetchall()

    by_step_ep: dict = {}
    for step, _, ep_mean in std_rows:
        by_step_ep.setdefault(step, []).append(ep_mean or 0.0)

    steps = np.array(sorted(by_step_ep))
    mean  = np.array([np.mean(by_step_ep[s]) for s in steps])
    std   = np.array([np.std(by_step_ep[s])  for s in steps])
    return {"steps": steps, "mean": mean, "std": std}


def load_env_metrics(conn: sqlite3.Connection, run_name: str) -> dict:
    """
    Returns per-step mean and std across all episodes for a given run_name.
    Metrics: bs_unique_packets, bs_duplicate_packets, avg_datarate_mbps,
             bs_link_ratio, bw_utilization.
    """
    rows = conn.execute("""
        SELECT m.step,
               m.bs_unique_packets,
               m.bs_duplicate_packets,
               m.avg_datarate_mbps,
               m.bs_link_ratio,
               COALESCE(m.bw_utilization, 0.0) as bw_util
        FROM env_step_metrics m
        JOIN episodes e ON m.episode_id = e.id
        WHERE e.run_name = ?
        ORDER BY m.step
    """, (run_name,)).fetchall()

    by_step = defaultdict(lambda: {"unique": [], "dup": [], "dr": [], "ratio": [], "bw": []})
    for step, unique, dup, dr, ratio, bw in rows:
        by_step[step]["unique"].append(unique or 0)
        by_step[step]["dup"].append(dup or 0)
        by_step[step]["dr"].append(dr or 0.0)
        by_step[step]["ratio"].append(ratio or 0.0)
        by_step[step]["bw"].append(bw or 0.0)

    steps = np.array(sorted(by_step))
    def _ms(key):
        vals = [np.mean(by_step[s][key]) for s in steps]
        stds = [np.std(by_step[s][key])  for s in steps]
        return np.array(vals), np.array(stds)

    unique_m, unique_s = _ms("unique")
    dup_m,    dup_s    = _ms("dup")
    dr_m,     dr_s     = _ms("dr")
    ratio_m,  ratio_s  = _ms("ratio")
    bw_m,     bw_s     = _ms("bw")

    # Delivery efficiency = unique / (unique + duplicates), per step
    total = unique_m + dup_m
    eff_m = np.where(total > 0, unique_m / total * 100, np.nan)

    return {
        "steps":      steps,
        "unique_m":   unique_m,   "unique_s":   unique_s,
        "dup_m":      dup_m,      "dup_s":      dup_s,
        "dr_m":       dr_m,       "dr_s":       dr_s,
        "ratio_m":    ratio_m,    "ratio_s":    ratio_s,
        "eff_m":      eff_m,
        "bw_m":       bw_m,       "bw_s":       bw_s,
    }


def load_delivery_ratio(conn: sqlite3.Connection, run_name: str) -> dict:
    """
    Delivery ratio = bs_unique_packets / total packets generated (summed over all agents).
    Standard DTN metric: fraction of generated packets that reached the BS uniquely.
    """
    # Total packets generated per episode per step (sum across agents)
    gen_rows = conn.execute("""
        SELECT s.episode_id, s.step, SUM(s.num_packets_generated) as gen
        FROM step_metrics s
        JOIN episodes e ON s.episode_id = e.id
        WHERE e.run_name = ?
        GROUP BY s.episode_id, s.step
        ORDER BY s.step
    """, (run_name,)).fetchall()

    unique_rows = conn.execute("""
        SELECT m.episode_id, m.step, m.bs_unique_packets
        FROM env_step_metrics m
        JOIN episodes e ON m.episode_id = e.id
        WHERE e.run_name = ?
        ORDER BY m.step
    """, (run_name,)).fetchall()

    gen_by    = defaultdict(dict)   # ep_id → step → gen
    unique_by = defaultdict(dict)   # ep_id → step → unique
    for ep_id, step, gen in gen_rows:
        gen_by[ep_id][step] = gen or 0
    for ep_id, step, u in unique_rows:
        unique_by[ep_id][step] = u or 0

    all_steps = sorted({s for ep in unique_by.values() for s in ep})
    by_step = defaultdict(list)
    for ep_id in unique_by:
        for step in all_steps:
            u = unique_by[ep_id].get(step, 0)
            g = gen_by[ep_id].get(step, 0)
            if g > 0:
                by_step[step].append(u / g * 100)

    steps  = np.array([s for s in all_steps if by_step[s]])
    ratio_m = np.array([np.mean(by_step[s]) for s in steps])
    ratio_s = np.array([np.std(by_step[s])  for s in steps])
    return {"steps": steps, "ratio_m": ratio_m, "ratio_s": ratio_s}


def load_coverage_metrics(conn: sqlite3.Connection, run_name: str) -> dict:
    """Covered pixels per step (union of BS + all rover radio maps above threshold)."""
    try:
        rows = conn.execute("""
            SELECT m.step, m.episode_id, COALESCE(m.covered_pixels, 0) as cov
            FROM env_step_metrics m
            JOIN episodes e ON m.episode_id = e.id
            WHERE e.run_name = ?
            ORDER BY m.step
        """, (run_name,)).fetchall()
    except sqlite3.OperationalError:
        return {"steps": np.array([]), "mean": np.array([]), "std": np.array([])}

    by_step: dict = {}
    for step, _, cov in rows:
        by_step.setdefault(step, []).append(cov or 0)

    steps = np.array(sorted(by_step))
    mean  = np.array([np.mean(by_step[s]) for s in steps])
    std   = np.array([np.std(by_step[s])  for s in steps])
    return {"steps": steps, "mean": mean, "std": std}


def load_goals_metrics(conn: sqlite3.Connection, run_name: str) -> dict:
    """Mean cumulative goals completed per agent per step, averaged across episodes."""
    rows = conn.execute("""
        SELECT s.step, s.episode_id, AVG(s.goals_completed) as ep_mean
        FROM step_metrics s
        JOIN episodes e ON s.episode_id = e.id
        WHERE e.run_name = ?
        GROUP BY s.step, s.episode_id
        ORDER BY s.step
    """, (run_name,)).fetchall()

    by_step: dict = {}
    for step, _, ep_mean in rows:
        by_step.setdefault(step, []).append(ep_mean or 0.0)

    steps = np.array(sorted(by_step))
    mean  = np.array([np.mean(by_step[s]) for s in steps])
    std   = np.array([np.std(by_step[s])  for s in steps])
    return {"steps": steps, "mean": mean, "std": std}


def load_episode_count(conn: sqlite3.Connection, run_name: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE run_name=?", (run_name,)
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def _band(ax, steps, mean, std, colour, label, lw=2.0, linestyle="-"):
    ax.plot(steps, mean, color=colour, label=label, linewidth=lw, linestyle=linestyle)
    ax.fill_between(steps, mean - std, mean + std, color=colour, alpha=0.18)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",  default=DEFAULT_DB)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--include", nargs="+", metavar="SUBSTR",
                        help="Only plot runs whose name contains any of these substrings")
    args = parser.parse_args()

    conn      = sqlite3.connect(args.db)
    run_names = load_run_names(conn, args.include)
    if not run_names:
        print("No runs found — check --db path or --include filter.")
        return

    colours = _COLOURS + plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(5, 2, figsize=(13, 21))
    ax_uniq  = axes[0, 0]   # BS unique packets (primary metric)
    ax_ratio = axes[0, 1]   # Delivery ratio (unique / generated) ← key DTN metric
    ax_eff   = axes[1, 0]   # Delivery efficiency (unique / total received)
    ax_buf   = axes[1, 1]   # DTN buffer packets
    ax_dup   = axes[2, 0]   # Duplicate packets
    ax_link  = axes[2, 1]   # BS link ratio / dwell fraction over episode
    ax_goals = axes[3, 0]   # Goals reached (cumulative, per agent)
    ax_cov   = axes[3, 1]   # Coverage area (pixels above threshold)
    ax_bw    = axes[4, 0]   # Bandwidth utilization (bits sent / BS link capacity)
    axes[4, 1].set_visible(False)

    for idx, run_name in enumerate(run_names):
        col   = colours[idx % len(colours)]
        n_eps = load_episode_count(conn, run_name)
        label = _RUN_LABELS.get(run_name, run_name) + f"  (n={n_eps})"
        ls    = "--" if any(s in run_name for s in _DASHED_SUBSTRINGS) else "-"

        env   = load_env_metrics(conn, run_name)
        dr_m  = load_delivery_ratio(conn, run_name)
        buf   = load_buffer_metrics(conn, run_name)
        goals = load_goals_metrics(conn, run_name)
        cov   = load_coverage_metrics(conn, run_name)

        _band(ax_uniq,  env["steps"],   env["unique_m"], env["unique_s"],  col, label, linestyle=ls)
        _band(ax_ratio, dr_m["steps"],  dr_m["ratio_m"], dr_m["ratio_s"], col, label, linestyle=ls)
        _band(ax_eff,   env["steps"],   env["eff_m"],    np.zeros_like(env["eff_m"]), col, label, linestyle=ls)
        _band(ax_buf,   buf["steps"],   buf["mean"],     buf["std"],       col, label, linestyle=ls)
        _band(ax_dup,   env["steps"],   env["dup_m"],    env["dup_s"],     col, label, linestyle=ls)
        _band(ax_link,  env["steps"],   env["ratio_m"] * 100, env["ratio_s"] * 100, col, label, linestyle=ls)
        _band(ax_goals, goals["steps"], goals["mean"],   goals["std"],     col, label, linestyle=ls)
        if len(cov["steps"]) > 0:
            _band(ax_cov, cov["steps"], cov["mean"], cov["std"], col, label, linestyle=ls)
        bw_mean_pct = float(np.mean(env["bw_m"]) * 100) if len(env["bw_m"]) > 0 else 0.0
        _band(ax_bw, env["steps"], env["bw_m"] * 100, env["bw_s"] * 100, col, label, linestyle=ls)
        ax_bw.axhline(bw_mean_pct, color=col, linewidth=1.0, linestyle=":", alpha=0.8,
                      label=f"{_RUN_LABELS.get(run_name, run_name)} mean={bw_mean_pct:.1f}%")

    conn.close()

    # ── Titles / labels ───────────────────────────────────────────────────
    def _fmt(ax, title, ylabel, pct=False, integer=False):
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Step")
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.3)
        if pct:
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
        elif integer:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    _fmt(ax_uniq,  "BS Unique Packets (cumulative)",               "Unique packets",   integer=True)
    _fmt(ax_ratio, "Delivery Ratio  [key DTN metric]\n"
                   "BS unique / packets generated",                "% delivered",      pct=True)
    _fmt(ax_eff,   "Delivery Efficiency\nunique / (unique + duplicates)", "% unique", pct=True)
    _fmt(ax_buf,   "DTN Buffer — mean packets per agent",          "Packets in buffer")
    _fmt(ax_dup,   "Duplicate Packets at BS (cumulative)",         "Duplicates",       integer=True)
    _fmt(ax_link,  "BS Link Ratio\n(navigation-dependent — identical for all A* runs)",
                   "% agents at BS",  pct=True)
    _fmt(ax_goals, "Goals Reached (cumulative, per agent)",
                   "Goals completed", integer=True)
    _fmt(ax_cov,   "Instantaneous Coverage (BS + rovers above threshold)",
                   "Covered pixels", integer=True)
    _fmt(ax_bw,    "Bandwidth Utilization\nbits sent to BS / total BS link capacity",
                   "% utilization", pct=True)

    n_str = ", ".join(run_names)
    fig.suptitle(f"Algorithm Comparison — {n_str}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
