"""
Plot evaluation metrics from eval.db, comparing all stored episodes.

Usage:
    python examples/plot_eval_metrics.py
    python examples/plot_eval_metrics.py --db examples/eval.db --out eval_comparison.png
"""

import argparse
import os
import sqlite3

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

DEFAULT_DB  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval.db")
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_comparison.png")


def _label_from_checkpoint(ckpt: str, ep_id: int, dummy_mode: int) -> str:
    """Build a human-readable label from the checkpoint path when run_name is absent."""
    trial = os.path.basename(os.path.dirname(ckpt))      # e.g. PPO_lunar_mesh_v1_db173_00000_0_2026-04-05_07-29-29
    ckpt_dir = os.path.basename(ckpt)                    # e.g. checkpoint_000009
    # extract training date: parts after the last plain-digit segment "0_YYYY-MM-DD"
    parts = trial.split("_")
    try:
        # date sits at parts[7]-[9]: 2026, 04, 05
        train_date = f"{parts[7]}-{parts[8]}-{parts[9]}"
    except IndexError:
        train_date = "?"
    try:
        ckpt_iter = int(ckpt_dir.split("_")[-1])
    except (ValueError, IndexError):
        ckpt_iter = -1
    mode = "dummy" if dummy_mode else "real NN"
    return f"Ep{ep_id}  {train_date}  iter {ckpt_iter}  [{mode}]"


def load_episodes(conn: sqlite3.Connection) -> list[dict]:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(episodes)").fetchall()}
    has_run_name   = "run_name"   in cols
    has_dummy_mode = "dummy_mode" in cols

    select = "SELECT id, timestamp, checkpoint, total_steps"
    select += ", run_name"   if has_run_name   else ", NULL"
    select += ", dummy_mode" if has_dummy_mode  else ", 1"
    select += " FROM episodes ORDER BY id"

    episodes = []
    for ep_id, ts, ckpt, total_steps, run_name, dummy_mode in conn.execute(select).fetchall():
        if run_name:
            label = run_name
        else:
            label = _label_from_checkpoint(ckpt, ep_id, dummy_mode)
        episodes.append({"id": ep_id, "label": label, "timestamp": ts[:16],
                         "total_steps": total_steps})
    return episodes


def load_env_metrics(conn: sqlite3.Connection, ep_id: int) -> dict:
    rows = conn.execute(
        "SELECT step, bs_packets_received FROM env_step_metrics "
        "WHERE episode_id=? ORDER BY step",
        (ep_id,)
    ).fetchall()
    steps = np.array([r[0] for r in rows])
    bs_pkts = np.array([r[1] for r in rows])
    return {"steps": steps, "bs_packets_received": bs_pkts}


def load_agent_metrics(conn: sqlite3.Connection, ep_id: int) -> dict:
    """
    Returns per-step aggregates across all agents:
      - total_goals:   sum of goals_completed across agents
      - mean_buffer:   mean buffer_usage across agents
      - mean_packets:  mean num_packets across agents
    """
    rows = conn.execute(
        "SELECT step, SUM(goals_completed), AVG(buffer_usage), AVG(num_packets) "
        "FROM step_metrics WHERE episode_id=? GROUP BY step ORDER BY step",
        (ep_id,)
    ).fetchall()
    steps         = np.array([r[0] for r in rows])
    total_goals   = np.array([r[1] for r in rows])
    mean_buffer   = np.array([r[2] for r in rows])
    mean_packets  = np.array([r[3] for r in rows])
    return {"steps": steps, "total_goals": total_goals,
            "mean_buffer": mean_buffer, "mean_packets": mean_packets}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",  default=DEFAULT_DB)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    conn     = sqlite3.connect(args.db)
    episodes = load_episodes(conn)
    if not episodes:
        print("No episodes found in database.")
        return

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=False)
    ax_bs, ax_goals, ax_buf = axes

    for i, ep in enumerate(episodes):
        col   = colours[i % len(colours)]
        label = ep["label"]

        env_m   = load_env_metrics(conn, ep["id"])
        agent_m = load_agent_metrics(conn, ep["id"])

        # ── BS packets received ───────────────────────────────────────────
        ax_bs.plot(env_m["steps"], env_m["bs_packets_received"],
                   color=col, label=label, linewidth=1.8)

        # ── Total goals reached ───────────────────────────────────────────
        ax_goals.plot(agent_m["steps"], agent_m["total_goals"],
                      color=col, label=label, linewidth=1.8)

        # ── Buffer (mean packets in buffer + mean buffer usage %) ─────────
        ax_buf.plot(agent_m["steps"], agent_m["mean_packets"],
                    color=col, label=label, linewidth=1.8, linestyle="-")
        ax_buf.plot(agent_m["steps"], agent_m["mean_buffer"] * 100,
                    color=col, linewidth=1.0, linestyle="--", alpha=0.5)

    conn.close()

    # ── Formatting ────────────────────────────────────────────────────────
    ax_bs.set_title("Base Station — Packets Received (cumulative)", fontweight="bold")
    ax_bs.set_ylabel("Packets")
    ax_bs.legend(fontsize=8, loc="upper left")
    ax_bs.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_bs.grid(True, alpha=0.3)

    ax_goals.set_title("Goals Reached (all agents combined)", fontweight="bold")
    ax_goals.set_ylabel("Total goals completed")
    ax_goals.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_goals.legend(fontsize=8, loc="upper left")
    ax_goals.grid(True, alpha=0.3)

    ax_buf.set_title("DTN Buffer  —  mean packets (solid) · mean usage % (dashed)",
                     fontweight="bold")
    ax_buf.set_ylabel("Packets  /  Usage %")
    ax_buf.legend(fontsize=8, loc="upper left")
    ax_buf.grid(True, alpha=0.3)
    ax_buf.set_xlabel("Step")

    fig.suptitle("PPO Evaluation Comparison", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
