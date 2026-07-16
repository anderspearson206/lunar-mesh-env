#!/usr/bin/env bash
# Run all three comparison evaluations and write results to a single DB.
# Usage: bash examples/run_comparison.sh
#
# Set the two checkpoint paths below before running.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB="$SCRIPT_DIR/metrics.db"
EPISODES=10

PPO_CKPT="/home/paolo/ray_results/lozano_ppo/PPO_lunar_mesh_lozano_v1_9fc57_00000_0_2026-07-01_14-05-44/checkpoint_000000"   # e.g. ~/ray_results/lozano_ppo/PPO_.../checkpoint_000200
DQN_CKPT="/home/paolo/ray_results/lozano_ddqn/DQN_lunar_mesh_lozano_v1_a58a1_00000_0_2026-07-01_20-10-58/checkpoint_000000"   # e.g. ~/ray_results/lozano_ddqn/DQN_.../checkpoint_000050

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [[ -z "$PPO_CKPT" || -z "$DQN_CKPT" ]]; then
    echo "ERROR: set PPO_CKPT and DQN_CKPT at the top of this script."
    exit 1
fi

cd "$SCRIPT_DIR/.."

echo "================================================================"
echo "  DB: $DB"
echo "  Episodes per run: $EPISODES  (seeds 199-$((199 + EPISODES - 1)))"
echo "================================================================"

# ---------------------------------------------------------------------------
# 1. Vahdat epidemic (BW-constrained) — runs on the same Lozano env as PPO/DQN
#    so packet generation (rate mode) and env are identical.
#    The PPO checkpoint is loaded but its comm actions are irrelevant:
#    epidemic routing floods all neighbor pairs via _apply_routing_protocol.
# ---------------------------------------------------------------------------
# echo ""
# echo "[1/3] Vahdat epidemic baseline (Lozano env, rate mode)..."
# conda run -n lunar_mesh python examples/eval_ppo_checkpoint.py \
#     --checkpoint "$PPO_CKPT" \
#     --model lozano \
#     --routing epidemic \
#     --routing-bw-limit \
#     --episodes "$EPISODES" \
#     --name epidemic_vahdat \
#     --db "$DB"

# ---------------------------------------------------------------------------
# 2. Lozano PPO
# ---------------------------------------------------------------------------
echo ""
echo "[2/3] Lozano PPO..."
conda run -n lunar_mesh python examples/eval_ppo_checkpoint.py \
    --checkpoint "$PPO_CKPT" \
    --model lozano \
    --algo ppo \
    --episodes "$EPISODES" \
    --name lozano_ppo_nav \
    --db "$DB"

# ---------------------------------------------------------------------------
# 3. Lozano DDQN
# ---------------------------------------------------------------------------
echo ""
echo "[3/3] Lozano DDQN..."
conda run -n lunar_mesh python examples/eval_ppo_checkpoint.py \
    --checkpoint "$DQN_CKPT" \
    --model lozano \
    --algo dqn \
    --episodes "$EPISODES" \
    --name lozano_ddqn_nav \
    --db "$DB"

echo ""
echo "Done. Results in $DB"
echo "  SELECT run_name, AVG(total_steps), COUNT(*) FROM episodes GROUP BY run_name;"
