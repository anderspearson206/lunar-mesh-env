# Runs

Get to goal only
/home/paolo/ray_results/lunar_mesh_ppo/PPO_lunar_mesh_v1_ce859_00000_0_2026-04-03_08-07-18/checkpoint_000009

Both
self.REWARD_PACKET_DELIVERY = 20.0 
/home/paolo/ray_results/lunar_mesh_ppo/PPO_lunar_mesh_v1_f53ce_00000_0_2026-04-04_06-55-37/checkpoint_000009

both
self.REWARD_PACKET_DELIVERY = 10.0 
/home/paolo/ray_results/lunar_mesh_ppo/PPO_lunar_mesh_v1_db173_00000_0_2026-04-05_07-29-29/checkpoint_000009

## Runs with real maps

Get to goal only
<!-- /home/paolo/ray_results/lunar_mesh_ppo/PPO_lunar_mesh_v1_e5f0c_00000_0_2026-04-17_12-16-12/checkpoint_000009 -->
/home/paolo/ray_results/lookup_3agents_goal/PPO_lunar_mesh_lookup_v1_4f3e1_00000_0_2026-05-10_09-57-20/checkpoint_000009

Goal + Ill
/home/paolo/ray_results/lookup_3agents_goal+illum/PPO_lunar_mesh_lookup_v1_75fb6_00000_0_2026-05-19_13-58-17/checkpoint_000009

Both
self.REWARD_PACKET_DELIVERY = 20.0 
<!-- /home/paolo/ray_results/lunar_mesh_ppo_lookup/PPO_lunar_mesh_lookup_v1_67ee7_00000_0_2026-04-23_09-28-05/checkpoint_000009 -->
/home/paolo/ray_results/lookup_3agents_goal+comms20/PPO_lunar_mesh_lookup_v1_8b338_00000_0_2026-05-10_16-11-15/checkpoint_000009

Both
self.REWARD_PACKET_DELIVERY = 10.0 
<!-- /home/paolo/ray_results/lunar_mesh_ppo_lookup/PPO_lunar_mesh_lookup_v1_e037c_00000_0_2026-04-24_09-51-44/checkpoint_000009 -->
/home/paolo/ray_results/lookup_3agents_goal+comms10/PPO_lunar_mesh_lookup_v1_aa74c_00000_0_2026-05-10_20-44-08/checkpoint_000009

Both + illumination
/home/paolo/ray_results/lookup_3agents_goal+comm20+illum/PPO_lunar_mesh_lookup_v1_8a096_00000_0_2026-05-11_06-30-13/checkpoint_000009


# Changes of note
Changing dummy_mode affects the policy evaluation. Most likely cause is that when feeding the real maps to a policy trained with dummy maps the policy sees observations out of distribution.

1. python examples/train_ppo_lookup.py --name "lookup_3agents_v1"
2.python examples/eval_ppo_checkpoint.py --checkpoint <path> --name "lookup_3agents_v1"


# Coverage reward
Done. Here's what was added:

coverage_map — a 256×256 running-max dBm grid, seeded at reset from the BS radio map. Agents earn nothing for coverage the BS already provides.
_compute_coverage_reward() — after every move, reads each agent's current radio map (cache hit since generate_map_batch just ran), updates the max, and splits the reward for newly illuminated pixels equally across active agents.
REWARD_COVERAGE_PER_PIXEL = 0.001 — at 1000 newly covered pixels, this gives +1.0 reward, comparable in scale to the distance shaping reward. Tune up if coverage exploration feels too weak.
The reward naturally decays to zero as the map saturates, so it acts as an exploration bonus early in training without dominating late-episode behavior.