import numpy as np
import torch
import sys
import os
import math
import pygame 


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lunar_mesh_env import LunarRoverMeshEnv, RadioMapModelNN


def get_circular_heuristic_action(agent_id, env, step, radius=60, center=(128, 128)):
    agent = env.agent_map[agent_id]
    
    # 1. Circular movement target
    agent_idx = env.agents.index(agent_id)
    angle_offset = (2 * math.pi / len(env.agents)) * agent_idx
    
    current_angle = (step * 0.05) + angle_offset
    target_x = center[0] + radius * math.cos(current_angle)
    target_y = center[1] + radius * math.sin(current_angle)
    
    vec_x = target_x - agent.x
    vec_y = target_y - agent.y
    
    # 2. Separation and Cohesion
    sep_x, sep_y = 0, 0
    coh_x, coh_y = 0, 0
    
    closest_neighbor_idx = 0 
    max_rssi = -float('inf')
    
    my_radio_map = env._get_radio_map(agent)

    CRASH_DIST = 15     
    SIGNAL_SAFE_MARGIN = -85.0 

    for other_id in env.agents:
        if other_id == agent_id: continue
            
        other = env.agent_map[other_id]
        dx = other.x - agent.x
        dy = other.y - agent.y
        dist = math.sqrt(dx**2 + dy**2)

        # Separation
        if dist < CRASH_DIST:
            weight = (CRASH_DIST - dist) / CRASH_DIST
            sep_x -= (dx / dist) * weight * 150 
            sep_y -= (dy / dist) * weight * 150

        # Cohesion (Signal based)
        rssi = env._get_signal_strength(my_radio_map, other)
        if rssi > max_rssi:
            max_rssi = rssi
            closest_neighbor_idx = env.agents.index(other_id) + 1
    
            if rssi < SIGNAL_SAFE_MARGIN:
                signal_deficit = abs(SIGNAL_SAFE_MARGIN - rssi)
                pull_strength = min(signal_deficit * 10.0, 300.0)
                coh_x += (dx / dist) * pull_strength
                coh_y += (dy / dist) * pull_strength

    final_dx = vec_x*2 + sep_x + coh_x
    final_dy = vec_y*2 + sep_y + coh_y
    
    move_action = 0
    if abs(final_dx) > 1.0 or abs(final_dy) > 1.0:
        if abs(final_dx) > abs(final_dy):
            move_action = 4 if final_dx > 0 else 3
        else:
            move_action = 1 if final_dy > 0 else 2
            
    comm_action = closest_neighbor_idx
    return [move_action, comm_action]

def main():
    # --- Configuration ---
    DATA_ROOT = '../../NASA_DCGR_NETWORKING/radio_data_2/radio_data_2'
    HM_PATH = f'{DATA_ROOT}/hm/hm_18.npy' 
    
    MODEL_PATHS = {
        'k2_model': '../RadioLunaDiff/pretrained_models_network/k2unet/best_k2_model.pth',
        'pmnet_model': '../RadioLunaDiff/pretrained_models_network/pmnet/best_pm_model.pt',
        'diffusion_model': '../RadioLunaDiff/pretrained_models_network/diffusion'
    }

    # --- Setup Radio Model ---
    print("Loading models...")
    try:
        global_hm = np.load(HM_PATH)
    except FileNotFoundError:
        print(f"Warning: Could not find {HM_PATH}. Using flat terrain.")
        global_hm = np.zeros((256, 256))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    radio_model = None
    try:
        radio_model = RadioMapModelNN(
            model_paths=MODEL_PATHS,
            heightmap=global_hm,
            env_width=256, 
            env_height=256,
            device=device
        )
    except Exception as e:
        print(f"Radio model skipped: {e}")

    env = LunarRoverMeshEnv(
        hm_path=HM_PATH,
        radio_model=radio_model,
        num_agents=3,
        render_mode="human"  
    )
    
    obs, info = env.reset()
    
    print("Starting Pygame Simulation...")
    print("Press CTRL+C in terminal or close the window to exit.")

    running = True
    step = 0
    
    try:
        while running:
  
            actions = {}
            for agent_id in env.agents:
                actions[agent_id] = get_circular_heuristic_action(agent_id, env, step)
            
            obs, rewards, terms, truncs, infos = env.step(actions)
            
            env.render()
            

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Logging
            if step % 10 == 0:
                avg_rew = sum(rewards.values()) / len(rewards) if rewards else 0
                print(f"Step {step}: Avg Reward: {avg_rew:.2f}")

            # Termination check
            if all(terms.values()) or all(truncs.values()) or len(env.agents) == 0:
                print("Episode finished.")
                break

            step += 1

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()