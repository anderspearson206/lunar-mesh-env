import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch
import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lunar_mesh_env import LunarRoverMeshEnv, RadioMapModelNN


def get_circular_heuristic_action(agent_id, env, step, radius=60, center=(128, 128)):
    """
    a simple heuristic that makes agents move in a circular pattern while avoiding collisions
    and trying to maintain good signal strength with neighbors.
    """
    agent = env.agent_map[agent_id]
    
    # circular movement target
    agent_idx = env.agents.index(agent_id)
    angle_offset = (2 * math.pi / len(env.agents)) * agent_idx
    
    current_angle = (step * 0.05) + angle_offset
    target_x = center[0] + radius * math.cos(current_angle)
    target_y = center[1] + radius * math.sin(current_angle)
    
    vec_x = target_x - agent.x
    vec_y = target_y - agent.y
    
    # sepration and cohesion
    sep_x, sep_y = 0, 0
    coh_x, coh_y = 0, 0
    
    closest_neighbor_idx = 0 
    max_rssi = -float('inf')
    max_rssi_neighbor = None
    
    # get the radio map for this agent
    my_radio_map = env._get_radio_map(agent)

    # Thresholds
    # if closer than this distance, apply strong separation force
    CRASH_DIST = 15    
    
    # Signal Thresholds (dBm)
    # if signal is below this, we try to move closer
    SIGNAL_SAFE_MARGIN = -85.0 

    for other_id in env.agents:
        if other_id == agent_id: continue
            
        other = env.agent_map[other_id]
        dx = other.x - agent.x
        dy = other.y - agent.y
        dist = math.sqrt(dx**2 + dy**2)


        # try to avoid collisions
        if dist < CRASH_DIST:
            weight = (CRASH_DIST - dist) / CRASH_DIST
            sep_x -= (dx / dist) * weight * 150 
            sep_y -= (dy / dist) * weight * 150

        # update closer signal neighbor
        rssi = env._get_signal_strength(my_radio_map, other)
        if rssi > max_rssi:
            max_rssi = rssi
            closest_neighbor_idx = env.agents.index(other_id) + 1
    
            if rssi < SIGNAL_SAFE_MARGIN:
                # if best signal is still below safe margin, try to move closer
                signal_deficit = abs(SIGNAL_SAFE_MARGIN - rssi)
                
                pull_strength = min(signal_deficit * 10.0, 300.0)

                coh_x += (dx / dist) * pull_strength
                coh_y += (dy / dist) * pull_strength
            else:
                coh_x = 0
                coh_y = 0
            
            
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
    DATA_ROOT = '../../NASA_DCGR_NETWORKING/radio_data_2/radio_data_2'
    # these paths will need to be changed based on where you store the data
    HM_PATH = f'{DATA_ROOT}/hm/hm_18.npy' 
    
    MODEL_PATHS = {
        'k2_model': '../RadioLunaDiff/pretrained_models_network/k2unet/best_k2_model.pth',
        'pmnet_model': '../RadioLunaDiff/pretrained_models_network/pmnet/best_pm_model.pt',
        'diffusion_model': '../RadioLunaDiff/pretrained_models_network/diffusion'
    }
    
    
    print("Loading models...")
    
    global_hm = None
    try:
        global_hm = np.load(HM_PATH)
    except FileNotFoundError:
        print(f"Warning: Could not find {HM_PATH}. Using flat terrain for demo.")
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
        print(f"Radio model skipped : {e}")

    # env set up
    env = LunarRoverMeshEnv(
        hm_path=HM_PATH,
        radio_model=radio_model,
        num_agents=4,
        render_mode="rgb_array"
    )
    
    obs, info = env.reset()
    frames = []
    
    print("Starting simulation...")
    
    SIM_STEPS = 10
    
    for step in range(SIM_STEPS):
        
        # here we use a simple heuristic for agent actions
        actions = {}
        for agent_id in env.agents:
            actions[agent_id] = get_circular_heuristic_action(agent_id, env, step)
        
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        frame = env.render()
        frames.append(frame)
        
        if step % 10 == 0:
            avg_rew = sum(rewards.values()) / len(rewards)
            print(f"Step {step}: Avg Reward: {avg_rew:.2f} | Action Ex: {actions['rover_0']}")
            
        if all(terms.values()) or all(truncs.values()):
            break
            
    env.close()
    
    if frames:
        print(f"Saving GIF ({len(frames)} frames)...")
        imageio.mimsave('marl_circle_heuristic.gif', frames, fps=10)
        try:
            imageio.mimwrite('marl_circle_heuristic.mp4', frames, fps=10, output_params=['-vcodec', 'libx264'])
        except:
            pass
        print("Done!")

if __name__ == "__main__":
    main()