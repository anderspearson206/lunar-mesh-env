# run_built_in_heuristic.py

import numpy as np
import imageio
import torch
import sys
import os
import random
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lunar_mesh_env import LunarRoverMeshEnv, RadioMapModelNN

def set_seed(seed=42):
    """Sets all possible seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")

def get_combined_heuristic_action(agent_id, env):
    """
    Implements a dumb epidemic plus the built in pathfinding heuristic for movement.
    """
    agent = env.agent_map[agent_id]
    move_action = env.heuristic_move_action(agent_id)
    
  
    num_rovers = len(env.possible_agents)
    comm_actions = [0] * (num_rovers + 1)
    packet_sent = False
    target_index = None
    
    # epidemic
    if len(agent.payload_manager.buffer) > 0:
        # just try to send next packet
        next_packet = agent.payload_manager.buffer[0]
        
        for i, target_id in enumerate(env.possible_agents):
            if target_id == agent_id:
                target_index = i
                
            
            target_agent = env.agent_map[target_id]
            # ff the rover is a neighbor and hasn't seen this packet yet
            if target_agent in agent.neighbors:
                if target_agent.id not in next_packet.touched:
                    comm_actions[i] = 1
                    packet_sent = True
                    
        if packet_sent:
            # send back to self as a duplicate to preserve buffer
            comm_actions[target_index] = 1

    # always offload to ba
    if agent.bs_connected:
        comm_actions[-1] = 1

    return [move_action] + comm_actions

def main():
    
    SEED = 1886
    set_seed(SEED)
    
    DATA_ROOT = '/mnt/2ndSSD/rm_raw_for_network'
    # Update this path if you are running locally without the full dataset
    HM_PATH = f'{DATA_ROOT}/hm/hm_18.npy'
    
    MODEL_PATHS = {
        'k2_model': '../RadioLunaDiff/pretrained_models_network/k2unet/best_k2_model.pth',
        'pmnet_model': '../RadioLunaDiff/pretrained_models_network/pmnet/best_pm_model.pt',
        'diffusion_model': '../RadioLunaDiff/pretrained_models_network/diffusion'
    }
    
    print(f"Loading Neural Network Models from {MODEL_PATHS['diffusion_model']}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference Device: {device}")

    
    try:
        global_hm = np.load(HM_PATH)
        print("Terrain loaded successfully.")
    except FileNotFoundError:
        print(f"Warning: Could not find {HM_PATH}. Using flat terrain.")
        global_hm = np.zeros((256, 256))

    
    radio_model = None
    try:
        radio_model = RadioMapModelNN(
            model_paths=MODEL_PATHS,
            heightmap=global_hm,
            env_width=256, 
            env_height=256,
            num_inference_steps=4,
            dummy_mode=False,
            device=device
            
        )
    except Exception as e:
        print(f"CRITICAL WARNING: Radio Model failed to load ({e}).")
        print("Simulation will run in 'Blind' mode (Physics only).")
    RADIO_BIAS = 0.0
    env = LunarRoverMeshEnv(
        hm_path=HM_PATH,
        radio_model=radio_model,
        num_agents=3, 
        render_mode="rgb_array",  
        radio_bias=RADIO_BIAS, 
        seed=SEED
    )
    
    obs, info = env.reset()
    frames = []
    
    print("\n--- Starting Simulation ---")
    print(f"Agents: {env.possible_agents}")
    print("Goal: Rovers will navigate to randomly assigned tasks (task marked as X).")
    
    SIM_STEPS = 300
    start = time.time()
    for step in range(SIM_STEPS):
        
        actions = {}
       
        for agent_id in env.agents:
            actions[agent_id] = get_combined_heuristic_action(agent_id, env)
        
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        if step % 5 == 0:
            avg_rew = sum(rewards.values()) / max(len(rewards), 1)
            
            if env.agents:
                a0 = env.agents[0]
                dist = infos[a0]['pos'] 
                print(f"Step {step}: Avg Reward: {avg_rew:.2f} | {a0} Action: {actions[a0]}")
            
        if not env.agents:
            print("All agents terminated (Tasks Complete or Battery Dead).")
            break
            
    env.close()
    print(f"Simulation completed in {time.time() - start:.2f} seconds.")
    
    if frames:
        out_name = 'marl_task_baseline.gif'
        print(f"\nSaving replay to {out_name} ({len(frames)} frames)...")
        imageio.mimsave(out_name, frames, fps=10)
        imageio.mimwrite(f'marl_task_baseline_{SEED}_{RADIO_BIAS}.mp4', frames, fps=10, output_params=['-vcodec', 'libx264'])
        print("Done!")
    else:
        print("No frames captured.")

if __name__ == "__main__":
    main()