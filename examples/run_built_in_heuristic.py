# run_built_in_heuristic.py

import numpy as np
import imageio
import torch
import sys
import os
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lunar_mesh_env import LunarRoverMeshEnv, RadioMapModelNN

def set_seed(seed=42):
    """Sets all possible seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # OS environment variable for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")

def get_combined_heuristic_action(agent_id, env):
    """
    Combines the environment's built-in movement heuristic (Goal Seeking)
    with a simple communication heuristic (Always try to talk to Base Station).
    """
    move_action = env.heuristic_move_action(agent_id)
    
    num_rovers = len(env.possible_agents)
    
    return [move_action] + [1]*num_rovers + [1]  # Only communicate with Base Station

def main():
    
    SEED = 67
    set_seed(SEED)
    
    DATA_ROOT = '../../NASA_DCGR_NETWORKING/radio_data_2/radio_data_2'
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

    env = LunarRoverMeshEnv(
        hm_path=HM_PATH,
        radio_model=radio_model,
        num_agents=3, 
        render_mode="rgb_array" 
    )
    
    obs, info = env.reset()
    frames = []
    
    print("\n--- Starting Simulation ---")
    print(f"Agents: {env.possible_agents}")
    print("Goal: Rovers will navigate to randomly assigned tasks (task marked as X).")
    
    SIM_STEPS = 5
    
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
    
    
    if frames:
        out_name = 'marl_task_baseline.gif'
        print(f"\nSaving replay to {out_name} ({len(frames)} frames)...")
        imageio.mimsave(out_name, frames, fps=10)
        imageio.mimwrite(f'marl_task_baseline_{SEED}.mp4', frames, fps=10, output_params=['-vcodec', 'libx264'])
        print("Done!")
    else:
        print("No frames captured.")

if __name__ == "__main__":
    main()