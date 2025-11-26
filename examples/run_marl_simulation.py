import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lunar_mesh_env import LunarRoverMeshEnv, RadioMapModelNN

def main():
    DATA_ROOT = '../../NASA_DCGR_NETWORKING/radio_data_2/radio_data_2'
    HM_PATH = f'{DATA_ROOT}/hm/hm_18.npy'
    
    MODEL_PATHS = {
        'k2_model': '../RadioLunaDiff/pretrained_models_network/k2unet/best_k2_model.pth',
        'pmnet_model': '../RadioLunaDiff/pretrained_models_network/pmnet/best_pm_model.pt',
        'diffusion_model': '../RadioLunaDiff/pretrained_models_network/diffusion'
    }
    
    # load models
    print("Loading models...")
    try:
        global_hm = np.load(HM_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {HM_PATH}. Please check path.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    radio_model = RadioMapModelNN(
        model_paths=MODEL_PATHS,
        heightmap=global_hm,
        env_width=256, 
        env_height=256,
        device=device
    )

    # environment set up
    env = LunarRoverMeshEnv(
        hm_path=HM_PATH,
        radio_model=radio_model,
        num_agents=3,
        render_mode="rgb_array"
    )
    
    obs, info = env.reset()
    frames = []
    
    print("Starting simulation...")
    for step in range(50):
        # Sample random actions
        # Action format: [Move (0-4), Comm (0-N)]
        actions = {
            agent: env.action_space(agent).sample() 
            for agent in env.agents
        }
        
        # Step
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        # Render
        frame = env.render()
        frames.append(frame)
        
        if step % 1 == 0:
            print(f"Step {step}: Agent 0 Reward: {rewards['rover_0']:.2f}")
            
        if all(terms.values()) or all(truncs.values()):
            break
            
    env.close()
    
    if frames:
        print(f"Saving GIF ({len(frames)} frames)...")
        imageio.mimsave('marl_simulation.gif', frames, fps=4)
        imageio.mimwrite('simulation_nn.mp4', frames, fps=5, output_params=['-vcodec', 'libx264'])
        print("Done!")

if __name__ == "__main__":
    main()