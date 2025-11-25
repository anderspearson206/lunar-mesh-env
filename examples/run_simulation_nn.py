import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import imageio
import sys
import os
# This file should do the same as the run_simulation.py file, 
# but now will generate radiomaps based on RLD model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lunar_mesh_env import CustomEnv
from lunar_mesh_env import RadioMapModelNN as RadioMapModel

if __name__ == "__main__":
    
    print("running script")
    DATA_ROOT = '../../NASA_DCGR_NETWORKING/radio_data_2/radio_data_2'
    
    i, j, k = 18, 29, 15
    HM_PATH = f'{DATA_ROOT}/hm/hm_{i}.npy'
    # TX_A_PATH = f'{DATA_ROOT}/tx/tx_{i}_{j}.npy' 
    # TX_B_PATH = f'{DATA_ROOT}/tx/tx_{i}_{k}.npy'

    # pretrained model paths
    MODEL_PATHS = {
        'k2_model': '../RadioLunaDiff/pretrained_models_network/k2unet/best_k2_model.pth',
        'pmnet_model': '../RadioLunaDiff/pretrained_models_network/pmnet/best_pm_model.pt',
        'diffusion_model': '../RadioLunaDiff/pretrained_models_network/diffusion'
    }

    print("Setting up simulation...")


    DEFAULT_CONFIG = CustomEnv.default_config()
    WIDTH = DEFAULT_CONFIG['width']
    HEIGHT = DEFAULT_CONFIG['height']

    # for now, since this should mimic the static version, we 
    # manually find the coords
    # try:
    #     tx_map_a = np.load(TX_A_PATH)
    #     row_a, col_a = np.unravel_index(np.argmax(tx_map_a), tx_map_a.shape)
    #     x_a = (col_a / (tx_map_a.shape[1] - 1)) * WIDTH
    #     y_a = (row_a / (tx_map_a.shape[0] - 1)) * HEIGHT
    #     agent_a_pos = (x_a, y_a)
    #     print(f"Loaded Agent A position: {agent_a_pos}")

    #     tx_map_b = np.load(TX_B_PATH)
    #     row_b, col_b = np.unravel_index(np.argmax(tx_map_b), tx_map_b.shape)
    #     x_b = (col_b / (tx_map_b.shape[1] - 1)) * WIDTH
    #     y_b = (row_b / (tx_map_b.shape[0] - 1)) * HEIGHT
    #     agent_b_pos = (x_b, y_b)
    #     print(f"Loaded Agent B position: {agent_b_pos}")
        
    # except Exception as e:
    #     print(f"FATAL: Could not load TX maps to find agent positions. Error: {e}")
    #     sys.exit(1)


    try:
        global_heightmap = np.load(HM_PATH)
        print(f"Loaded global heightmap from {HM_PATH}")
    except Exception as e:
        print(f"FATAL: Could not load heightmap {HM_PATH}. Error: {e}")
        sys.exit(1)
        

    print("Loading neural network models...")
    radio_model = RadioMapModel(
        model_paths=MODEL_PATHS,
        heightmap=global_heightmap,
        env_width=WIDTH,
        env_height=HEIGHT
        # You can also pass:
        # num_inference_steps=4,
        # device="cpu" 
    )
    print("Generative radio model is ready.")

    env = CustomEnv(
        config=DEFAULT_CONFIG, 
        render_mode="rgb_array",
        hm_path=HM_PATH,         
        radio_model=radio_model,  
        # agent_a_pos=agent_a_pos, 
        # agent_b_pos=agent_b_pos 
    )

    obs, info = env.reset() 
    frames = [] 
    done = False
    
    num_steps = 5
    print(f"Running simulation for {num_steps} steps and collecting frames...")

    plot_live = False 
    
    for i in range(num_steps):
        print(f"Step {i+1}/{num_steps}")
        
        # Using dummy actions for now
        dummy_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(dummy_action)

        frame = env.render()
        frames.append(frame)
        
        if plot_live:
            plt.imshow(frame)
            display.display(plt.gcf())
            display.clear_output(wait=True)
        
        if terminated or truncated:
            print("Episode finished.")
            done = True
            break

    if plot_live:
        plt.close()

    if frames:
        print("Saving simulation outputs...")
        imageio.mimsave('simulation_nn.gif', frames, fps=5) 
        imageio.mimwrite('simulation_nn.mp4', frames, fps=5, output_params=['-vcodec', 'libx264'])
        print("Done. Saved simulation_nn.gif and simulation.mp4")
    else:
        print("No frames were generated.")

    env.close()