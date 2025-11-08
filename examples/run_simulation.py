import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import imageio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lunar_mesh_env import CustomEnv, RadioMapModel



if __name__ == "__main__":
    
    DATA_ROOT = '../../NASA_DCGR_NETWORKING/radio_data_2/radio_data_2'
    
    i, j, k = 18, 29, 15
    HM_PATH = f'{DATA_ROOT}/hm/hm_{i}.npy'
    TX_PATH = f'{DATA_ROOT}/tx/tx_{i}_{j}.npy' 
    RM_PATH = f'{DATA_ROOT}/rm58/rm_{i}_{j}.npy' 

    RM_B_PATH = f'{DATA_ROOT}/rm58/rm_{i}_{k}.npy'
    TX_B_PATH = f'{DATA_ROOT}/tx/tx_{i}_{k}.npy' 

    RM_415_PATH = f'{DATA_ROOT}/rm415/rm_{i}_{j}.npy'
    RM_B_415_PATH = f'{DATA_ROOT}/rm415/rm_{i}_{k}.npy'

    print("Setting up simulation...")

    DEFAULT_CONFIG = CustomEnv.default_config()
    WIDTH = DEFAULT_CONFIG['width']
    HEIGHT = DEFAULT_CONFIG['height']

    # determine starting positions of A and B,
    # For now, we need this because RLD model is not trained
    try:
        tx_map_a = np.load(TX_PATH)
        row_a, col_a = np.unravel_index(np.argmax(tx_map_a), tx_map_a.shape)
        x_a = (col_a / (tx_map_a.shape[1] - 1)) * WIDTH
        y_a = (row_a / (tx_map_a.shape[0] - 1)) * HEIGHT
        agent_a_pos = (x_a, y_a)
        print(f"Loaded Agent A position: {agent_a_pos}")

        tx_map_b = np.load(TX_B_PATH)
        row_b, col_b = np.unravel_index(np.argmax(tx_map_b), tx_map_b.shape)
        x_b = (col_b / (tx_map_b.shape[1] - 1)) * WIDTH
        y_b = (row_b / (tx_map_b.shape[0] - 1)) * HEIGHT
        agent_b_pos = (x_b, y_b)
        print(f"Loaded Agent B position: {agent_b_pos}")
        
    except Exception as e:
        print(f"Could not load TX maps to find agent positions. Error: {e}")
        
        agent_a_pos = (0, 0)
        agent_b_pos = (WIDTH, HEIGHT)

    data_points = [
        {
            'id': 'A',
            'pos': agent_a_pos,
            '5.8': RM_PATH,
            '415': RM_415_PATH 
        },
        {
            'id': 'B',
            'pos': agent_b_pos,
            '5.8': RM_B_PATH,
            '415': RM_B_415_PATH 
        }
    ]

    # This is our radio map generator model, 
    # it is a sub for when the RLD model is fully trained
    radio_model = RadioMapModel(data_points)

    # start the environment
    config  = {'seed':1234}
    env = CustomEnv(
        config, 
        render_mode="rgb_array",
        hm_path=HM_PATH,
        radio_model=radio_model,  
        agent_a_pos=agent_a_pos, 
        agent_b_pos=agent_b_pos 
    )



    obs, info = env.reset() 
    frames = []  
    done = False

    print("Running simulation and collecting frames...")

    plot_live = False 
    
    for i in range(100):
        print(f"Step {i+1}/100")
        dummy_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(dummy_action)

        frame = env.render()
        frames.append(frame)
        if plot_live:
            plt.imshow(frame)
            display.display(plt.gcf())
            display.clear_output(wait=True)
        
        if terminated or truncated:
            print("Episode finished. Resetting.")
            obs, info = env.reset()

    if plot_live:
        plt.close()


    imageio.mimsave('simulation.gif', frames, fps=5) 
    imageio.mimwrite('simulation.mp4', frames, fps=5)

    print("Done. Saved simulation.gif and simulation.mp4")