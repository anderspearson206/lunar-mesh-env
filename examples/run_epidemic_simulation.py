import numpy as np
import random
import imageio
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lunar_mesh_env import LunarRoverMeshEnv, RadioMapModelNN


def build_epidemic_actions(env):
    """Build actions that forward to every currently connected target."""
    actions = {}
    num_rovers = len(env.possible_agents)

    for agent_id in env.agents:
        agent = env.agent_map[agent_id]
        if agent.energy <= 0:
            continue

        action = [random.randint(0,8)] + [0] * (num_rovers + 1)

        for i, target_id in enumerate(env.possible_agents):
            if target_id == agent_id:
                continue

            target = env.agent_map.get(target_id)
            if target is not None and target in agent.neighbors:
                action[i + 1] = 1

        if agent.bs_connected:
            action[-1] = 1

        actions[agent_id] = np.array(action, dtype=np.int64)

    return actions

def main():
    DATA_ROOT = '../../NASA_DCGR_NETWORKING/radio_data_2/radio_data_2'
    HM_PATH = f'{DATA_ROOT}/hm/hm_18.npy'
    
    MODEL_PATHS = {
        'k2_model': '../RadioLunaDiff/pretrained_models_network/k2unet/best_k2_model.pth',
        'pmnet_model': '../RadioLunaDiff/pretrained_models_network/pmnet/best_pm_model.pt',
        'diffusion_model': '../RadioLunaDiff/pretrained_models_network/diffusion'
    }

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

    env = LunarRoverMeshEnv(
        hm_path=HM_PATH,
        radio_model=radio_model,
        num_agents=3,
        render_mode="rgb_array"
    )

    obs, info = env.reset()
    frames = []
    
    print("Starting simulation...")
    for step in range(10):
        actions = build_epidemic_actions(env)
        obs, rewards, terms, truncs, infos = env.step(actions)

        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        if step % 1 == 0:
            r0 = rewards.get('rover_0', 0.0)
            print(
                f"Step {step}: Agent 0 Reward: {r0:.2f}, "
                f"BS unique: {env.unique_packets_rcvd}, raw: {env.total_packets_rcvd}"
            )

        if all(terms.values()) or all(truncs.values()):
            break

    env.close()
    
    if frames:
        print(f"Saving GIF ({len(frames)} frames)...")
        imageio.mimsave('epidemic_simulation.gif', frames, fps=4)
        imageio.mimwrite('epidemic_simulation.mp4', frames, fps=5, output_params=['-vcodec', 'libx264'])
        print("Done!")

if __name__ == "__main__":
    main()