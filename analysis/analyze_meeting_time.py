# run_built_in_heuristic.py

import numpy as np
import imageio
import torch
import sys
import os
import random
import time
import matplotlib.pyplot as plt
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

def get_meeting_time(step, env, times, last_steps):
    
    for agent_id in env.agents:
        agent = env.agent_map[agent_id]
        # meeting_times[agent_id] = {}
        # last_step_meet[agent_id] = {}
        for n in agent.neighbors:
            if n.id == agent_id:
                print(f" Agent id:{agent_id} has neighbor {n.id}")
            else:
                k = agent_id + "-" + n.id
                last_step_met = last_steps[k]
                times[k].append(step-last_step_met)
                last_steps[k] = step
        
        if agent.bs_connected:
            k = agent_id + "-" + env.base_station.id
            last_step_met = last_steps[k]
            times[k].append(step-last_step_met)
            last_steps[k] = step 
            
    # print(f"UPDATED TIMES AND STEPS FOR STEP: {step}")
    # print(times)
    # print(step)
            
import matplotlib.pyplot as plt
import numpy as np

def plot_meeting_time_distribution(meeting_times):
    """
    Plots the distribution of inter-encounter times from the simulation.
    """
    all_times = []
    
    # Flatten the dictionary and filter out continuous contacts (1) and init artifacts (0)
    for pair, times in meeting_times.items():
        # A time > 1 means the nodes were disconnected for that many steps before meeting
        valid_inter_encounter_times = [t for t in times if t > 0]
        all_times.extend(valid_inter_encounter_times)

    if not all_times:
        print("Not enough disconnected->reconnected events to plot a distribution yet.")
        print("Try increasing SIM_STEPS to give the rovers time to roam and reconnect.")
        return

    all_times = np.array(all_times)
    
    # Calculate the expected value E[U] and lambda parameter
    expected_u = np.mean(all_times)
    lambda_param = 1.0 / expected_u if expected_u > 0 else 0
    
    print(f"Total valid meetings: {len(all_times)}")
    print(f"E[U] (Average meeting time): {expected_u:.2f} steps")
    print(f"Calculated lambda (λ): {lambda_param:.4f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot empirical data histogram
    counts, bins, _ = plt.hist(
        all_times, 
        bins=max(10, min(50, len(set(all_times)))), # Dynamic bin sizing
        density=True, 
        alpha=0.6, 
        color='royalblue', 
        edgecolor='black',
        label='Simulated Inter-encounter Times'
    )
    
    # Plot theoretical exponential tail
    x_vals = np.linspace(min(all_times), max(all_times), 200)
    y_vals = lambda_param * np.exp(-lambda_param * x_vals)
    plt.plot(x_vals, y_vals, 'r--', linewidth=2.5, label=rf'Exponential Fit ($\lambda={lambda_param:.3f}$)')
    
    # Formatting
    plt.title('Distribution of Inter-Encounter Times', fontsize=14, fontweight='bold')
    plt.xlabel('Time (Simulation Steps)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    
    # Save and show
    plt.tight_layout()
    plt.savefig('meeting_time_distribution.png', dpi=300)
    plt.show()
       
            
    

def main():
    
    SEED = 1886
    set_seed(SEED)
    
    DATA_ROOT = '../../NASA_DCGR_NETWORKING/radio_data_2/radio_data_2'
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
            dummy_mode=True,
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
        render_mode=None,  
        radio_bias=RADIO_BIAS, 
        seed=SEED,
        packet_mode='rate',
        num_goals=200
    )
    
    obs, info = env.reset()
    frames = []
    
    meeting_times = {}
    last_step_meet = {}
    # setup the last meeting step tracker as
    # well as meeting time tracker
    for agent_id in env.agents:
        agent = env.agent_map[agent_id]
        # meeting_times[agent_id] = {}
        # last_step_meet[agent_id] = {}
        for n_id in env.agents:
            if n_id==agent_id:
                pass
            else:
                # for each neighbor (not the agent itself)
                # add an array to track all meeting times
                k = agent_id + "-"+ n_id
                if k not in meeting_times.keys():
                    meeting_times[k] = []
                if k not in last_step_meet.keys():
                    last_step_meet[k] = -1
        bs_k = agent_id + "-" + env.base_station.id
        meeting_times[bs_k] = []
        last_step_meet[bs_k] = -1
               
    print("Meeting time graph: ", meeting_times)
    print("last step meet: ", last_step_meet)
    print("\n--- Starting Simulation ---")
    print(f"Agents: {env.possible_agents}")
    print("Goal: Rovers will navigate to randomly assigned tasks (task marked as X).")
    
    SIM_STEPS = 2000
    start = time.time()
    for step in range(SIM_STEPS):
        
        actions = {}
       
        for agent_id in env.agents:
            actions[agent_id] = get_combined_heuristic_action(agent_id, env)
        
        obs, rewards, terms, truncs, infos = env.step(actions)
        get_meeting_time(step, env, meeting_times, last_step_meet)
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
        
        # THIS CAN BE UNCOMMENTED OUT TO GO STEP BY STEP,
        # BEST USED WITH THE HUMAN RENDER
        # input()    
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
        
    plot_meeting_time_distribution(meeting_times)
    
if __name__ == "__main__":
    main()