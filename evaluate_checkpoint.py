"""
Evaluate a saved checkpoint to retrieve performance metrics.
FIXED: formatting error by ensuring metrics are scalars.
"""

import torch
import numpy as np
import argparse
import os
import sys

# Import your modules
from envs.swarm_env import PyBulletSwarmEnv
from envs.async_wrapper import AsyncDelayWrapper
from models.gnn_policy import GNNAgent

def evaluate_checkpoint(checkpoint_path, n_episodes=10, render=False):
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"📂 Loading checkpoint: {checkpoint_path}...")
    
    # 1. Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    config = checkpoint['config']
    
    # Force device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['device'] = device
    
    print("   ✓ Config loaded.")
    print(f"   ✓ Training Device: {checkpoint['config'].get('device', 'unknown')}")
    print(f"   ✓ Current Device: {device}")

    # 2. Recreate the Environment
    print("\n[1/3] Reconstructing Environment...")
    env_config = {
        'n_agents': config['n_agents'],
        'max_steps': config['max_steps'],
        'task_type': config['task_type'],
        'render_mode': 'gui' if render else 'direct',
        'collision_radius': config.get('collision_radius', 0.3),
        'world_size': config.get('world_size', 20.0)
    }
    
    env = PyBulletSwarmEnv(env_config)

    # Apply Async Wrapper
    node_dim = config.get('node_dim', 9)
    if node_dim == 10:
        print("   🌊 Re-activating AsyncDelayWrapper (Sim-to-Real Mode)")
        env = AsyncDelayWrapper(
            env, 
            min_delay=config.get('min_delay', 1),
            max_delay=config.get('max_delay', 5)
        )
    
    # 3. Recreate the Agent
    print("[2/3] Reconstructing Agent...")
    agent = GNNAgent(
        node_dim=node_dim,
        neighbor_dim=6,
        hidden_dim=config['hidden_dim'],
        device=device,
        lr=config['learning_rate'],
        use_cbf=config.get('use_cbf', True),
        n_agents=config['n_agents'],
        collision_radius=config.get('collision_radius', 0.3)
    )
    
    # Load weights
    agent.policy.load_state_dict(checkpoint['policy_state_dict'])
    agent.policy.eval()
    print("   ✓ Weights loaded successfully.")

    # 4. Run Evaluation Loop
    print(f"\n[3/3] Running {n_episodes} evaluation episodes...")
    
    success_rates = []
    collision_rates = []
    rewards = []
    
    for i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        # Get max_steps safely
        max_steps = env.max_steps if hasattr(env, 'max_steps') else env.unwrapped.max_steps
        
        while not done and steps < max_steps:
            # Deterministic action
            action, _, _ = agent.step(obs, deterministic=True)
            
            next_obs, reward, term, trunc, info = env.step(action)
            
            # Sum reward (ensure it's a scalar sum)
            episode_reward += np.sum(reward)
            
            obs = next_obs
            steps += 1
            
            if np.logical_or(term, trunc).all():
                done = True
                
        # --- FIXED CALCULATION LOGIC ---
        n_agents = config['n_agents']
        
        # 1. Success (Agents at goal)
        agents_at_goal = info.get('agents_at_goal', 0)
        success_rate = float(agents_at_goal) / n_agents
        
        # 2. Collisions (Use cumulative count from Env info)
        total_collisions = info.get('episode_collisions', 0)
        # Ensure it is a scalar float
        if isinstance(total_collisions, (np.ndarray, list)):
            total_collisions = np.sum(total_collisions)
        
        col_rate = float(total_collisions) / (steps * n_agents) if steps > 0 else 0.0
        
        # 3. Mean Reward
        mean_reward = float(episode_reward) / n_agents
        
        success_rates.append(success_rate)
        collision_rates.append(col_rate)
        rewards.append(mean_reward)
        
        print(f"   > Episode {i+1}: Success={success_rate:.1%} | Collisions={col_rate:.4f} | Reward={mean_reward:.2f}")

    # 5. Final Report
    print("\n" + "="*50)
    print("FINAL RESULTS RETRIEVED")
    print("="*50)
    print(f"Model: {checkpoint_path}")
    print(f"Mean Success Rate:  {np.mean(success_rates):.1%}")
    print(f"Mean Collision Rate: {np.mean(collision_rates):.5f}")
    print(f"Mean Reward:         {np.mean(rewards):.2f}")
    print("="*50)
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='Path to .pt file')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to run')
    parser.add_argument('--render', action='store_true', help='Visualize in GUI')
    args = parser.parse_args()
    
    evaluate_checkpoint(args.checkpoint, args.episodes, args.render)