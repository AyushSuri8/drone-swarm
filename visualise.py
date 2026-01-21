"""
Visualization script for trained swarm policy.

Usage:
    python visualize.py --checkpoint checkpoints/checkpoint_50.pt --n_agents 64

File: visualize.py
"""

import argparse
import numpy as np
import torch
import time

from envs.swarm_env import PyBulletSwarmEnv
from models.gnn_policy import GNNAgent


def visualize_policy(checkpoint_path: str, n_episodes: int = 3, record_video: bool = False):
    """Visualize trained policy in PyBullet GUI."""
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']
    
    # Create environment with GUI
    env_config = {
        'n_agents': config['n_agents'],
        'max_steps': config['max_steps'],
        'task_type': config['task_type'],
        'render_mode': 'gui',  # Enable visualization
        'collision_radius': config.get('collision_radius', 0.3),
        'world_size': config.get('world_size', 20.0)
    }
    
    env = PyBulletSwarmEnv(env_config)
    
    # Create agent and load weights
    agent = GNNAgent(
        node_dim=9,
        neighbor_dim=6,
        hidden_dim=config['hidden_dim'],
        device=config['device']
    )
    agent.policy.load_state_dict(checkpoint['policy_state_dict'])
    agent.policy.eval()
    
    print(f"\n{'='*60}")
    print(f"VISUALIZING TRAINED POLICY")
    print(f"{'='*60}")
    print(f"Agents: {env.n_agents}")
    print(f"Task: {config['task_type']}")
    print(f"Episodes: {n_episodes}")
    print(f"{'='*60}\n")
    
    # Run episodes
    for ep in range(n_episodes):
        print(f"\nEpisode {ep + 1}/{n_episodes}")
        print("-" * 40)
        
        obs, info = env.reset(seed=ep)
        episode_reward = 0
        episode_length = 0
        episode_collisions = 0
        
        done = False
        while not done and episode_length < env.max_steps:
            # Select action (deterministic)
            action, _ = agent.select_action(obs, deterministic=True)
            
            # Step environment
            obs, rewards, terminated, truncated, info = env.step(action)
            
            # Render
            env.render()
            
            # Update stats
            episode_reward += rewards.mean()
            episode_length += 1
            episode_collisions = info['episode_collisions']
            
            # Check done
            done = np.logical_or(terminated, truncated).all()
            
            # Print periodic updates
            if episode_length % 50 == 0:
                print(f"  Step {episode_length:3d}: "
                      f"Reward={episode_reward:.2f}, "
                      f"Goal dist={info['mean_goal_distance']:.2f}m, "
                      f"At goal={info['agents_at_goal']}/{env.n_agents}, "
                      f"Collisions={episode_collisions}")
        
        # Episode summary
        success_rate = info['agents_at_goal'] / env.n_agents
        collision_rate = episode_collisions / (episode_length * env.n_agents)
        
        print(f"\n  Episode Summary:")
        print(f"    Total reward: {episode_reward:.2f}")
        print(f"    Length: {episode_length} steps")
        print(f"    Success rate: {success_rate:.1%}")
        print(f"    Collision rate: {collision_rate:.4f}")
        print(f"    Total collisions: {episode_collisions}")
    
    env.close()
    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"{'='*60}")


def compare_policies(checkpoint_paths: list, n_episodes: int = 5):
    """Compare multiple checkpoints side-by-side."""
    
    results = []
    
    for ckpt_path in checkpoint_paths:
        print(f"\nEvaluating: {ckpt_path}")
        
        # Load checkpoint
        checkpoint = torch.load(ckpt_path)
        config = checkpoint['config']
        
        # Create environment (no GUI for comparison)
        env_config = {
            'n_agents': config['n_agents'],
            'max_steps': config['max_steps'],
            'task_type': config['task_type'],
            'render_mode': 'direct'
        }
        env = PyBulletSwarmEnv(env_config)
        
        # Create agent
        agent = GNNAgent(
            node_dim=9,
            neighbor_dim=6,
            hidden_dim=config['hidden_dim'],
            device=config['device']
        )
        agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        agent.policy.eval()
        
        # Evaluate
        episode_rewards = []
        success_rates = []
        collision_rates = []
        
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done and episode_length < env.max_steps:
                action, _ = agent.select_action(obs, deterministic=True)
                obs, rewards, terminated, truncated, info = env.step(action)
                
                episode_reward += rewards.mean()
                episode_length += 1
                done = np.logical_or(terminated, truncated).all()
            
            episode_rewards.append(episode_reward)
            success_rates.append(info['agents_at_goal'] / env.n_agents)
            collision_rates.append(info['episode_collisions'] / (episode_length * env.n_agents))
        
        env.close()
        
        # Store results
        results.append({
            'checkpoint': ckpt_path,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'success_rate': np.mean(success_rates),
            'collision_rate': np.mean(collision_rates)
        })
    
    # Print comparison
    print(f"\n{'='*80}")
    print("CHECKPOINT COMPARISON")
    print(f"{'='*80}")
    print(f"{'Checkpoint':<40} {'Reward':>12} {'Success':>10} {'Collisions':>12}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['checkpoint']:<40} "
              f"{r['mean_reward']:>12.2f} "
              f"{r['success_rate']:>10.1%} "
              f"{r['collision_rate']:>12.4f}")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize trained swarm policy')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--n_episodes', type=int, default=3,
                       help='Number of episodes to run')
    parser.add_argument('--compare', nargs='+', default=None,
                       help='Compare multiple checkpoints (no visualization)')
    parser.add_argument('--record', action='store_true',
                       help='Record video (not implemented yet)')
    
    args = parser.parse_args()
    
    if args.compare:
        # Comparison mode
        compare_policies(args.compare, n_episodes=args.n_episodes)
    else:
        # Visualization mode
        visualize_policy(args.checkpoint, n_episodes=args.n_episodes, record_video=args.record)


if __name__ == "__main__":
    main()