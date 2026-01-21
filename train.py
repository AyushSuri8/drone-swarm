"""
Complete training script for ultra-large swarm RL

File: train.py
"""

import numpy as np
import torch
from collections import defaultdict, deque
import time
import argparse
import os
import json
from typing import Dict, Optional

# Prevent test code from running on import
import sys
if __name__ != "__main__":
    sys.exit(0)

# Import your modules
from envs.swarm_env import PyBulletSwarmEnv
from envs.async_wrapper import AsyncDelayWrapper
from models.gnn_policy import GNNAgent


class RolloutBuffer:
    """Buffer for storing rollout data with GAE computation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def add(self, obs, action, reward, done, log_prob, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def compute_returns_and_advantages(self, gamma=0.99, gae_lambda=0.95):
        """Compute discounted returns and GAE advantages."""
        rewards = np.array(self.rewards)  # (T, n_agents)
        dones = np.array(self.dones)
        values = np.array(self.values)  # (T, n_agents)
        
        T, n_agents = rewards.shape
        
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        # Compute GAE per agent
        for i in range(n_agents):
            gae = 0
            for t in reversed(range(T)):
                if t == T - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1, i]
                
                delta = rewards[t, i] + gamma * next_value * (1 - dones[t, i]) - values[t, i]
                gae = delta + gamma * gae_lambda * (1 - dones[t, i]) * gae
                advantages[t, i] = float(gae)
                returns[t, i] = gae + values[t, i]
        
        return returns, advantages
    
    def get(self):
        """Get all data as dictionary."""
        returns, advantages = self.compute_returns_and_advantages()
        
        return {
            'obs': np.array(self.obs),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'dones': np.array(self.dones),
            'log_probs': np.array(self.log_probs),
            'values': np.array(self.values),
            'returns': returns,
            'advantages': advantages
        }


class SwarmTrainer:
    """Main trainer class."""
    
    def __init__(self, config: dict):
        self.config = config
        self.n_agents = config['n_agents']  # Store locally to avoid wrapper issues
        
        # Create environment
        print("\n[1/4] Creating environment...")
        env_config = {
            'n_agents': config['n_agents'],
            'max_steps': config['max_steps'],
            'task_type': config['task_type'],
            'render_mode': 'direct',  # No GUI during training
            'collision_radius': config.get('collision_radius', 0.3),
            'world_size': config.get('world_size', 20.0)
        }
        
        # Create Base Environment
        self.env = PyBulletSwarmEnv(env_config)

        # Check for async flag
        node_dim_setting = config.get('node_dim', 9)
        
        if node_dim_setting == 10:
            print("🌊 ACTIVATING ASYNC LATENCY WRAPPER (Sim-to-Real Mode)")
            # Wrap the environment to inject 1-5 steps of delay
            self.env = AsyncDelayWrapper(
                self.env, 
                min_delay=config.get('min_delay', 1),
                max_delay=config.get('max_delay', 5)
            )
            
            # Verify it worked
            # Note: access via .unwrapped if needed, but wrapper logic usually exposes this
            try:
                print(f"   -> Environment is now Asynchronous. Obs Dim: {self.env.orig_obs_dim} -> {self.env.orig_obs_dim + 1}")
            except AttributeError:
                print(f"   -> Environment is now Asynchronous (Attribute check skipped)")

        else:
            print("Using Standard Synchronous Env")
        
        # Create agent
        print("[2/4] Creating GNN agent...")
        
        node_dim = node_dim_setting
        if config.get('node_dim') is not None:
            print(f"  ✓ Using command line node_dim: {node_dim}")
        else:
            print(f"  ✓ Auto-detected node_dim: {node_dim} (9 for Sync Env)")


        self.agent = GNNAgent(
            node_dim=node_dim,
            neighbor_dim=6,
            hidden_dim=config['hidden_dim'],
            device=config['device'],
            lr=config['learning_rate'],
            use_cbf=config.get('use_cbf', True),
            n_agents=config['n_agents'],
            collision_radius=config.get('collision_radius', 0.3)
        )
        
        # Training parameters
        self.n_steps = config['n_steps']
        self.n_epochs = config['n_epochs']
        self.total_timesteps = config['total_timesteps']
        self.eval_freq = config['eval_freq']
        self.save_freq = config['save_freq']
        self.max_steps = config['max_steps'] # Store locally
        
        # Logging
        print("[3/4] Setting up logging...")
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.get('wandb_project', 'ultra-swarm'),
                    config=config,
                    name=config.get('run_name', None)
                )
                print("  ✓ WandB initialized")
            except Exception as e:
                print(f"  ✗ WandB failed: {e}")
                self.use_wandb = False
        
        # Metrics tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.collision_rates = deque(maxlen=100)
        self.success_rates = deque(maxlen=100)
        
        # Checkpointing
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"[4/4] Checkpoint directory: {self.checkpoint_dir}")
        
        is_async = isinstance(self.env, AsyncDelayWrapper)
        
        print("\n" + "=" * 60)
        print("SWARM TRAINER INITIALIZED")
        print("=" * 60)
        # Use local self.n_agents instead of self.env.n_agents to avoid wrapper crash
        print(f"Environment: {self.n_agents} agents")
        print(f"Policy: GNN with hidden_dim={config['hidden_dim']}")
        print(f"Using Async Env: {is_async}")
        print(f"Using Safety Layer (CBF): {config.get('use_cbf', True)}")
        print(f"Node Dimension: {node_dim}")
        print(f"Total timesteps: {self.total_timesteps:,}")
        print(f"Rollout length: {self.n_steps}")
        print(f"PPO Mini-Batch Size: {config['batch_size']}")
        print(f"Device: {config['device']}")
        print("=" * 60 + "\n")
    
    def train(self):
        """Main training loop."""
        print("🚀 STARTING TRAINING...")
        print()
        
        buffer = RolloutBuffer()
        
        obs, info = self.env.reset(seed=self.config.get('seed', None))
        episode_reward = np.zeros(self.n_agents) # Use local var
        episode_length = 0
        
        n_updates = 0
        start_time = time.time()
        
        for step in range(self.total_timesteps):
            # Collect rollout
            if len(buffer.obs) < self.n_steps:
                
                # 1. Get action, log_prob, and value in ONE pass (Agent step)
                action, log_prob, value = self.agent.step(obs, deterministic=False)
                
                # 2. Step physics
                next_obs, rewards, terminated, truncated, info = self.env.step(action)
                
                # Store transition
                done = np.logical_or(terminated, truncated).astype(float)
                buffer.add(obs, action, rewards, done, log_prob, value)
                
                # Update episode stats
                episode_reward += rewards
                episode_length += 1
                
                # Check if episode done
                # Use self.max_steps instead of self.env.max_steps
                if done.all() or episode_length >= self.max_steps:
                    self.episode_rewards.append(episode_reward.mean())
                    self.episode_lengths.append(episode_length)
                    
                    # Use self.n_agents instead of self.env.n_agents
                    if episode_length > 0:
                         col_rate = info.get('episode_collisions', 0) / (episode_length * self.n_agents)
                    else:
                         col_rate = 0
                    
                    self.collision_rates.append(col_rate)
                    self.success_rates.append(info.get('agents_at_goal', 0) / self.n_agents)
                    
                    obs, info = self.env.reset()
                    episode_reward = np.zeros(self.n_agents)
                    episode_length = 0
                else:
                    obs = next_obs
            
            # Update policy
            if len(buffer.obs) >= self.n_steps:
                rollout_data = buffer.get()
                
                update_stats = self.agent.update(
                    rollout_data, 
                    n_epochs=self.n_epochs,
                    batch_size=self.config['batch_size']
                )
                
                n_updates += 1
                buffer.reset()
                
                # Logging
                if n_updates % 10 == 0 or n_updates == 1:
                    elapsed_time = time.time() - start_time
                    fps = step / elapsed_time if elapsed_time > 0 else 0
                    
                    metrics = {
                        'update': n_updates,
                        'step': step,
                        'fps': fps,
                        'policy_loss': update_stats['policy_loss'],
                        'value_loss': update_stats['value_loss'],
                        'entropy': update_stats['entropy'],
                    }
                    
                    if len(self.episode_rewards) > 0:
                        metrics.update({
                            'episode_reward': np.mean(self.episode_rewards),
                            'episode_length': np.mean(self.episode_lengths),
                            'collision_rate': np.mean(self.collision_rates),
                            'success_rate': np.mean(self.success_rates)
                        })
                    
                    # Print progress
                    print(f"Update {n_updates:4d} | Step {step:8d}/{self.total_timesteps:8d} | "
                          f"Reward: {metrics.get('episode_reward', 0):7.2f} | "
                          f"Success: {metrics.get('success_rate', 0):5.1%} | "
                          f"Collisions: {metrics.get('collision_rate', 0):6.4f} | "
                          f"FPS: {fps:6.0f}")
                    
                    if self.use_wandb:
                        import wandb
                        wandb.log(metrics, step=step)
                
                # Evaluation
                if n_updates % self.eval_freq == 0:
                    eval_stats = self.evaluate(n_episodes=5)
                    print(f"  [EVAL] Reward: {eval_stats['mean_reward']:.2f} | "
                          f"Success: {eval_stats['success_rate']:.1%} | "
                          f"Collisions: {eval_stats['collision_rate']:.4f}")
                    
                    if self.use_wandb:
                        import wandb
                        wandb.log({f'eval/{k}': v for k, v in eval_stats.items()}, step=step)
                
                # Save checkpoint
                if n_updates % self.save_freq == 0:
                    self.save_checkpoint(f'checkpoint_{n_updates}.pt')
        
        # Final save
        self.save_checkpoint('final_model.pt')
        
        print(f"\n{'='*60}")
        print(f"✅ TRAINING COMPLETE")
        print(f"Total time: {time.time() - start_time:.1f}s")
        print(f"Final success rate: {np.mean(self.success_rates):.1%}")
        print(f"{'='*60}")
        
        if self.use_wandb:
            import wandb
            wandb.finish()
        
        self.env.close()
    
    def evaluate(self, n_episodes: int = 5) -> dict:
        """Evaluate policy."""
        self.agent.policy.eval()
        avg_rewards = []
        success_rates = []
        collision_rates = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_collisions = 0
            episode_length = 0
            
            while not done and episode_length < self.max_steps: # Use local var
                # Select deterministic action
                action, _, _ = self.agent.step(obs, deterministic=True)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward.sum() # Sum over agents
                
                # Count collisions (if env provides info)
                if isinstance(info, dict):
                    episode_collisions += info.get('collisions', 0)
                
                obs = next_obs
                episode_length += 1
                
                # Check for episode termination (corrected to use .all() for vectorized env)
                if np.logical_or(terminated, truncated).all():
                    done = True
            
            avg_rewards.append(episode_reward / self.n_agents) # Use local var
            
            # Check success (assuming 'success' key exists in info on episode end)
            success = info.get('success', False) if isinstance(info, dict) else False
            success_rates.append(1.0 if success else 0.0)
            
            # Calculate collision rate for this episode
            if episode_length > 0:
                # episode_collisions is the total number of collision events
                # Denominator is total possible collision steps (agents * steps)
                col_rate = episode_collisions / (episode_length * self.n_agents)
            else:
                col_rate = 0.0
                
            collision_rates.append(col_rate) 
            
        self.agent.policy.train()
        
        # Adjust return keys to match the train function's expectation
        return {
            'mean_reward': np.mean(avg_rewards),
            'std_reward': np.std(avg_rewards),
            'collision_rate': np.mean(collision_rates),
            'success_rate': np.mean(success_rates)
        }
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'policy_state_dict': self.agent.policy.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'config': self.config
        }, path)
        print(f"  💾 Checkpoint saved: {path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path)
        self.agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  📂 Checkpoint loaded: {path}")


def main():
    parser = argparse.ArgumentParser(description='Train swarm control policy')
    
    # Environment
    parser.add_argument('--n_agents', type=int, default=128, help='Number of agents')
    parser.add_argument('--task_type', type=str, default='formation', choices=['formation', 'migration', 'coverage'])
    parser.add_argument('--max_steps', type=int, default=500, help='Max episode length')
    
    # Training
    parser.add_argument('--total_timesteps', type=int, default=1_000_000, help='Total training steps')
    parser.add_argument('--n_steps', type=int, default=2048, help='Rollout length')
    parser.add_argument('--n_epochs', type=int, default=4, help='PPO epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size for PPO update')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    
    # Node dimension
    parser.add_argument('--node_dim', type=int, default=None, help='Node dimension (default: 9 for sync, 10 for async)')
    
    # Async Wrapper
    # NOTE: Setting --node_dim=10 activates the async wrapper and temporal policy.
    parser.add_argument('--use_async_env', action='store_true', help='Use the async delay wrapper (setting --node_dim=10 is preferred)')
    parser.add_argument('--min_delay', type=int, default=1, help='Min observation delay steps for async wrapper')
    parser.add_argument('--max_delay', type=int, default=5, help='Max observation delay steps for async wrapper')
    
    # Safety Layer
    parser.add_argument('--no_cbf', action='store_true', help='Disable the CBF safety layer')
    parser.add_argument('--collision_radius', type=float, default=0.3, help='Collision radius for environment and CBF')

    # Logging
    parser.add_argument('--eval_freq', type=int, default=10, help='Evaluation frequency (updates)')
    parser.add_argument('--save_freq', type=int, default=20, help='Save frequency (updates)')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='ultra-swarm', help='WandB project name')
    parser.add_argument('--run_name', type=str, default=None, help='Run name')
    
    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Convert to config dict
    config = vars(args)
    config['use_cbf'] = not config['no_cbf'] # If --no_cbf is present, use_cbf becomes False
    
    # Ensure node_dim defaults correctly if not specified
    if config['node_dim'] is None:
        if config['use_async_env']:
             config['node_dim'] = 10
        else:
             config['node_dim'] = 9

    # Set seeds
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    # Create trainer
    trainer = SwarmTrainer(config)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()