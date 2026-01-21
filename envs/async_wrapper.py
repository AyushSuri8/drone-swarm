"""
Asynchronous Environment Wrapper (The "Real World" Simulator)

Innovates beyond standard PyBullet by injecting variable latency.
- Simulates sensor delay (Age of Information).
- Augments observation with 'latency' feature for the TEM.
"""
import numpy as np
import gymnasium as gym
from collections import deque

class AsyncDelayWrapper(gym.Wrapper):
    def __init__(self, env, min_delay=1, max_delay=5):
        super().__init__(env)
        self.min_delay = min_delay
        self.max_delay = max_delay
        
        # Buffer to store past observations: (obs, timestamp)
        self.obs_buffer = deque(maxlen=max_delay + 1)
        self.current_step = 0
        
        # We need to expand the observation space to include 'latency'
        # The GNN expects specific dimensions, so we will append latency 
        # as a global feature or part of the node state.
        # For simplicity in this architecture, we append it to every agent's node state.
        # Old Node Dim: 9 [pos, vel, goal]
        # New Node Dim: 10 [pos, vel, goal, latency]
        self.orig_obs_dim = env.obs_dim  # e.g., 9 + neighbors
        # We assume the first 9 are node features. We add 1.
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_step = 0
        self.obs_buffer.clear()
        
        # Fill buffer with initial frame
        for _ in range(self.max_delay + 1):
            self.obs_buffer.append((obs, 0))
            
        return self._get_delayed_obs(0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        
        # Store current "Ground Truth" observation
        self.obs_buffer.append((obs, self.current_step))
        
        # Determine random delay for this step (simulate jitter)
        delay = np.random.randint(self.min_delay, self.max_delay + 1)
        target_time = max(0, self.current_step - delay)
        
        delayed_obs = self._get_delayed_obs(target_time)
        
        # Add latency info to the info dict for debugging
        info['latency'] = delay * self.env.dt
        
        return delayed_obs, reward, terminated, truncated, info

    def _get_delayed_obs(self, target_time):
        # Find the observation closest to target_time
        # In a real deque, we'd search. Since we append sequentially, 
        # we can index from the end.
        
        # Simple retrieval: Get the obs from 'delay' steps ago
        # Note: In a real async system, we might miss frames. 
        # Here we just grab the historic frame.
        best_match = self.obs_buffer[0] # Default to oldest
        
        for stored_obs, stored_time in reversed(self.obs_buffer):
            if stored_time <= target_time:
                best_match = (stored_obs, stored_time)
                break
        
        obs_data, obs_time = best_match
        
        # Calculate Age of Information (AoI)
        latency = self.current_step - obs_time
        
        # INNOVATION: Inject Latency into Observation
        # obs shape: (n_agents, 9 + neighbor_feats)
        # We append 'latency' to the NODE features (first 9)
        # New structure: [pos(3), vel(3), goal(3), LATENCY(1), neighbors(...)]
        
        n_agents = obs_data.shape[0]
        
        # Split node and neighbor features
        # Assuming node_dim is 9 (hardcoded in original env, we should verify)
        node_feats = obs_data[:, :9]
        neighbor_feats = obs_data[:, 9:]
        
        # Create latency vector (normalized)
        # 1.0 means max_delay, 0.0 means fresh
        latency_val = latency / self.max_delay
        latency_vec = np.full((n_agents, 1), latency_val, dtype=np.float32)
        
        # Concatenate
        new_obs = np.hstack([node_feats, latency_vec, neighbor_feats])
        
        return new_obs
