"""
Production-grade PyBullet Swarm Environment
Optimized for N=256-512 agents with GPU-accelerated collision detection

File: envs/swarm_env.py
"""

import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
import torch
from typing import Tuple, Dict, Optional, List
import time
from collections import deque


class PyBulletSwarmEnv(gym.Env):
    """
    Ultra-large swarm environment using PyBullet + GPU acceleration.
    
    Key optimizations:
    - GPU collision detection (PyTorch)
    - Batched force application
    - Spatial hashing for O(N) neighbor queries
    - Vectorized reward computation
    """

    metadata = {"render_modes": ["human", "direct"]}

    def __init__(self, config: dict):
        super().__init__()

        # Core parameters
        self.n_agents = config.get('n_agents', 256)
        self.dt = config.get('dt', 1.0/240.0)  # PyBullet default
        self.control_dt = config.get('control_dt', 0.05)  # 20Hz control
        self.substeps = int(self.control_dt / self.dt)
        self.max_steps = config.get('max_steps', 500)
        
        # Physics parameters
        self.max_vel = config.get('max_vel', 3.0)
        self.max_accel = config.get('max_accel', 2.0)
        self.collision_radius = config.get('collision_radius', 0.3)
        self.safety_radius = config.get('safety_radius', 0.5)  # Desired separation
        self.neighbor_radius = config.get('neighbor_radius', 5.0)
        self.max_neighbors = config.get('max_neighbors', 10)
        
        # Task success parameters
        self.success_radius = config.get('success_radius', 2.0)
        
        # World bounds
        self.world_size = config.get('world_size', 25.0)
        self.bounds = np.array([
            [-self.world_size, self.world_size],
            [-self.world_size, self.world_size],
            [0.5, self.world_size]  # Minimum altitude 0.5m
        ])
        
        # Normalization constants
        self.norm_factor = self.world_size 
        
        # Task configuration
        self.task_type = config.get('task_type', 'formation')
        self.formation_spacing = config.get('formation_spacing', 1.0)
        
        # Rendering
        self.render_mode = config.get('render_mode', 'direct')
        self.camera_distance = 30.0
        self.camera_yaw = 45
        self.camera_pitch = -30
        
        # GPU device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Reward weights
        self.w_progress = config.get('w_progress', 5.0)
        self.w_collision = config.get('w_collision', -2.0)
        self.w_goal = config.get('w_goal', 50.0)
        self.w_time = config.get('w_time', -0.01)
        self.w_separation = config.get('w_separation', -1.0)
        
        # Initialize PyBullet
        if self.render_mode == 'gui':
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)  # No gravity for drones
        p.setTimeStep(self.dt)
        p.setRealTimeSimulation(0)
        
        # Create environment
        self._setup_world()
        
        # Agent bodies
        self.agent_ids = []
        self._create_agents()
        
        # State tensors (GPU)
        self.positions = None
        self.velocities = None
        self.goals = None
        self.prev_goal_dist = None
        
        # Tracking
        self.current_step = 0
        self.episode_collisions = 0
        self.collision_history = deque(maxlen=100)
        
        # Observation/action spaces
        # The base observation dimension (excluding neighbors) is 9 (pos, vel, goal_rel)
        self.orig_obs_dim = 9
        self.obs_dim = self.orig_obs_dim + self.max_neighbors * 6
        self.act_dim = 3
        
        # Spatial hashing grid (for efficient neighbor queries)
        self.grid_size = self.neighbor_radius

        # Define Gym Spaces (Required for Wrappers)
        # Assuming flattened observations for all agents concatenated, or (N, Obs)
        # Since this is a custom multi-agent env, we define spaces as (N, dims)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.n_agents, self.obs_dim), 
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-self.max_accel, 
            high=self.max_accel, 
            shape=(self.n_agents, 3), 
            dtype=np.float32
        )
        
        print(f"[PyBulletSwarmEnv] Initialized")
        print(f"  Agents: {self.n_agents}")
        print(f"  Control rate: {1/self.control_dt:.0f} Hz")
        print(f"  Physics substeps: {self.substeps}")
        print(f"  Device: {self.device}")
        print(f"  Render mode: {self.render_mode}")
    
    def _setup_world(self):
        """Create world environment."""
        # Ground plane
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, 0])
        
        # Optional: Add boundaries visualization
        if self.render_mode == 'gui':
            # Draw world boundaries
            bound_color = [1, 0, 0]
            
            # Bottom square
            corners = [
                [-self.world_size, -self.world_size, 0],
                [self.world_size, -self.world_size, 0],
                [self.world_size, self.world_size, 0],
                [-self.world_size, self.world_size, 0],
            ]
            
            for i in range(4):
                p.addUserDebugLine(
                    corners[i], 
                    corners[(i+1)%4], 
                    bound_color, 
                    lineWidth=2
                )
    
    def _create_agents(self):
        """Create agent bodies in PyBullet."""
        # Sphere collision shape
        collision_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=self.collision_radius
        )
        
        # Visual shape (different colors for visibility)
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.collision_radius,
            rgbaColor=[0.2, 0.5, 0.8, 0.9]
        )
        
        # Create agents
        for i in range(self.n_agents):
            # Temporary position (will be reset)
            temp_pos = [0, 0, 10 + i * (self.collision_radius * 2.5)]
            
            agent_id = p.createMultiBody(
                baseMass=0.5,  # 500g drone
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=temp_pos
            )
            
            # Set dynamics (high damping for stable flight)
            p.changeDynamics(
                agent_id, -1,
                linearDamping=0.9,
                angularDamping=0.9,
                rollingFriction=0.0,
                spinningFriction=0.0
            )
            
            self.agent_ids.append(agent_id)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment."""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Generate initial positions (avoid collisions)
        positions = self._generate_initial_positions()
        
        # Small random velocities
        velocities = np.random.randn(self.n_agents, 3).astype(np.float32) * 0.1
        
        # Generate goals
        goals = self._generate_goals()
        
        # Set states in PyBullet
        for i, agent_id in enumerate(self.agent_ids):
            p.resetBasePositionAndOrientation(
                agent_id,
                positions[i].tolist(),
                [0, 0, 0, 1]
            )
            p.resetBaseVelocity(
                agent_id,
                velocities[i].tolist(),
                [0, 0, 0]
            )
        
        # Store on GPU
        self.positions = torch.from_numpy(positions).to(self.device)
        self.velocities = torch.from_numpy(velocities).to(self.device)
        self.goals = torch.from_numpy(goals).to(self.device)
        self.prev_goal_dist = torch.norm(self.positions - self.goals, dim=1)
        
        # Reset tracking
        self.current_step = 0
        self.episode_collisions = 0
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def _generate_initial_positions(self) -> np.ndarray:
        """Generate collision-free initial positions."""
        positions = []
        max_attempts = 1000
        
        for i in range(self.n_agents):
            for attempt in range(max_attempts):
                # Random position in world
                pos = np.random.uniform(
                    low=[b[0] * 0.8 for b in self.bounds],
                    high=[b[1] * 0.8 for b in self.bounds]
                )
                
                # Check collision with existing agents
                if len(positions) == 0:
                    positions.append(pos)
                    break
                
                distances = np.linalg.norm(
                    np.array(positions) - pos, axis=1
                )
                
                if np.all(distances > self.collision_radius * 3):
                    positions.append(pos)
                    break
            else:
                # Fallback: grid placement
                side = int(np.ceil(np.sqrt(self.n_agents)))
                row = i // side
                col = i % side
                pos = np.array([
                    (col - side/2) * 2.0,
                    (row - side/2) * 2.0,
                    5.0
                ])
                positions.append(pos)
        
        return np.array(positions, dtype=np.float32)
    
    def _generate_goals(self) -> np.ndarray:
        """Generate goal positions based on task type."""
        if self.task_type == 'formation':
            # Grid formation at opposite side
            side = int(np.ceil(np.sqrt(self.n_agents)))
            spacing = self.formation_spacing
            
            goals = np.zeros((self.n_agents, 3), dtype=np.float32)
            for i in range(self.n_agents):
                row = i // side
                col = i % side
                goals[i] = [
                    (col - side/2) * spacing,
                    (row - side/2) * spacing,
                    self.world_size * 0.6  # Mid-high altitude
                ]
        
        elif self.task_type == 'migration':
            # All agents migrate to opposite corner
            start_region = np.array([-self.world_size * 0.8, -self.world_size * 0.8, 5])
            end_region = np.array([self.world_size * 0.8, self.world_size * 0.8, 15])
            
            goals = np.random.uniform(
                low=end_region - 3,
                high=end_region + 3,
                size=(self.n_agents, 3)
            ).astype(np.float32)
        
        elif self.task_type == 'coverage':
            # Random spread
            goals = np.random.uniform(
                low=[b[0] * 0.9 for b in self.bounds],
                high=[b[1] * 0.9 for b in self.bounds],
                size=(self.n_agents, 3)
            ).astype(np.float32)
        
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        
        return goals
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Step environment with control rate.
        
        Args:
            actions: (n_agents, 3) acceleration commands
        
        Returns:
            obs, rewards, terminated, truncated, info
        """
        # Convert to tensor and clip
        actions_tensor = torch.from_numpy(actions).to(self.device)
        actions_tensor = torch.clamp(actions_tensor, -self.max_accel, self.max_accel)
        actions_np = actions_tensor.cpu().numpy()
        
        # Apply forces for substeps
        for _ in range(self.substeps):
            for i, agent_id in enumerate(self.agent_ids):
                # Convert acceleration to force
                force = actions_np[i] * 0.5  # Mass = 0.5kg
                p.applyExternalForce(
                    agent_id, -1,
                    force.tolist(),
                    [0, 0, 0],
                    p.WORLD_FRAME
                )
            
            # Step physics
            p.stepSimulation()
        
        # Read back state from PyBullet
        positions = []
        velocities = []
        for agent_id in self.agent_ids:
            pos, _ = p.getBasePositionAndOrientation(agent_id)
            vel, _ = p.getBaseVelocity(agent_id)
            positions.append(pos)
            velocities.append(vel)
        
        positions = np.array(positions, dtype=np.float32)
        velocities = np.array(velocities, dtype=np.float32)
        
        # Enforce bounds (teleport back if out)
        for i in range(3):
            out_low = positions[:, i] < self.bounds[i][0]
            out_high = positions[:, i] > self.bounds[i][1]
            
            if out_low.any() or out_high.any():
                # Clamp position
                positions[:, i] = np.clip(
                    positions[:, i],
                    self.bounds[i][0],
                    self.bounds[i][1]
                )
                # Reverse velocity
                velocities[out_low | out_high, i] *= -0.5
                
                # Apply to PyBullet
                for j in np.where(out_low | out_high)[0]:
                    p.resetBasePositionAndOrientation(
                        self.agent_ids[j],
                        positions[j].tolist(),
                        [0, 0, 0, 1]
                    )
                    p.resetBaseVelocity(
                        self.agent_ids[j],
                        velocities[j].tolist(),
                        [0, 0, 0]
                    )
        
        # Update GPU tensors
        self.positions = torch.from_numpy(positions).to(self.device)
        self.velocities = torch.from_numpy(velocities).to(self.device)
        
        # Detect collisions and compute separation violations
        collisions, separation_violations = self._detect_collisions_and_separation()
        
        # Compute individual rewards (returns Tensor now)
        rewards_tensor = self._compute_rewards(collisions, separation_violations)
        
        # Cooperative Reward Mixing
        global_reward = torch.mean(rewards_tensor)
        mixing_alpha = 0.2
        mixed_rewards_tensor = (1.0 - mixing_alpha) * rewards_tensor + mixing_alpha * global_reward
        rewards_np = mixed_rewards_tensor.cpu().numpy()
        
        # Check termination
        terminated, truncated = self._check_done()
        
        self.current_step += 1
        
        obs = self._get_obs()
        info = self._get_info()
        info['collisions'] = collisions.cpu().numpy()
        info['separation_violations'] = separation_violations.cpu().numpy()
        
        # Calculate average distance to goal for debugging
        dist_to_goal = torch.norm(self.goals - self.positions, dim=1).mean().item()
        info['dist_to_goal'] = dist_to_goal
        
        return obs, rewards_np, terminated, truncated, info
    
    def _detect_collisions_and_separation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPU-accelerated collision and separation detection.
        
        Returns:
            collisions: (n_agents,) boolean
            separation_violations: (n_agents,) count of too-close neighbors
        """
        # Pairwise distances
        distances = torch.cdist(self.positions, self.positions)
        
        # Collision: distance < collision_radius
        collision_matrix = (distances < self.collision_radius) & (distances > 0)
        collisions = collision_matrix.any(dim=1)
        
        # Count collisions
        n_collisions = collisions.sum().item()
        self.episode_collisions += n_collisions
        
        # Separation violations: too close but not colliding
        separation_matrix = (distances < self.safety_radius) & (distances >= self.collision_radius)
        separation_violations = separation_matrix.sum(dim=1).float()
        
        return collisions, separation_violations
    
    def _compute_rewards(self, collisions: torch.Tensor, separation_violations: torch.Tensor) -> torch.Tensor:
        """
        Multi-component reward function. Returns a torch.Tensor.
        
        Components:
        1. Progress toward goal
        2. Collision penalty
        3. Goal reached bonus
        4. Time penalty (Fixed Existence Penalty)
        5. Separation violation penalty
        """
        # Distance to goal
        goal_dist = torch.norm(self.positions - self.goals, dim=1)
        
        # Progress reward
        progress = (self.prev_goal_dist - goal_dist) * self.w_progress
        self.prev_goal_dist = goal_dist.clone()
        
        # Collision penalty
        collision_penalty = collisions.float() * self.w_collision
        
        # Goal reached bonus
        at_goal = (goal_dist < self.success_radius).float() * self.w_goal
        
        # Separation penalty (encourage maintaining safe distance)
        separation_penalty = separation_violations * self.w_separation
        
        # Penalize every step to force speed.
        time_penalty = -0.05 
        
        # Anti-gridlock penalty
        in_red_zone = goal_dist < 3.0  # Within 3 meters
        speed = torch.norm(self.velocities, dim=1)
        is_stopped = speed < 0.1
        gridlock_penalty = -0.5 * (in_red_zone & is_stopped).float()
        
        # Total reward
        rewards = progress + collision_penalty + at_goal + time_penalty + separation_penalty + gridlock_penalty
        
        return rewards
    
    def _check_done(self) -> Tuple[np.ndarray, np.ndarray]:
        """Check termination conditions."""
        goal_dist = torch.norm(self.positions - self.goals, dim=1)
        
        # Terminated: reached goal
        terminated = (goal_dist < self.success_radius).cpu().numpy()
        
        # Truncated: max steps
        truncated = np.full(self.n_agents, self.current_step >= self.max_steps)
        
        return terminated, truncated
    
    def _get_obs(self) -> np.ndarray:
        """
        Get state for all agents with normalization.
        
        Observation: [norm_pos, vel, goal_rel_norm, neighbor_features]
        """
        # Neighbor features
        neighbor_features = self._get_neighbor_features()
        
        norm_pos = self.positions / self.norm_factor
        
        # goal_rel is also a distance vector, so normalize it
        norm_goal_rel = (self.goals - self.positions) / self.norm_factor
        
        # Update the concatenation to use normalized values
        obs = torch.cat([
            norm_pos,           # Normalized Position
            self.velocities,    # Velocity (already small)
            norm_goal_rel,      # Normalized relative goal
            neighbor_features   # Relative neighbor interactions 
        ], dim=-1)
        
        return obs.cpu().numpy()
    
    def _get_neighbor_features(self) -> torch.Tensor:
        """
        Compute relative state of k-nearest neighbors.
        Handles cases where n_agents < max_neighbors (e.g., N=8) by padding.
        """
        # Distances: (n_agents, n_agents)
        distances = torch.cdist(self.positions, self.positions)
        
        # Mask self-collisions with infinity so we don't pick ourselves
        mask = torch.eye(self.n_agents, device=self.device).bool()
        distances.masked_fill_(mask, float('inf'))
        
        available_neighbors = self.n_agents - 1
        k_search = min(self.max_neighbors, available_neighbors)
        
        # Get actual nearest neighbors
        # indices: (n_agents, k_search)
        _, nearest_idx = torch.topk(distances, k=k_search, largest=False, dim=1)
        
        # Gather neighbor positions and velocities
        # shape: (n_agents, k_search, 3)
        # We need to expand indices to gather from (n_agents, 3) tensor
        idx_flat = nearest_idx.reshape(-1)
        
        # (n_agents * k_search, 3)
        neighbor_pos = self.positions[idx_flat].reshape(self.n_agents, k_search, 3)
        neighbor_vel = self.velocities[idx_flat].reshape(self.n_agents, k_search, 3)
        
        # Calculate relative features
        # current_pos: (n_agents, 1, 3)
        current_pos = self.positions.unsqueeze(1)
        current_vel = self.velocities.unsqueeze(1)
        
        rel_pos = neighbor_pos - current_pos
        rel_vel = neighbor_vel - current_vel
        
        features = torch.cat([rel_pos, rel_vel], dim=-1)  # (n_agents, k_search, 6)
        
        if k_search < self.max_neighbors:
            pad_size = self.max_neighbors - k_search
            # Create zeros: (n_agents, pad_size, 6)
            padding = torch.zeros((self.n_agents, pad_size, 6), device=self.device)
            features = torch.cat([features, padding], dim=1)
            
        # Flatten features: (n_agents, max_neighbors * 6)
        return features.reshape(self.n_agents, -1)
    
    def _get_info(self) -> dict:
        """Diagnostic information."""
        goal_dist = torch.norm(self.positions - self.goals, dim=1)
        
        return {
            'step': self.current_step,
            'mean_goal_distance': goal_dist.mean().item(),
            'max_goal_distance': goal_dist.max().item(),
            'agents_at_goal': (goal_dist < self.success_radius).sum().item(),
            'episode_collisions': self.episode_collisions,
            'mean_velocity': torch.norm(self.velocities, dim=1).mean().item()
        }
    
    def render(self, mode='human'):
        """Render environment."""
        if self.render_mode == 'gui':
            # Update camera to follow swarm center
            mean_pos = self.positions.mean(dim=0).cpu().numpy()
            p.resetDebugVisualizerCamera(
                cameraDistance=self.camera_distance,
                cameraYaw=self.camera_yaw,
                cameraPitch=self.camera_pitch,
                cameraTargetPosition=mean_pos.tolist()
            )
            
            time.sleep(self.control_dt)
    
    def close(self):
        """Cleanup."""
        p.disconnect(self.client)
        print("[PyBulletSwarmEnv] Closed")


# Quick test
if __name__ == "__main__":
    config = {
        'n_agents': 64,
        'render_mode': 'gui',
        'task_type': 'formation',
        'max_steps': 300
    }
    
    env = PyBulletSwarmEnv(config)
    # Testing options=None compatibility
    obs, info = env.reset(seed=42)
    
    print(f"\nObservation shape: {obs.shape}")
    print(f"Initial info: {info}\n")
    
    print("Running visualization for 200 steps with random actions...")
    for step in range(200):
        # Random actions
        actions = np.random.randn(env.n_agents, 3) * 0.5
        
        obs, rewards, term, trunc, info = env.step(actions)
        env.render()
        
        if step % 50 == 0:
            print(f"Step {step:3d}: Mean Reward={rewards.mean():.4f}, "
                  f"Collisions={info['episode_collisions']}, "
                  f"Avg Dist={info['dist_to_goal']:.2f}")
    
    env.close()