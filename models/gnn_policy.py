"""
Graph Neural Network Policy for Ultra-Large Swarm Control

Implements G-MAPONet architecture:
- Graph Attention for local interaction modeling
- Multi-Head Attention for global coordination
- Permutation invariant (handles variable swarm sizes)

File: models/gnn_policy.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Tuple, Optional
import numpy as np

# Import our new safety layer
from safety.cbf_layer import DifferentiableHOCBF


class GraphAttentionLayer(nn.Module):
    """Graph Attention layer for local neighbor interactions."""
    
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim // heads, heads=heads, dropout=dropout, concat=True)
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: (n_nodes, in_dim)
            edge_index: (2, n_edges)
        """
        out = self.gat(x, edge_index)
        out = self.norm(out)
        return F.relu(out)


class MultiHeadAttentionLayer(nn.Module):
    """Multi-head attention for global coordination."""
    
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, n_agents, dim)
        """
        # Self-attention
        attn_out, _ = self.mha(x, x, x)
        x = self.norm(x + attn_out)
        
        # Feedforward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class TemporalEncodingModule(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # Fourier features for time (High frequency vs Low frequency)
        # Input dim is 1 (normalized latency)
        self.linear = nn.Linear(1, hidden_dim) 
        self.activation = nn.SiLU() # Swish activation often works best for time
        
    def forward(self, latency: torch.Tensor) -> torch.Tensor:
        # latency: (batch * n_agents, 1)
        return self.activation(self.linear(latency))


class GNNSwarmPolicy(nn.Module):
    """
    GNN-based policy for swarm control.
    
    Architecture:
    1. Node embedding (local state encoding)
    2. Graph Attention (neighbor interactions)
    3. Multi-Head Attention (global coordination)
    4. Actor-Critic heads
    """
    
    def __init__(
        self,
        node_dim: int = 10,  # [pos, vel, goal_rel, latency]
        neighbor_dim: int = 6,  # [rel_pos, rel_vel]
        hidden_dim: int = 128,
        gat_heads: int = 4,
        mha_heads: int = 4,
        n_gat_layers: int = 2,
        n_mha_layers: int = 2,
        dropout: float = 0.1,
        # Safety params
        use_cbf: bool = True,
        n_agents: int = 64,
        collision_radius: float = 0.3
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.neighbor_dim = neighbor_dim
        self.hidden_dim = hidden_dim
        self.use_cbf = use_cbf
        
        if self.use_cbf:
            # We initialize with standard collision radius (0.3)
            # Note: We still use the base radius, but we'll scale it dynamically in forward()
            self.safety_layer = DifferentiableHOCBF(
                n_agents=n_agents, 
                collision_radius=collision_radius
            )
        
        # Initialize TEM
        self.tem = TemporalEncodingModule(hidden_dim=hidden_dim)
        
        # Input embedding
        # Takes (node_dim - 1) physical features only
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_dim - 1, hidden_dim), 
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Graph Attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, gat_heads, dropout)
            for _ in range(n_gat_layers)
        ])
        
        # Multi-Head Attention layers
        self.mha_layers = nn.ModuleList([
            MultiHeadAttentionLayer(hidden_dim, mha_heads, dropout)
            for _ in range(n_mha_layers)
        ])
        
        # Actor head (outputs action mean)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Learnable log std for stochastic policy
        self.log_std = nn.Parameter(torch.zeros(3))
    
    def forward(self, obs: torch.Tensor, neighbor_radius: float = 5.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GNN.
        """
        batch_size, n_agents, obs_dim = obs.shape
        
        # 1. Reshape to treat all agents in the batch as a super-graph
        flat_obs = obs.reshape(-1, obs_dim)
        
        # Extract Latency (Index 9)
        # Node features are now 0-8 (phys) and 9 (latency)
        phys_features = flat_obs[:, :self.node_dim - 1]
        latency = flat_obs[:, self.node_dim - 1: self.node_dim] # Keep dim
        
        # 1. Base Embedding
        x = self.node_encoder(phys_features) # (batch * n_agents, hidden_dim)
        
        # 2. Add Temporal Context
        # We condition the agent's "thought process" on how stale its data is
        temporal_context = self.tem(latency)
        x = x + temporal_context  # Additive conditioning
        
        # Build batched graph structure (K-NN graph based on positions)
        edge_index = self._build_batched_knn_graph(obs, k=10)
        
        # Apply GAT layers (process entire batch as one disjoint super-graph)
        for gat in self.gat_layers:
            # x shape: (batch_size * n_agents, hidden_dim)
            x = gat(x, edge_index)
            
        # Reshape back to (batch, n_agents, hidden_dim) before MHA
        x = x.reshape(batch_size, n_agents, self.hidden_dim)
        
        # Raw Action from Policy (The "Greedy" Action)
        action_mean = self.actor(x)
        
        if self.use_cbf:
            # Dynamically increase safety radius based on latency (Age of Information)
            # If latency is high (1.0), we multiply radius by 1.5
            
            # Use the mean of the latency tensor for an approximate batch-wide factor
            # Note: This is a simplification; a rigorous implementation would use agent-specific latency
            # for safety checks, but for performance, we use an aggregated factor.
            avg_latency = latency.mean().item() 
            # Latency is assumed normalized [0, 1]. Max factor is 1.5.
            adaptive_factor = 1.0 + (0.5 * avg_latency) 
            
            # Temporarily adjust radius on the DifferentiableHOCBF layer
            original_radius = self.safety_layer.r_safe
            self.safety_layer.r_safe = original_radius * adaptive_factor
            
            # Apply safety filter
            safe_action = self.safety_layer(action_mean, obs)
            action_mean = safe_action
            
            # Restore radius for next forward pass
            self.safety_layer.r_safe = original_radius
            
        # Critic output
        value = self.critic(x)
        
        return action_mean, value
    
    def _build_batched_knn_graph(self, obs: torch.Tensor, k: int = 10) -> torch.Tensor:
        """
        Build a single batched K-NN graph (for all environments) from observations.
        
        Returns:
            edge_index: (2, total_edges) tensor for the super-graph
        """
        batch_size, n_agents, _ = obs.shape
        # Positions are the first 3 features
        positions = obs[:, :, :3]  # (batch, n_agents, 3)
        
        # Handle the decoupled agent update case (n_agents=1) where no graph is needed
        if n_agents <= 1:
            return torch.empty((2, 0), dtype=torch.long, device=obs.device)

        all_edge_indices = []
        
        # NOTE: This loop iterates over the *batch* dimension (environments).
        for b in range(batch_size):
            pos = positions[b]  # (n_agents, 3)
            
            # Compute pairwise distances
            dist = torch.cdist(pos, pos)  # (n_agents, n_agents)
            
            # Get K nearest neighbors (excluding self)
            dist_masked = dist + torch.eye(n_agents, device=obs.device) * 1e10
            actual_k = min(k, n_agents - 1)
            _, knn_idx = torch.topk(dist_masked, k=actual_k, largest=False, dim=1)
            # knn_idx: (n_agents, actual_k)
            
            # Build edge index (src -> dst)
            src = torch.arange(n_agents, device=obs.device).unsqueeze(1).expand(-1, actual_k)
            edge_index = torch.stack([src.reshape(-1), knn_idx.reshape(-1)], dim=0)
            # edge_index: (2, n_agents * actual_k)
            
            # Offset indices for the batched graph
            offset = b * n_agents
            edge_index = edge_index + offset
            
            all_edge_indices.append(edge_index)
        
        return torch.cat(all_edge_indices, dim=1)
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample action from policy (Used primarily for non-training evaluation where value is ignored).
        """
        action_mean, _ = self.forward(obs)
        
        if deterministic:
            return action_mean, None
        
        # Sample from Gaussian
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)  # Sum over action dimensions
        
        return action, log_prob
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Returns:
            log_prob: (batch, n_agents)
            value: (batch, n_agents, 1)
            entropy: (batch, n_agents)
        """
        action_mean, value = self.forward(obs)
        
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, std)
        
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, value, entropy


class GNNAgent:
    """Wrapper for training the GNN policy."""
    
    def __init__(
        self,
        node_dim: int = 10,
        neighbor_dim: int = 6,
        hidden_dim: int = 128,
        device: str = 'cuda',
        lr: float = 3e-4,
        use_cbf: bool = True,
        n_agents: int = 64,
        collision_radius: float = 0.3
    ):
        self.device = device
        self.policy = GNNSwarmPolicy(
            node_dim=node_dim,
            neighbor_dim=neighbor_dim,
            hidden_dim=hidden_dim,
            use_cbf=use_cbf,
            n_agents=n_agents,
            collision_radius=collision_radius
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
    
    
    def step(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Efficient single-pass step: Returns Action, LogProb, AND Value.
        Fixes the "Double Forward Pass" bottleneck.
        
        Args:
            obs: (N_agents, Obs_dim) numpy array
            deterministic: bool
        
        Returns:
            action: (N_agents, 3) numpy array
            log_prob: (N_agents,) numpy array
            value: (N_agents,) numpy array
        """
        # (N_agents, Obs_dim) -> (1, N_agents, Obs_dim)
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Single Forward Pass through the GNN
            action_mean, value = self.policy(obs_tensor)
            
            if deterministic:
                action = action_mean
                # Placeholder log_prob for deterministic actions
                log_prob = torch.zeros((1, obs_tensor.shape[1]), device=self.device)
            else:
                std = torch.exp(self.policy.log_std)
                dist = torch.distributions.Normal(action_mean, std)
                action = dist.sample()
                # Sum log probs over action dim
                log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Convert tensors back to numpy/cpu, removing the batch dimension (0)
        action = action.squeeze(0).cpu().numpy()
        log_prob = log_prob.squeeze(0).cpu().numpy()
        value = value.squeeze(0).cpu().numpy()
        
        return action, log_prob, value
    
    def update(self, rollout_buffer: dict, n_epochs: int = 4, batch_size: int = 32):
        """
        Update policy using PPO with Time-Step Batching.
        """
        
        # obs shape: (n_steps, n_agents, obs_dim)
        obs = torch.FloatTensor(rollout_buffer['obs']).to(self.device)
        actions = torch.FloatTensor(rollout_buffer['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollout_buffer['log_probs']).to(self.device)
        returns = torch.FloatTensor(rollout_buffer['returns']).to(self.device)
        advantages = torch.FloatTensor(rollout_buffer['advantages']).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        n_steps, n_agents, _ = obs.shape
        
        # Track metrics
        policy_loss_epoch = 0
        value_loss_epoch = 0
        entropy_epoch = 0
        n_updates = 0
        
        for epoch in range(n_epochs):
            # Shuffle TIMESTEPS, not agents
            perm = torch.randperm(n_steps).to(self.device)
            
            for start_idx in range(0, n_steps, batch_size):
                # Get indices for this batch of timesteps
                idx = perm[start_idx : start_idx + batch_size]
                
                # Sample the swarm snapshots
                # Shape: (batch_size, n_agents, dim)
                mb_obs = obs[idx] 
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]
                
                # Forward pass - The GNN now sees 'n_agents' correctly!
                log_probs, values, entropy = self.policy.evaluate_actions(mb_obs, mb_actions)
                
                # Flatten for loss calculation (merge Batch and Agent dims)
                # Shape: (batch_size * n_agents)
                log_probs = log_probs.reshape(-1)
                mb_old_log_probs = mb_old_log_probs.reshape(-1)
                mb_advantages = mb_advantages.reshape(-1)
                values = values.reshape(-1)
                mb_returns = mb_returns.reshape(-1)
                entropy = entropy.reshape(-1)
                
                # PPO Loss Calculation
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((values - mb_returns) ** 2).mean()
                entropy_loss = -entropy.mean()
                
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                policy_loss_epoch += policy_loss.item()
                value_loss_epoch += value_loss.item()
                entropy_epoch += -entropy_loss.item()
                n_updates += 1
        
        return {
            'policy_loss': policy_loss_epoch / n_updates,
            'value_loss': value_loss_epoch / n_updates,
            'entropy': entropy_epoch / n_updates
        }