"""
High-Order Control Barrier Function (HOCBF) Layer.

This layer intercepts the GNN's actions and projects them into the safe set
defined by the barrier function h(x) >= 0.

Logic:
1. Define Barrier h(x): ||p_i - p_j||^2 - D_safe^2
2. Compute derivatives for relative degree 2 (acceleration controlled):
   h_dot = 2 * (p_i - p_j) * (v_i - v_j)
3. Enforce HOCBF constraint:
   ddot{h} + alpha1 * h_dot + alpha2 * h >= 0
   
   Where ddot{h} contains the control input 'u' (acceleration).
   This creates a linear constraint on u: A*u <= b

File: safety/cbf_layer.py
"""

import torch
import torch.nn as nn

class DifferentiableHOCBF(nn.Module):
    def __init__(self, n_agents, collision_radius, alpha1=2.0, alpha2=2.0, max_accel=2.0):
        super().__init__()
        self.n_agents = n_agents
        self.r_safe = collision_radius * 1.5  # Add margin for safety
        
        # HOCBF Gains
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        
        self.max_accel = max_accel

    def forward(self, nominal_actions, obs):
        """
        Args:
            nominal_actions: (batch, n_agents, 3) - Actions from GNN
            obs: (batch, n_agents, 9) - State [pos, vel, ...]
            
        Returns:
            safe_actions: (batch, n_agents, 3)
        """
        batch_size = nominal_actions.shape[0]
        
        # Extract state
        # pos: (batch, n_agents, 3)
        pos = obs[:, :, :3]
        vel = obs[:, :, 3:6]
        
        # This uses a Projected Gradient Descent (PGD) approximation
        # instead of a full QP solver.
        
        safe_actions = nominal_actions.clone()
        
        # Number of PGD steps to enforce safety
        n_steps = 10
        step_size = 0.1
        
        for _ in range(n_steps):
            # 1. Pairwise diffs
            # delta_p: (batch, n_agents, n_agents, 3)
            delta_p = pos.unsqueeze(2) - pos.unsqueeze(1)
            delta_v = vel.unsqueeze(2) - vel.unsqueeze(1)
            
            # 2. Distance squared (h)
            # dist_sq: (batch, n_agents, n_agents)
            dist_sq = (delta_p ** 2).sum(dim=-1)
            
            # Mask self-collisions (diagonal)
            mask = torch.eye(self.n_agents, device=pos.device).bool().unsqueeze(0)
            dist_sq.masked_fill_(mask, float('inf'))
            
            # Barrier h(x) = ||dp||^2 - R^2
            h = dist_sq - self.r_safe**2
            
            # 3. Time derivative h_dot(x) = 2 * dp * dv
            h_dot = 2 * (delta_p * delta_v).sum(dim=-1)
            
            # 4. Second derivative ddot{h} part (L_f^2 h)
            # ddot{h} = 2 * (dv * dv + dp * du)
            drift_term = 2 * (delta_v ** 2).sum(dim=-1)
            
            # The constraint is:
            # 2 * dp * (u_i - u_j) + drift + alpha1 * h_dot + alpha2 * h >= 0
            
            # Recompute control impact based on current safe_actions estimate
            u_current = safe_actions
            delta_u = u_current.unsqueeze(2) - u_current.unsqueeze(1)
            control_term = 2 * (delta_p * delta_u).sum(dim=-1)
            
            constraint_val = control_term + drift_term + self.alpha1 * h_dot + self.alpha2 * h
            
            # Identify violations
            violations = -constraint_val
            violations = torch.relu(violations) # Only positive violations matter
            
            # If max violation is 0, we are safe
            if violations.max() < 1e-4:
                break
                
            # Compute Gradient of violation w.r.t actions u_i
            # Grad_u_i (Violation) = - 2 * (p_i - p_j)
            
            # Sum gradients from all violated constraints
            # (batch, n_agents, 3)
            grad = (violations.unsqueeze(-1) * (-2 * delta_p)).sum(dim=2)
            
            # Normalize grad to avoid exploding updates
            grad_norm = grad.norm(dim=-1, keepdim=True) + 1e-6
            grad = grad / grad_norm
            
            # Update actions: Move u_i in direction that reduces violation
            safe_actions = safe_actions - step_size * grad * violations.max(dim=-1, keepdim=True)[0]
            
            # Clamp to physical limits
            safe_actions = torch.clamp(safe_actions, -self.max_accel, self.max_accel)
            
        return safe_actions