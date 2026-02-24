"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, chunk_size * action_dim))
        
        self.model = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        action_pred = self.model(state)
        action_pred = action_pred.reshape(-1, self.chunk_size, self.action_dim)

        loss = nn.MSELoss()(action_pred, action_chunk)

        return loss
        

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        action_pred = self.model(state)
        action_pred = action_pred.reshape(-1, self.chunk_size, self.action_dim)
        
        return action_pred


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        input_dim = state_dim + chunk_size * action_dim + 1
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, chunk_size * action_dim))
        
        self.model = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        noise = torch.randn_like(action_chunk)
        timestep = torch.rand(batch_size, 1, device=state.device)

        timestep_expanded = timestep.unsqueeze(-1)
        interpolated_action_chunk = timestep_expanded * action_chunk + (1 - timestep_expanded) * noise

        action_flat = interpolated_action_chunk.reshape(batch_size, -1)
        network_input = torch.cat([state, action_flat, timestep], dim=1)

        velocity_pred = self.model(network_input)
        velocity_pred = velocity_pred.reshape(batch_size, self.chunk_size, self.action_dim)
        velocity_target = action_chunk - noise

        loss = nn.MSELoss()(velocity_pred, velocity_target)

        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        action_chunk = torch.randn(batch_size, self.chunk_size, self.action_dim, device=state.device)
        d_timestep = 1.0 / num_steps

        for step in range(num_steps):
            timestep = step * d_timestep
            network_input = torch.cat(
                [state, action_chunk.reshape(batch_size, -1), 
                torch.full((batch_size, 1), timestep, device=state.device)], 
                dim=-1)
            velocity_pred = self.model(network_input)
            velocity_pred = velocity_pred.reshape(batch_size, self.chunk_size, self.action_dim)
            action_chunk = action_chunk + velocity_pred * d_timestep

        return action_chunk


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
