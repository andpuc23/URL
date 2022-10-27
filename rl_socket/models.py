"""TD based modules"""
from typing import Iterator

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .camera_transition import CameraTransition
from .nn_models import ActorModel, CriticModel


class CriticTD(nn.Module):
    """
    Critic Temporal Difference model
    """

    def __init__(self, actor: ActorModel,
                       critic: CriticModel, 
                       transition: CameraTransition,
                       satellite_discount: float = .98) -> None:
        """
        Init method
        :param actor: Actor nn model
        :param critic: Critic nn model
        :param transition: Camera Transition model
        :param satellite_discount: satellite discount for TD method
        """
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.transition = transition
        self.satellite_discount = satellite_discount
        self.loss = nn.MSELoss()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Main forward method
        :param state: (batch_size, 6) camera's state
        :return: TD loss
        """
        with torch.no_grad():
            action = self.actor(state)
            next_state, reward = self.transition(state, action)
            td_target = reward + self.satellite_discount * self.critic(next_state)
        value = self.critic(state)
        return self.loss(value, td_target)

    def parameters(self) -> Iterator[Parameter]:
        return self.critic.parameters()


class ActorImprovedValue(nn.Module):

    def __init__(self, actor: ActorModel,
                       critic: CriticModel, 
                       transition: CameraTransition,
                       satellite_discount: float = .98) -> None:
        """
        Init method
        :param actor: Actor nn model
        :param critic: Critic nn model
        :param transition: Camera Transition model
        :param satellite_discount: satellite discount for TD method
        """
        super().__init__()
        self.critic = critic
        self.actor = actor
        self.transition = transition
        self.satellite_discount = satellite_discount

    def forward(self, state):
        """
        Main forward method
        :param state: (batch_size, 6) camera's state
        :return: actor's improved value
        """
        action = self.actor(state)
        next_state, reward = self.transition(state, action)
        improved_value = reward + self.satellite_discount * self.critic(next_state)
        return -improved_value.mean()

    def parameters(self) -> Iterator[Parameter]:
        return self.actor.parameters()
