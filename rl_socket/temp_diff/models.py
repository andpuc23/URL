from typing import Iterator

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from camera_transition import CameraTransition


class Scale(nn.Module):

    def __init__(self, constraint_linear: float,
                       constraint_angle: float) -> None:
        super().__init__()
        self.constraint_linear = constraint_linear
        self.constraint_angle = constraint_angle

    def forward(self, x):
        linear = x[:3]
        angle = x[3:]
        linear_norm = torch.norm(linear, 2)
        angle_norm = torch.norm(angle, 2)
        if linear_norm > self.constraint_linear:
            linear = (self.constraint_linear - 1e-5) / linear_norm * linear
        if angle_norm > self.constraint_angle:
            angle = (self.constraint_angle - 1e-5) / angle_norm * angle

        return torch.cat((linear, angle))


class Head(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        norm = torch.matmul(input.reshape(input.shape[0], 1, input.shape[1]),
                            input.reshape(input.shape[0], input.shape[1], 1)).squeeze(1) ** 0.5
        return input * (torch.nn.functional.tanh(norm) / norm)


class ActorModel(nn.Module):

    def __init__(self, actor_layer_size: int = 160, 
                       constraints_linear: float = 1.,
                       constraints_angle: float =.5) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            torch.nn.Linear(6, actor_layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(actor_layer_size, actor_layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(actor_layer_size, actor_layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(actor_layer_size, actor_layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(actor_layer_size, actor_layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(actor_layer_size, 6),
            # Head(),
            Scale(constraints_linear, constraints_angle)
        )

    def forward(self, x):
        return self.actor(x)


class CriticModel(nn.Module):

    def __init__(self, critic_layer_size: int = 70,
                       scale_factor: int = -50000) -> None:
        super().__init__()
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(6, critic_layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(critic_layer_size, critic_layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(critic_layer_size, critic_layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(critic_layer_size, critic_layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(critic_layer_size, critic_layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(critic_layer_size, 1)
        )
        self.act = torch.nn.Tanh()
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.critic(x)
        # print(x)
        x = self.act(x)
        print(x)
        return x * self.scale_factor


class CriticTD(nn.Module):

    def __init__(self, actor: ActorModel,
                       critic: CriticModel, 
                       transition: CameraTransition,
                       satellite_discount: float = .98) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.transition = transition
        self.satellite_discount = satellite_discount
        self.loss = nn.MSELoss()

    def forward(self, state):
        with torch.no_grad():
            action = self.actor(state)
            next_state, reward = self.transition(state, action)
            # print(self.critic(next_state))
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
        super().__init__()
        self.critic = critic
        self.actor = actor
        self.transition = transition
        self.satellite_discount = satellite_discount

    def forward(self, state):
        action = self.actor(state)
        next_state, reward = self.transition(state, action)
        improved_value = reward + self.satellite_discount * self.critic(next_state)
        return -improved_value.mean()

    def parameters(self) -> Iterator[Parameter]:
        return self.actor.parameters()
