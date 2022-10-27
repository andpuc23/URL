"""Actor and Critic nn based modules"""
import torch
import torch.nn as nn

from .utils import Scale

class ActorModel(nn.Module):
    """Actor nn model"""

    def __init__(self, actor_layer_size: int = 100, 
                       constraints_linear: float = 1.,
                       constraints_angle: float = .5) -> None:
        """
        Init method
        :param actor_layer_size: hidden layer's size
        :param constraints_linear: linear constraints
        :param constraints: angle constraints
        """
        super().__init__()
        self.actor = nn.Sequential(
            torch.nn.Linear(6, actor_layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(actor_layer_size, actor_layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(actor_layer_size, 6),
            Scale(constraints_linear, constraints_angle)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Main forward method
        :param x: (batch_size, 6) state's tensor
        """
        x = self.actor(x)
        return x


class CriticModel(nn.Module):
    """Critic nn model"""

    def __init__(self, critic_layer_size: int = 70,
                       scale_factor: int = -5000) -> None:
        """
        Init method
        :param critic_layer_size: hidden critic's size
        :param scale_factor: scaling for model's output
        """
        super().__init__()
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(6, critic_layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(critic_layer_size, critic_layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(critic_layer_size, 1)
        )
        self.act = torch.nn.Tanh()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Main forward method
        :param x: (batch_size, 6) state's tensor
        """
        x = self.critic(x)
        x = -(x ** 2)
        x = self.act(x)
        return x * self.scale_factor
