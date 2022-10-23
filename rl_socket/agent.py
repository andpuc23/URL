"""Camera module"""
from re import M
from turtle import forward
import numpy as np
import cv2

import torch
from torch import nn
from torch import norm

from .camera import Camera


class Scaler(nn.Module):

    def __init__(self, constraint_linear: float = 1.0, 
                       constraint_angle: float = .5) -> None:
        super().__init__()
        self.constraint_linear = constraint_linear
        self.constraint_angle = constraint_angle

    def forward(self, x):
        linear = x[:3]
        angle = x[3:]
        linear_norm = norm(linear, 2)
        angle_norm = norm(angle, 2)
        if linear_norm > self.constraint_linear:
            linear = (self.constraint_linear - 1e-5) / linear_norm * linear
        if angle_norm > self.constraint_angle:
            angle = (self.constraint_angle - 1e-5) / angle_norm * angle

        return torch.cat((linear, angle))
        

class ActorModel(nn.Module):

    def __init__(self, in_channels: int,
                       out_channels: int,
                       hidden_size: int,
                       activation: nn.Module = nn.LeakyReLU(),
                       use_batch_norm: bool = False,
                       dropout: float = .0) -> None:
        super().__init__()
        self.linear_part = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, out_channels)
        )
        self.out = nn.Softmax()
        self.scale = Scaler()

    def forward(self, x):
        x = self.linear_part(x)
        return self.scale(self.out(x)) 


IMPLEMENTED = True

class Agent:
    """Main camera class"""

    def __init__(self,
                 t_coords: np.ndarray,
                 r_coords: np.ndarray,
                 camera: Camera,
                 target_points,
                 model
                 ) -> None:
        """
        init method.
        :param t_coords: 3x1 global coords vector.
        :param r_coords: 3x1 global angles vector.
        :param camera: Camera instance to get camera params
        :param target_points: target points located on camera surface.
        :param model: predictor model
        """

        self.tvec = t_coords
        self.angles = r_coords
        self.rvec = self._compute_rvec()
        self.camera_mat = camera.camera_mat
        self.dist_coeffs = camera.dist_coeffs
        self.target_points = target_points
        self.model = model

        # we want to bound our initial position
        self.boundaries = {'x': [-100, 100], 'y': [-100, 100], 'z': [0, 100]}

    def _compute_rvec(self):
        first = np.array([
            [np.cos(self.angles[0]), -np.sin(self.angles[0]), 0],
            [np.sin(self.angles[0]), np.cos(self.angles[0]), 0],
            [0, 0, 1]
        ])
        second = np.array([
            [1, 0, 0],
            [0, np.cos(self.angles[1]), -np.sin(self.angles[1])],
            [0, np.sin(self.angles[1]), np.cos(self.angles[1])]
        ])
        third = np.array([
            [np.cos(self.angles[2]), 0, np.sin(self.angles[2])],
            [0, 1, 0],
            [-np.sin(self.angles[2]), 0, np.cos(self.angles[2])]
        ])
        rotation_matrix = first @ second @ third
        return cv2.Rodrigues(rotation_matrix)[0]

    # remove?
    def get_target_points(self) -> np.ndarray:
        """
        Getter for target_points.
        :return: target points.
        """
        return self.target_points


    def _move(self, delta: np.ndarray) -> None:
        """
        adds delta to current camera coordinates
        :param delta: deltas coords [x1, x2, x3, ang1, ang2, ang3]
        """
        assert len(delta) == 6, 'wrong delta size in agent.move()'
        self.tvec = self.tvec + delta[:3]
        self.angles = self.angles + delta[3:]
        self.rvec = self._compute_rvec()

    def set_initial_position(self) -> None:
        """
        set initial position for agent to random within boundaries
        """
        self.tvec = [np.random.randint(*self.boundaries['x']),
                     np.random.randint(*self.boundaries['y']),
                     np.random.randint(*self.boundaries['z'])]

    def step(self, observation: np.ndarray) -> np.ndarray:
        if IMPLEMENTED:
            action = self.model(observation)
        else:
            action = np.random.uniform(-1, 1, 6) # placeholder
        self._move(action)
        return action

    def finetune(self, observations: list, actions: list, rewards: list):
        #TODO
        """
        trains the agent's model on one episode (a data batch)
        """
        
