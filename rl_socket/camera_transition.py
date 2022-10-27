"""Camera transition module"""
from typing import Optional

import torch
import torch.nn as nn
from scipy.stats import norm


class CameraTransition(nn.Module):
    """Class for camera's coordinats transition"""

    dist_supported = False

    def __init__(self, device: str,
                 camera_mat: torch.Tensor,
                 target_points: torch.Tensor,
                 points_env: torch.Tensor,
                 dist_coeffs: Optional[torch.Tensor] = None) -> None:
        """
        Initial method
        :param device: 'cuda' or 'cpu'
        :param camera_mat: 3x3 camera's instrinct parameters
        :param target points: (num_points, 2) tensor with target points to compare
        :param points_env: (num_points, 3) coords of enviroment points in the world's coordinates
        :param dist_coeffs: distortion coefficients for the camera
        """
        super().__init__()
        self._device        = device
        self.camera_mat     = camera_mat
        self.target_points  = target_points
        self.points_env     = points_env

        if not CameraTransition.dist_supported and dist_coeffs is not None:
            print('Sorry, distortion coeffitients are not supported yet')
            self.dist_coeffs = None
        else:
            self.dist_coeffs = dist_coeffs

        self._to_device()


    def _to_device(self):
        """Transfer tensors to the device"""
        self.camera_mat     = self.camera_mat.to(self._device)
        self.target_points  = self.target_points.to(self._device)
        self.points_env     = self.points_env.to(self._device)


    def _project_points(self,
                        state: torch.Tensor) -> torch.Tensor:
        """
        Project points from world coordinates to the camera's surface
        :param state: (batch_size, 6) tensor of camera's world's coordinates
        :return: (batch_size, num_points, 2) tensor of projected points' coordinates
        """

        batch_size = state.shape[0]
        tvec = state[:, :3]
        angles = state[:, 3:]

        rmat = self._compute_rmat(angles)
        r_t_mat = torch.concat((rmat, torch.unsqueeze(tvec, -1)), 2)
        coords = torch.hstack(
            (self.points_env, torch.ones(self.points_env.shape[0], 1).to(self._device))
        ).T
        camera_mat_batched = self.camera_mat.repeat(batch_size, 1, 1).to(self._device)
        coords = coords.repeat(batch_size, 1, 1).to(self._device)
        projected = torch.bmm(torch.bmm(camera_mat_batched, r_t_mat), coords)
        projected = projected / projected[:, -1, :].unsqueeze(1).repeat(1, 3, 1)
        projected = projected[:, :-1, :].mT
        return projected


    def _compute_rmat(self, angles: torch.Tensor):
        """
        Compute rotation matrix for given camera's angles
        :param angles: (batch_size, 3) angles tensor
        :return: (batch_size, 3, 3) rotation matrices' tensor
        """

        batch_size = angles.shape[0]
        first   = torch.zeros(batch_size, 3, 3).to(self._device)
        second  = torch.zeros(batch_size, 3, 3).to(self._device)
        third   = torch.zeros(batch_size, 3, 3).to(self._device)
        
        cos_a = torch.cos(angles[:, 0])
        sin_a = torch.sin(angles[:, 0])
        first[:, 0, 0] = cos_a
        first[:, 0, 1] = -sin_a
        first[:, 1, 0] = sin_a
        first[:, 1, 1] = cos_a
        first[:, 2, 2] = 1

        cos_b = torch.cos(angles[:, 1])
        sin_b = torch.sin(angles[:, 1])
        second[:, 1, 1] = cos_b
        second[:, 1, 2] = -sin_b
        second[:, 2, 1] = sin_b
        second[:, 2, 2] = cos_b
        second[:, 0, 0] = 1

        cos_c = torch.cos(angles[:, 2])
        sin_c = torch.sin(angles[:, 2])
        third[:, 0, 0] = cos_c
        third[:, 0, 2] = sin_c
        third[:, 2, 0] = -sin_c
        third[:, 2, 2] = cos_c
        third[:, 1, 1] = 1

        return torch.bmm(torch.bmm(first, second), third)


    def _calculate_reward(self, proj_points: torch.Tensor) -> float:
        """
        Calculate reward (sum of the euclidian distances) for current state
        :param proj_points: (batch_size, num_points, 2) tensor of projected points
        """
        batch_size = proj_points.shape[0]
        return torch.sqrt(
            torch.pow(self.target_points.repeat(batch_size, 1, 1) - proj_points, 2).sum(-1)
        ).sum()


    def forward(self,
                state: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        """
        Main forward method
        :param state: (batch_size, 6) previous camera's position
        :param action: (batch_size, 6) action for camera's transition
        :return: tuple of
            * (batch_size, 6) new camera's states
            * (batch_size, 1) rewards' tensor
        """
        state = state + action
        reward = self._calculate_reward(self._project_points(state))

        return state, -reward
