from turtle import distance
import numpy as np

import torch
import torch.nn as nn
from scipy.stats import norm

import cv2

class CameraTransition(nn.Module):

    def __init__(self, device: str,
                 camera_mat: np.ndarray,
                 dist_coeffs: np.ndarray,
                 target_points: torch.Tensor,
                 points_env: np.array,
                 reward_scale: float) -> None:
        super().__init__()
        self.device = device
        self.camera_mat = camera_mat
        self.dist_coeffs = dist_coeffs
        self.target_points = target_points
        self.points_env = points_env
        self.reward_scale = reward_scale


    def _project_points(self,
                        points_env: np.ndarray,
                        tvec: np.ndarray,
                        rvec: np.ndarray) -> np.ndarray:
        projected = []
        for point in points_env:
            print('point in _project_points',point, 'shape', point.reshape(-1,3).shape)
            projected.append(cv2.projectPoints(point.reshape(1,3).astype(np.float64),
                                 rvec,
                                 tvec,
                                 self.camera_mat,
                                 self.dist_coeffs)[0])
            
        return np.array(projected)


    def _compute_rvec(self, angles: np.ndarray):
        rmats = []
        for angles_ in angles:
            first = np.array([
                [np.cos(angles_[0]), -np.sin(angles_[0]), 0],
                [np.sin(angles_[0]), np.cos(angles_[0]), 0],
                [0, 0, 1]
            ], dtype=np.float32)
            second = np.array([
                [1, 0, 0],
                [0, np.cos(angles_[1]), -np.sin(angles_[1])],
                [0, np.sin(angles_[1]), np.cos(angles_[1])]
            ], dtype=np.float32)
            third = np.array([
                [np.cos(angles_[2]), 0, np.sin(angles_[2])],
                [0, 1, 0],
                [-np.sin(angles_[2]), 0, np.cos(angles_[2])]
            ], dtype=np.float32)
            rotation_matrix = first @ second @ third
        rmats.append(cv2.Rodrigues(rotation_matrix)[0])
        return np.array(rmats)


    def forward(self,
                state:torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        positions = state[:, :3]
        angles = state[:, 3:]
        positions = positions + action[:, :3]
        angles = angles + action[:, 3:]

        positions_arr = positions.detach().numpy()
        angles_arr = angles.detach().numpy()
        points_env_proj = self._project_points(self.points_env,
                                               positions_arr,
                                               self._compute_rvec(angles_arr))
        points_env_proj = torch.tensor(points_env_proj).to(self.device)

        distance_coeff = torch.sqrt(
            torch.pow(self.target_points - points_env_proj, 2).sum(1)
        ).sum()
        reward = norm.pdf(distance_coeff, scale=self.reward_scale)

        return state, -reward


