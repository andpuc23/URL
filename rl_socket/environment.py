"""Environments' file"""
import numpy as np
from agent import Agent
from camera import Camera


class Environment:
    """Environment class"""

    def __init__(self, points_coords: np.ndarray, agent: Agent, camera: Camera) -> None:
        self.points_coords = points_coords
        self.agent = agent
        self.camera = camera

    def get_observation(self) -> np.ndarray:
        """
        returns points' coordinates on camera image
        :return: array of points': [(i1, j1), (i2, j2), ... (i6, j6)]
        """
        return self.camera.project_points(self.points_coords)

    def running_cost(self) -> float:
        """
        returns sum of distances from true points to target values
        
        :return: float distance
        """
        return np.sqrt(
            np.power(
                self.get_observation() - self.agent.get_target_points(), 2
                ).sum(1)).sum()
