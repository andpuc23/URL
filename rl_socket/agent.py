"""Camera module"""
import numpy as np
import cv2
from camera import Camera


class Agent:
    """Main camera class"""

    def __init__(self,
                 t_coords: np.ndarray,
                 r_coords: np.ndarray,
                 camera: Camera,
                 target_points
                 ) -> None:
        """
        init method.
        :param t_coords: 3x1 global coords vector.
        :param r_coords: 3x1 global angles vector.
        :param camera: Camera instance to get camera params
        :param target_points: target points located on camera surface.
        """

        self.tvec = t_coords
        self.angles = r_coords
        self.rvec = self._compute_rvec()
        self.camera_mat = camera.camera_mat
        self.dist_coeffs = camera.dist_coeffs
        self.target_points = target_points


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


    def move(self, delta: np.ndarray) -> None:
        """
        adds delta to current camera coordinates
        :param delta: delts coords [x1, x2, x3]
        """
        self.tvec = self.tvec + delta[:3]
        self.angles = self.angles + delta[3:]
        self.rvec = self._compute_rvec()
