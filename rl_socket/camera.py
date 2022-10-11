import numpy as np
from numpy import cos, sin
import cv2

class Camera:
    """Main camera class"""

    def __init__(self,
                 t_coords: np.ndarray,
                 r_coords: np.ndarray,
                 camera_mat: np.ndarray,
                 dist_coeffs: np.ndarray,
                 target_points: np.ndarray) -> None:
        """
        init method.
        :param t_coords: 3x1 global coords vector.
        :param rvec: 3x1 global angles vector.
        :param camera_mat: intrinsic matrix.
        :param dist_coeffs: distortion coefficients.
        :param target_points: target points located on camera surface.
        """

        self.tvec = t_coords
        self.angles = r_coords
        self.rvec = self._compute_rvec()
        self.camera_mat = camera_mat
        self.dist_coeffs = dist_coeffs
        self.target_points = target_points

    def _compute_rvec(self):
        first = np.array([
            [cos(self.angles[0]), -sin(self.angles[0]), 0],
            [sin(self.angles[0]), cos(self.angles[0]), 0],
            [0, 0, 1]
        ])
        second = np.array([
            [1, 0, 0],
            [0, cos(self.angles[1]), -sin(self.angles[1])],
            [0, sin(self.angles[1]), cos(self.angles[1])]
        ])
        third = np.array([
            [cos(self.angles[2]), 0, sin(self.angles[2])],
            [0, 1, 0],
            [-sin(self.angles[2]), 0, cos(self.angles[2])]
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

    def project_points(self, points_env: np.ndarray) -> np.ndarray:
        """
        method for projecting points in the 3d area into camera
        :param points_env: points in the enviroment ((x1, x2, x3), ...)
        :return: projected points

        """
        return cv2.projectPoints(points_env,
                                 self.rvec,
                                 self.tvec,
                                 self.camera_mat,
                                 self.dist_coeffs)[0]