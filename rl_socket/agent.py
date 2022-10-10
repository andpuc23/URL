"""Camera module"""
import numpy as np
import cv2


class Agent:
    """Main camera class"""

    def __init__(self,
                 rvec: np.ndarray,
                 tvec: np.ndarray,
                 camera_mat: np.ndarray,
                 dist_coeffs: np.ndarray,
                 target_points: np.ndarray) -> None:
        """
        init method.
        :param rvec: init rotation vector.
        :param tvec: init translation vector.
        :param camera_mat: intrinsic matrix.
        :param dist_coeffs: distortion coefficients.
        :param target_points: target points located on camera surface.
        """

        self.rvec = rvec
        self.tvec = tvec
        self.camera_mat = camera_mat
        self.dist_coeffs = dist_coeffs
        self.target_points = target_points

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
                                 self.dist_coeffs)

    # TODO Разобраться с дельтой для движения камеры
    # нужно менять tvec и rvec, а не глобальные координаты
    def move(self, delta: np.ndarray) -> None:
        """
        adds delta to current camera coordinates
        :param delta: delts coords [x1, x2, x3]
        """
        self.camera_coords += delta