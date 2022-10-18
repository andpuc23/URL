import numpy as np
from numpy import cos, sin
import cv2

class Camera:
    """Main camera class"""

    def __init__(self,
                 camera_mat: np.ndarray,
                 dist_coeffs: np.ndarray) -> None:
        """
        init method.
        :param camera_mat: intrinsic matrix.
        :param dist_coeffs: distortion coefficients.
        """

        self.camera_mat = camera_mat
        self.dist_coeffs = dist_coeffs


    def project_points(self,
                       points_env: np.ndarray,
                       tvec: np.ndarray,
                       rvec: np.ndarray) -> np.ndarray:
        """
        method for projecting points in the 3d area into camera
        :param points_env: points in the enviroment ((x1, x2, x3), ...)
        :param tvec: ...
        :param rvec: ...
        :return: projected points

        """
        
        return cv2.projectPoints(points_env,
                                 rvec,
                                 tvec,
                                 self.camera_mat,
                                 self.dist_coeffs)[0]
