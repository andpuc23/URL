"""Camera module"""
import numpy as np


class Agent:
    """Main camera class"""
    # Пока подразумевается, что в camera_params хранятся все параметры камеры
    # Можно разбить на несколько отдельных полей
    def __init__(self, camera_params: tuple, target_points: np.ndarray) -> None:
        """
        init method
        :param camera_params: tuple of camara's parameters (inculding init position)
        :param target_points: target points located on camera surface
        """

        # TODO что там со внутренними параметрами камеры
        # как их передавать и хранить
        if camera_params[-1] is not None:
            self.camera_coords = camera_params[-1]
        else:
            self.camera_coords = np.zeros(3, dtype=np.int32)

        self.target_points = target_points

    def get_target_points(self) -> np.ndarray:
        """
        Getter for target_points
        :return: target points
        """
        return self.target_points

    # TODO реализовать проекцию точек в пронстранстве на плоскость камеры
    def project_points(self, points_env: np.ndarray) -> np.ndarray:
        """
        method for projecting points in the 3d area into camera
        :param points_env: points in the enviroment ((x1, x2, x3), ...)
        :return: projected points
        
        """
        raise NotImplementedError

    def move(self, delta: np.ndarray) -> None:
        """
        adds delta to current camera coordinates
        :param delta: delts coords [x1, x3, x3]
        """
        self.camera_coords += delta