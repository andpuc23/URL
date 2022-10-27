"""Additional utils modules"""
import torch
import torch.nn as nn
import torch.linalg as LA


class Scale(nn.Module):
    """
    Constraints scaler for actions
    """
    def __init__(self, constraint_linear: float,
                       constraint_angle: float) -> None:
        """
        Init method
        :param constraint_linear: linear constraint
        :param constraint_angle: angle constraint
        """
        super().__init__()
        self.constraint_linear = constraint_linear
        self.constraint_angle = constraint_angle

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Main forward method
        :param x: (batch_size, 6) action's tensor
        """
        linear = x[:3]
        angle = x[3:]
        linear_norm = torch.norm(linear, 2)
        angle_norm = torch.norm(angle, 2)
        if linear_norm > self.constraint_linear:
            linear = (self.constraint_linear - 1e-5) / linear_norm * linear
        if angle_norm > self.constraint_angle:
            angle = (self.constraint_angle - 1e-5) / angle_norm * angle

        return torch.cat((linear, angle))


def torch_rodrigues(mat: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula's realisation"""
    U, _, V_T = LA.svd(mat)
    R = U @ V_T
    r1 = R[:, 2, 1] - R[:, 1, 2]
    r2 = R[:, 0, 2] - R[:, 2, 0]
    r3 = R[:, 1, 0] - R[:, 0, 1]

    r = torch.stack((r1, r2, r3), 1)
    s = LA.norm(r, dim=1) / 2
    c = (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2
    c = torch.clip(c, -1., 1.)
    theta = torch.acos(c)
    vth = 1 / (2 * s)
    r = r.T * vth * theta
    return r.T
