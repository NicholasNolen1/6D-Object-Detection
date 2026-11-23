"""
Camera models and intrinsics handling.
"""

from dataclasses import dataclass
from typing import Tuple, Union
import numpy as np


@dataclass
class CameraIntrinsics:
    """
    Pinhole camera intrinsic parameters.
    fx, fy: Focal lengths in pixels
    cx, cy: Principal point offsets in pixels
    width, height: Image dimensions in pixels
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def matrix(self) -> np.ndarray:
        """Returns 3x3 camera matrix K."""
        return np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0]
        ], dtype=np.float64)

    @classmethod
    def from_matrix(cls, K: np.ndarray, width: int = 640, height: int = 480) -> "CameraIntrinsics":
        """Instantiates CameraIntrinsics from a 3x3 matrix."""
        return cls(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            width=int(width),
            height=int(height)
        )

    @classmethod
    def from_file(cls, path: str, width: int = 640, height: int = 480) -> "CameraIntrinsics":
        """Loads camera matrix from a text file (whitespace or comma separated)."""
        K = np.loadtxt(path, dtype=np.float64)
        return cls.from_matrix(K, width=width, height=height)

    @classmethod
    def default_primesense(cls, width: int = 640, height: int = 480) -> "CameraIntrinsics":
        """Standard PrimeSense / RealSense default intrinsics for 640x480."""
        return cls(
            fx=525.0,
            fy=525.0,
            cx=319.5,
            cy=239.5,
            width=width,
            height=height
        )
