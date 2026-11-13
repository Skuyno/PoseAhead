"""Validated pinhole camera intrinsics used by perception backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if type(self.width) is not int or type(self.height) is not int:
            raise ValueError("camera image dimensions must be integers")
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all():
            raise ValueError("camera intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("focal lengths and image dimensions must be positive")
