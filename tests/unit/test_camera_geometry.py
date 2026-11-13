import pytest

from poseahead.perception.camera_geometry import CameraIntrinsics


def test_invalid_intrinsics_rejected() -> None:
    with pytest.raises(ValueError):
        CameraIntrinsics(0, 400, 320, 240, 640, 480)


@pytest.mark.parametrize("size", [float("nan"), float("inf"), 640.5, True, 0])
def test_invalid_image_dimensions_rejected(size):
    with pytest.raises(ValueError):
        CameraIntrinsics(500, 400, 320, 240, size, 480)
    with pytest.raises(ValueError):
        CameraIntrinsics(500, 400, 320, 240, 640, size)
