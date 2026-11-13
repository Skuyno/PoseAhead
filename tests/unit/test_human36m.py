import numpy as np
import pytest

from poseahead.core import Joint
from poseahead.core.serialization import decode_pose_frame, encode_pose_frame
from poseahead.datasets.human36m import camera_poses_to_sequence


def test_human36m_mapping_axes_units_root_and_exact_event_time():
    source = np.zeros((2, 32, 3))
    source[:] = [1000, 2000, 3000]
    source[:, 17] += [400, 0, 0]  # left shoulder
    source[:, 25] += [-400, 0, 0]  # right shoulder
    source[:, 12] += [0, -200, 0]
    source[:, 13] += [0, -600, 0]
    source[:, 14] += [0, -700, 0]
    source[:, 15] += [0, -900, 0]  # head-top must not replace head
    sequence = camera_poses_to_sequence(
        source, sequence_id="S1/Walking/1", camera_id="54138969", first_frame_index=50
    )
    first, second = sequence.frames
    np.testing.assert_allclose(first.root_position, [1, 2, 3])
    np.testing.assert_allclose(first.joint_positions[Joint.LEFT_SHOULDER], [.4, 0, 0])
    np.testing.assert_allclose(first.joint_positions[Joint.RIGHT_SHOULDER], [-.4, 0, 0])
    np.testing.assert_allclose(first.joint_positions[Joint.CHEST], [0, -.4, 0])
    np.testing.assert_allclose(first.joint_positions[Joint.HEAD], [0, -.7, 0])
    assert first.source_timestamp_ns == first.produced_timestamp_ns == 1_000_000_000
    assert second.source_timestamp_ns == 1_020_000_000
    assert sequence.duration_ns == 20_000_000
    assert first.root_rotation is None and first.joint_rotations is None
    decoded = decode_pose_frame(encode_pose_frame(first))
    assert decoded.metadata["root_translation_status"] == "ground_truth"
    assert decoded.is_valid
    source[:] = 0
    np.testing.assert_allclose(first.root_position, [1, 2, 3])


@pytest.mark.parametrize("source", [
    np.zeros((0, 32, 3)), np.zeros((1, 17, 3)), np.full((1, 32, 3), np.nan),
])
def test_human36m_rejects_wrong_layout_empty_or_nonfinite_ground_truth(source):
    with pytest.raises(ValueError):
        camera_poses_to_sequence(source, sequence_id="clip", camera_id="cam")
