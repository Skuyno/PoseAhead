import hashlib
import json

import numpy as np
import pytest

from poseahead.datasets.cmu import (
    _JOINTS,
    forward_kinematics,
    load_cmu,
    parse_amc,
    parse_asf,
    rotation_xyz,
)
from poseahead.datasets.splits import sequence_split

ASF = """:units
length 0.45
angle deg
:root
order TX TY TZ RX RY RZ
axis XYZ
position 0 0 0
orientation 0 0 0
:bonedata
begin
id 1
name bone
direction 1 0 0
length 1
axis 0 0 0 XYZ
dof rx ry rz
end
:hierarchy
begin
root bone
end
"""
AMC = """:FULLY-SPECIFIED
:DEGREES
1
root 1 2 3 0 0 0
bone 0 0 90
2
root 1 2 3 0 0 90
bone 0 0 0
"""


def test_fk_scale_root_and_local_rotation():
    skeleton = parse_asf(ASF)
    motion = parse_amc(AMC, skeleton[1])
    for _, channels in motion:
        points = forward_kinematics(skeleton, channels)
        np.testing.assert_allclose(points["root"], np.array([1, 2, 3]) * 0.0254 / 0.45)
        np.testing.assert_allclose(points["bone"] - points["root"],
                                   [0, 0.0254 / 0.45, 0], atol=1e-12)


def test_nonzero_axis_basis():
    skeleton = parse_asf(ASF.replace("axis 0 0 0 XYZ", "axis 0 0 90 XYZ"))
    points = forward_kinematics(skeleton, {"root": np.zeros(6), "bone": [90, 0, 0]})
    np.testing.assert_allclose(points["bone"], [0, 0, -0.0254 / 0.45], atol=1e-12)


def test_rotation_is_proper():
    matrix = rotation_xyz([31, -27, 83])
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(1)


@pytest.mark.parametrize("text", [AMC.replace(":DEGREES", ":RADIANS"),
                                  AMC.replace("bone 0 0 90\n", ""),
                                  AMC.replace("2\nroot", "3\nroot"),
                                  AMC.replace("bone 0 0 90", "bone 0 0 nan"),
                                  AMC + "bone 0 0 0\n"])
def test_bad_motion_rejected(text):
    with pytest.raises(ValueError):
        parse_amc(text, parse_asf(ASF)[1])


@pytest.mark.parametrize("text", [ASF.replace("root bone", "bone bone"),
                                  ASF.replace("length 0.45", "length 0"),
                                  ASF.replace("0 0 0 XYZ", "0 0 0 ZYX")])
def test_bad_skeleton_rejected(text):
    with pytest.raises(ValueError):
        parse_asf(text)


def test_load_checks_provenance_and_keeps_pilot_unlabelled(tmp_path):
    names = list(dict.fromkeys(_JOINTS[1:]))
    asf = ASF.split(":bonedata")[0] + ":bonedata\n"
    for index, name in enumerate(names):
        asf += (f"begin\nid {index + 1}\nname {name}\ndirection 0 1 0\n"
                "length 1\naxis 0 0 0 XYZ\nend\n")
    asf += ":hierarchy\nbegin\nroot " + " ".join(names) + "\nend\n"
    amc = ":FULLY-SPECIFIED\n:DEGREES\n1\nroot 0 0 0 0 0 0\n"
    entries = []
    for name, content in {"source.html": "pilot", "07.asf": asf, "07_01.amc": amc}.items():
        data = content.encode()
        (tmp_path / name).write_bytes(data)
        entries.append({"file": name, "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"format": "poseahead-cmu-download", "version": 1, "subject": "07",
                "trial": "01", "fps_numerator": 120, "files": entries}
    (tmp_path / "download.json").write_text(json.dumps(manifest))
    sequence = load_cmu(tmp_path)
    assert sequence_split(sequence) == ("unlabelled", None)
    assert sequence.frames[0].source_timestamp_ns == 0
    assert sequence.frames[0].metadata["cmu_subject"] == "07"
    np.testing.assert_allclose(sequence.frames[0].joint_positions[1],
                               [0, -0.0254 / 0.45, 0])
    (tmp_path / "07_01.amc").write_text(amc + "2\n")
    with pytest.raises(ValueError, match="checksum"):
        load_cmu(tmp_path)
