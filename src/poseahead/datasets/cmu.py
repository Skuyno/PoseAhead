"""Strict CMU ASF/AMC pilot importer (XYZ axes, fully specified degree channels).

CMU lengths are inches divided by the ASF length scale. Coordinates are mapped
to a fixed virtual-camera frame with diag(1,-1,-1), not a calibrated real camera.
No training split is silently inferred from a subject number.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from poseahead.core import JOINT_COUNT, PoseFrame, PoseSequence
from poseahead.core.sequence import frame_timestamp_ns

_JOINTS = ("root", "lowerback", "upperback", "thorax", "upperneck",
           "lclavicle", "lhumerus", "lradius", "rclavicle", "rhumerus", "rradius",
           "lhipjoint", "lfemur", "ltibia", "rhipjoint", "rfemur", "rtibia")


def rotation_xyz(angles):
    """Extrinsic XYZ in degrees, acting on column vectors: Rz @ Ry @ Rx."""
    x, y, z = np.deg2rad(angles)
    cx, cy, cz = np.cos([x, y, z])
    sx, sy, sz = np.sin([x, y, z])
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def _lines(text):
    return [line.split("#", 1)[0].strip() for line in text.splitlines()
            if line.split("#", 1)[0].strip()]


def parse_asf(text):
    sections = {}
    section = None
    for line in _lines(text):
        if line.startswith(":"):
            section = line.split()[0]
            sections[section] = []
        elif section:
            sections[section].append(line)
    try:
        units = dict(line.split(maxsplit=1) for line in sections[":units"])
        root = dict(line.split(maxsplit=1) for line in sections[":root"])
        if (units["angle"] != "deg" or root["axis"] != "XYZ"
                or root["order"] != "TX TY TZ RX RY RZ"
                or any(float(v) != 0 for key in ("position", "orientation")
                       for v in root[key].split())):
            raise ValueError("unsupported ASF root/angle convention")
        scale = 0.0254 / float(units["length"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("invalid ASF scale")
        bones, block = {}, {}
        for line in sections[":bonedata"]:
            if line == "begin":
                block = {}
            elif line == "end":
                name = block["name"][0]
                if name in bones or name == "root":
                    raise ValueError("duplicate ASF bone")
                axis = block["axis"]
                if len(axis) != 4 or axis[3] != "XYZ":
                    raise ValueError("only XYZ ASF bone axes are supported")
                direction = np.array(block["direction"], dtype=float)
                length = float(block["length"][0])
                angles = np.array(axis[:3], dtype=float)
                dof = tuple(block.get("dof", ()))
                if (direction.shape != (3,) or not np.isfinite(direction).all()
                        or not np.isfinite(angles).all() or not np.isfinite(length)
                        or length < 0 or len(set(dof)) != len(dof)
                        or any(v not in ("rx", "ry", "rz") for v in dof)
                        or list(dof) != sorted(dof)):
                    raise ValueError("invalid or unsupported ASF bone")
                bones[name] = (direction * length * scale, rotation_xyz(angles), dof)
            else:
                parts = line.split()
                block[parts[0]] = parts[1:]
        parents = {}
        for line in sections[":hierarchy"]:
            if line in ("begin", "end"):
                continue
            parent, *children = line.split()
            for child in children:
                if child in parents or child not in bones or parent not in {*bones, "root"}:
                    raise ValueError("invalid ASF hierarchy")
                parents[child] = parent
        if set(parents) != set(bones):
            raise ValueError("incomplete ASF hierarchy")
        ordered, remaining = [], set(bones)
        while remaining:
            ready = sorted(n for n in remaining if parents[n] == "root"
                           or parents[n] in ordered)
            if not ready:
                raise ValueError("cyclic ASF hierarchy")
            ordered.extend(ready)
            remaining.difference_update(ready)
        return scale, bones, parents, ordered
    except (KeyError, IndexError, ZeroDivisionError) as error:
        raise ValueError("incomplete ASF skeleton") from error


def parse_amc(text, bones):
    lines = _lines(text)
    if ":FULLY-SPECIFIED" not in lines or ":DEGREES" not in lines:
        raise ValueError("AMC must be fully specified in degrees")
    expected = {"root": 6, **{name: len(bone[2]) for name, bone in bones.items() if bone[2]}}
    frames, current, frame_id = [], {}, None

    def finish():
        if frame_id is not None:
            if current.keys() != expected.keys():
                raise ValueError("incomplete AMC frame channels")
            frames.append((frame_id, current.copy()))

    for line in lines:
        if line.startswith(":"):
            if line not in (":FULLY-SPECIFIED", ":DEGREES"):
                raise ValueError("unsupported AMC directive")
            continue
        if line.isdigit():
            finish()
            next_id = int(line)
            if next_id != (1 if frame_id is None else frame_id + 1):
                raise ValueError("AMC frames must start at 1 and be contiguous")
            frame_id, current = next_id, {}
        else:
            name, *values = line.split()
            array = np.array(values, dtype=float)
            if (frame_id is None or name in current or name not in expected
                    or len(array) != expected[name] or not np.isfinite(array).all()):
                raise ValueError("invalid AMC channels")
            current[name] = array
    finish()
    if not frames:
        raise ValueError("empty AMC motion")
    return frames


def forward_kinematics(skeleton, channels):
    scale, bones, parents, ordered = skeleton
    positions = {"root": channels["root"][:3] * scale}
    rotations = {"root": rotation_xyz(channels["root"][3:])}
    for name in ordered:
        offset, basis, dof = bones[name]
        angles = np.zeros(3)
        for channel, value in zip(dof, channels.get(name, ()), strict=True):
            angles[("rx", "ry", "rz").index(channel)] = value
        parent = parents[name]
        rotations[name] = rotations[parent] @ basis @ rotation_xyz(angles) @ basis.T
        positions[name] = positions[parent] + rotations[name] @ offset
    return positions


def load_cmu(directory):
    """Import a checksum-verified download as an unlabelled pilot recording."""
    directory = Path(directory)
    manifest = json.loads((directory / "download.json").read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict)
            or manifest.get("format") != "poseahead-cmu-download"
            or type(manifest.get("version")) is not int or manifest["version"] != 1):
        raise ValueError("unsupported CMU download manifest")
    for key in ("subject", "trial"):
        value = manifest.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9]{2,3}", value):
            raise ValueError("invalid CMU subject/trial ID")
    fps = manifest.get("fps_numerator")
    frame_timestamp_ns(0, fps_numerator=fps)
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != 3:
        raise ValueError("CMU manifest requires three source files")
    sources = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid CMU source entry")
        name = entry.get("file")
        if (not isinstance(name, str) or not name or Path(name).name != name
                or "/" in name or "\\" in name or ":" in name or name in sources):
            raise ValueError("invalid CMU source filename")
        data = (directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
            raise ValueError("CMU source checksum mismatch")
        sources[name] = data.decode("utf-8")
    subject, trial = manifest["subject"], manifest["trial"]
    if set(sources) != {"source.html", f"{subject}.asf", f"{subject}_{trial}.amc"}:
        raise ValueError("CMU manifest source set does not match subject/trial")
    skeleton = parse_asf(sources[f"{subject}.asf"])
    if not set(_JOINTS) <= {"root", *skeleton[1]}:
        raise ValueError("CMU skeleton is missing canonical landmarks")
    motion = parse_amc(sources[f"{subject}_{trial}.amc"], skeleton[1])
    sequence_id = f"cmu/{subject}/{trial}"
    clock = f"dataset:{sequence_id}:virtual-camera"
    frames = []
    for frame_id, channels in motion:
        positions = forward_kinematics(skeleton, channels)
        absolute = np.array([positions[name] for name in _JOINTS]) * [1, -1, -1]
        absolute[2] = (absolute[1] + absolute[3]) / 2
        timestamp = frame_timestamp_ns(frame_id - 1, fps_numerator=manifest["fps_numerator"])
        frames.append(PoseFrame(
            frame_id=frame_id - 1, source_timestamp_ns=timestamp,
            produced_timestamp_ns=timestamp, joint_positions=absolute - absolute[0],
            root_position=absolute[0], joint_confidence=np.ones(JOINT_COUNT),
            joint_validity=np.ones(JOINT_COUNT, dtype=bool), metadata={
                "dataset": "CMU", "cmu_subject": subject, "cmu_trial": trial,
                "clock_domain": clock, "source_skeleton": "cmu_asf",
                "coordinate_frame": "virtual_camera_diag_1_minus1_minus1",
                "timestamp_semantics": "dataset_event_time_no_processing_delay",
                "root_translation_status": "mocap_reconstruction",
                "confidence_semantics": "source_validity_not_detector_probability",
                "source_sha256": {entry["file"]: entry["sha256"]
                                  for entry in manifest["files"]},
                "fps_numerator": manifest["fps_numerator"],
                "derived_joints": ["chest"], "usage": "pilot_not_training_split",
            }))
    return PoseSequence(sequence_id, clock, tuple(frames))
