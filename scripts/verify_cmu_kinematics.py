"""Compare importer matrices with independent SciPy quaternion FK on real clips.

The recurrence is checked against the official CMU amc_viewer.r1806 source:
Library/Skeleton.cpp build_pose and Character/pose_utils.cpp get_world_bones.
This checks rotation math, not independent ASF decoding or anatomical accuracy.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from poseahead.artifacts import sha256_file
from poseahead.datasets.cmu import forward_kinematics, parse_amc, parse_asf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for directory in args.directories:
        manifest = json.loads((directory / "download.json").read_text())
        subject, trial = manifest["subject"], manifest["trial"]
        asf, amc = directory / f"{subject}.asf", directory / f"{subject}_{trial}.amc"
        skeleton = parse_asf(asf.read_text())
        scale, bones, parents, ordered = skeleton
        motion = parse_amc(amc.read_text(), bones)
        maximum = 0.
        for _, channels in motion:
            # SciPy composes normalized quaternions, separate from production matrices.
            orientations = {"root": Rotation.from_euler("xyz", channels["root"][3:],
                                                        degrees=True)}
            points = {"root": channels["root"][:3] * scale}
            for name in ordered:
                offset, basis, dof = bones[name]
                angles = np.zeros(3)
                for channel, value in zip(dof, channels.get(name, ()), strict=True):
                    angles[("rx", "ry", "rz").index(channel)] = value
                c = Rotation.from_matrix(basis)
                orientations[name] = (orientations[parents[name]] * c
                                      * Rotation.from_euler("xyz", angles, degrees=True) * c.inv())
                points[name] = points[parents[name]] + orientations[name].apply(offset)
            actual = forward_kinematics(skeleton, channels)
            maximum = max(maximum, max(float(np.linalg.norm(actual[n] - points[n]))
                                      for n in points))
        if maximum > 1e-10:
            raise ValueError(f"quaternion FK mismatch: {maximum}")
        reports.append({"clip": f"{subject}/{trial}", "frames": len(motion),
                        "max_endpoint_difference_m": maximum,
                        "asf_sha256": sha256_file(asf), "amc_sha256": sha256_file(amc)})
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"comparison": "scipy_quaternion_vs_numpy_matrix_shared_asf_parser",
                   "clips": reports}, stream, indent=2)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
