# MeTRAbs live observation backend

PoseAhead uses the official experimental PyTorch implementation of MeTRAbs at a
pinned source revision. It predicts absolute 3D joints in millimetres from each
current RGB frame. PoseAhead converts these to metres, stores the predicted pelvis
as `root_position`, and makes all canonical joints pelvis-relative.

This removes the former floor-plane/support-foot estimator completely. There is
no body-length or floor-contact fallback and no floor-point calibration command.
A jump can therefore produce a root on airborne frames instead of being withheld
solely because neither foot touches the floor.

## Installation and files

Run:

```powershell
.\scripts\setup_metrabs.ps1
```

The script pins:

- MeTRAbs source commit `8b2b116dd27372e7dbd8207809f868df4e3f852e`;
- official `metrabs_eff2s_256px_800k_28ds_pytorch` weights;
- official YOLOv8n detector weights;
- Torch 2.1.0/CUDA 12.1, torchvision 0.16.0, Ultralytics 8.0.145 and einops 0.8.1.

The two model files come from the authors' [official MeTRAbs download
index](https://omnomnom.vision.rwth-aachen.de/data/metrabs/); source code is pinned
from the [official repository](https://github.com/isarandi/metrabs).

PoseAhead verifies every required model file before deserializing it. The runtime
does not deserialize the upstream `skeleton_infos.pkl`; the audited H36M-17 names,
indices and edges are fixed in source. `torch.load(..., weights_only=True)` is used
for the pose state dictionary.

Only the inference subset of PosePile is recreated internally. This avoids pulling
the upstream dataset-preparation/TensorFlow dependency graph into the live Windows
environment.

## Output and confidence

The source skeleton is H36M-17. Direct limbs are mapped by name. Canonical chest is
the midpoint of the source spine and neck. The source head point is used; the head-
top point is not transmitted.

MeTRAbs does not return a per-joint confidence vector. PoseAhead repeats the chosen
person detector score as a proxy and labels metadata with
`confidence_semantics=person_detection_proxy_no_per_joint_scores`. This value must
not be analysed as calibrated per-joint uncertainty.

Successful frames use:

```text
root_translation_status = estimated_monocular
scale_status = metrabs_metric_estimate
backend = metrabs-eff2s-pytorch
```

The UDP estimated-root flag remains set. Neither the flag nor the word `metric`
means measured ground truth.

## Camera geometry

If measured `camera_intrinsics` are supplied, their resolution must equal the
source frame. Frames larger than `max_input_side=640` are resized isotropically;
the intrinsic matrix is scaled by the same X/Y factors before MeTRAbs receives
it. Otherwise the backend constructs a centered pinhole camera using
`default_fov_degrees=55` across the processed image and records
`intrinsics_source=assumed_fov`.

The canonical result remains in camera axes: +X image-right, +Y down, +Z away.
Unity flips Y and anchors the first received root at the current object position.
For correct room-horizontal movement the camera should initially be level; camera
extrinsic/orientation calibration is not implemented.

## Causality and limitations

Every pose estimate uses the current image. YOLO runs on the first frame and then
every `detector_interval=10` processed frames. Between detections the crop box is
updated from the current frame's projected 2D joints, so the only temporal state
is a box from the current/past frames. There is no future-frame context, root
smoothing, contact dwell, or hidden temporal refinement. A non-increasing frame
ID clears tracking state and forces redetection.

The detector score is reused as the confidence proxy between redetections. The
metadata fields `detector_mode`, `bbox_age_processed_frames`, and
`tracking_box_updated` make that state observable. If a redetection finds nobody,
the backend returns `None` and retries detection on the next frame instead of
crashing inside the upstream empty-box path.

The method emitted valid poses throughout the local power-jump sample, including
crouching, take-off, airborne, and landing frames. This checks execution and
detection continuity, not 3D accuracy. A single RGB view has depth/scale ambiguity;
fast motion blur, self-occlusion, truncation and unusual poses can still cause root
jitter, depth jumps or detection loss. On loss the Unity container holds its last
position and reanchors when a new sender session starts.

The initial EfficientNetV2-L/YOLOv8m repeated-still smoke on the reference RTX
3070 Ti produced about 13 valid FPS and 75 ms inference with `num_aug=1`. Stage
profiling showed roughly 52 ms in the pose network and 14 ms in the detector, so
changing crop resolution alone could not meet the gate.

The documented M1 fallback uses the separately published EfficientNetV2-S 256 px
checkpoint, YOLOv8n at 320 px, pose-guided crop tracking, and one augmentation.
Measured results are stored under ignored `outputs/benchmarks/`:

- synthetic 640x426 video: 25.33 valid frames/s, median 36.63 ms, p95 47.26 ms;
- 960x540 power-jump video (internally 640x360): every measured frame valid,
  roughly 22-24 valid frames/s, median about 39-42 ms, p95 about 50-55 ms;
- peak allocated CUDA memory in short probes: about 293-322 MiB.

The synthetic result used an intermediate four-frame redetection configuration
without pose-guided updates; it does not establish final-profile throughput.
The 25 FPS target was therefore met only in that intermediate probe. The
motion result is accepted as an explicit feasibility fallback rather than being
misreported as a throughput pass. M1 defines stability as at least 590 seconds
without a crash, no more than 128 MiB process working-set growth, and no more than
32 MiB CUDA reserved-memory growth after warm-up. The recorded-motion stress test
processed 13,999/13,999 valid frames over 600.01 seconds at 23.33 FPS (median
40.23 ms, p95 53.08 ms), with 3.98 MiB working-set growth and 0 MiB CUDA
reserved-memory growth. It passed the recorded fallback thresholds; it is not
the live-camera result. The historical static index-0 zero-pose probe was
superseded by a usable person-bearing 600-second raw camera -> Unity run:
10,809/11,105 valid processed observations, all 10,809 received/applied, passing
RAM/reserved-VRAM growth checks. The live result is 18.01 valid FPS with capture-to-send
p95 98.43 ms, not a >=25-FPS / <=50-ms pass. Artifact:
`outputs/ui_20261005_140958_4ed33ebe/`. A new Unity render target of 60 FPS is
implemented but not yet remeasured with a person; no improvement is inferred.

The local motion fixture is a ten-second excerpt of the public-domain U.S. Army
video [“Conditioning Drill 1 - Power Jump”](https://commons.wikimedia.org/wiki/File:Conditioning_Drill_1-_Power_Jump.webm)
from Wikimedia Commons. Generated media, benchmarks, and 60 serialized PAFR
sample frames remain under ignored `outputs/`.

The published weights are restricted by the authors to non-commercial use. Do not
reuse this backend in a commercial product without resolving the model/data licence.
