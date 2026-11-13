# M1 MeTRAbs Feasibility Result

Original measurements: 2026-09-10; live status updated 2026-10-07.
Reference machine: Ryzen 7 5700X, RTX 3070 Ti 8 GB,
16 GB RAM, Windows 11.

## Decision

M1 remains partially complete: recorded inference and its documented fallback
are verified; person-bearing live delivery/stability are now measured, but live
throughput and processing latency do not meet SPEC. The earlier static-camera
0/150-pose observation is historical and no longer describes the usable feed.

The 2026-10-05 raw camera-0 -> Unity run completed 600.01 seconds. It produced
10,809 valid poses from 11,105 processed observations (97.33%); every sent pose was
received and applied by Unity, with no rejected packets/receiver/rig errors.
RAM-growth and reserved-VRAM-growth gates pass. Capture accounting is exact:
18,000 = 11,105 processed + 6,894 latest-slot overwrites + 1 pending.
Performance is 18.01 valid FPS, capture-to-send median 74.04 ms and p95 98.43 ms,
below the >=25 FPS and above the <=50-ms p95 goals. This is not capture-to-screen
latency or ground-truth accuracy. Artifact: `outputs/ui_20261005_140958_4ed33ebe/`.
Unity then had about 706.62 Game-camera render callbacks/s. A scoped 60-FPS target
is implemented; a person-bearing post-cap measurement is still required before
attributing the slowdown or claiming an improvement. Raw NPZ recording was off.

The selected stack is:

- MeTRAbs source commit `8b2b116dd27372e7dbd8207809f868df4e3f852e`;
- official `metrabs_eff2s_256px_800k_28ds_pytorch` checkpoint;
- YOLOv8n at 320 px for person detection;
- one pose augmentation per frame;
- inference input capped at 640 pixels on the longest side;
- YOLO redetection every ten processed frames, with the box updated from the
  current frame's MeTRAbs 2D joints between redetections.

All model archives/checkpoints and the source revision are pinned and verified.
The tracking state uses only current and past observations; no future frame is
available to the backend.

| Artifact | SHA-256 |
|---|---|
| MeTRAbs S archive | `d6861899d0c05a93a4bae078dca5382186f04d568255eb688ace1c7dea7a7634` |
| MeTRAbs `ckpt.pt` | `82833e7b0d9bf282271e17aeae66e69cc2b1070a21d94540ad864313c477962a` |
| MeTRAbs `config.yaml` | `147e259ef04457c0cfadd24a44a961057f78399b2da5aa5625627fb75aa95c94` |
| YOLOv8n | `31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95` |

## Why the backend changed

The original EfficientNetV2-L 384 px plus YOLOv8m configuration measured about
75 ms per frame (roughly 13 FPS). Stage profiling attributed about 52 ms to the
pose network and 14 ms to detection. Running the same large checkpoint with a
256 px crop barely changed the result, so that unsupported resolution override
was rejected.

The official MeTRAbs download index also publishes a native EfficientNetV2-S
256 px PyTorch checkpoint. That compatible checkpoint and the smaller detector
were selected instead. This is an explicit speed/accuracy trade-off; comparative
ground-truth accuracy must be measured later on Human3.6M and is not inferred from
visual inspection.

## Measurements

| Probe | Validity | FPS | Median | p95 | Peak allocated CUDA |
|---|---:|---:|---:|---:|---:|
| Synthetic 640x426 recorded stream | 85/85 | 25.33 | 36.63 ms | 47.26 ms | not retained in this report |
| 960x540 power-jump motion, short | 289/289 | 22.38 | 41.77 ms | 54.62 ms | about 293 MiB |
| Power-jump loop, 600.01 s | 13,999/13,999 | 23.33 | 40.23 ms | 53.08 ms | 292.82 MiB |

The 25.33 FPS synthetic result used the intermediate configuration with redetection
every four frames, before pose-guided box updates were introduced. It must not be
used as evidence that the final ten-frame tracking configuration passes 25 FPS.

The ten-minute recorded fallback completed 46 rewinds without a crash. Process
working set grew by 3.98 MiB and CUDA reserved memory by 0 MiB. The declared
stability limits are at least 590 seconds, at most 128 MiB working-set growth,
and at most 32 MiB CUDA reserved-memory growth; the recorded fallback passes.

The 25 valid-pose FPS threshold is reached on the simpler stream but not on the
jump clip. Per the M1 acceptance rule, the measured 22-24 FPS motion configuration
is retained as the documented fallback so later causal processing can proceed.
No TensorRT conversion is introduced at this stage because it would add an
unverified deployment path after a compatible official small checkpoint already
reduced the main risk.

## Output inspection

- MeTRAbs output is mapped to the canonical H36M-derived 17-joint skeleton.
- Positions are converted from millimetres to metres.
- The pelvis becomes `root_position`; canonical joints are pelvis-relative.
- All measured sample frames contain 17 finite joints and a root estimate.
- Detector confidence is a person-level proxy, not per-joint calibrated
  uncertainty.
- A 60-frame PAFR sample round-trips through the versioned decoder. Its p95
  frame-to-frame root step is 0.0234 m and mean per-bone temporal coefficient of
  variation is 0.0063. These are continuity diagnostics, not accuracy metrics.

## Reproduction

```powershell
.\scripts\setup_metrabs.ps1
poetry run python scripts/profile_metrabs_stages.py `
  .vendor/metrabs-source/test_image_3dpw.jpg --warmup 10 --iterations 60
poetry run python scripts/benchmark_video.py path/to/motion.mp4 `
  --input-kind motion --sample-dir outputs/fixtures/m1_pose_sample
poetry run python scripts/stress_metrabs_video.py path/to/motion.mp4 `
  --duration 600 --output outputs/benchmarks/m1_stress_10min_recorded_motion.json
poetry run python scripts/analyze_pose_sample.py outputs/fixtures/m1_pose_sample
poetry run python scripts/benchmark_webcam.py --duration 600
```

The last command can reproduce the live-input check with a full-body person in
view; the initial ten-minute live stability/delivery check has now been performed.
Further matched measurements must report capture
drops, valid FPS, capture-to-pose p95, GPU samples, and no memory trend/crash.

The motion fixture used locally is a ten-second excerpt from Wikimedia Commons'
public-domain U.S. Army video [“Conditioning Drill 1 - Power
Jump”](https://commons.wikimedia.org/wiki/File:Conditioning_Drill_1-_Power_Jump.webm).
Official model files are listed in the [MeTRAbs download
index](https://omnomnom.vision.rwth-aachen.de/data/metrabs/).

## Follow-up fixes (2026-09-21)

Invalid tracked predictions now clear the cached box and force detection on the
next frame. Previously a failed prediction could leave the age counter unchanged
and reuse the stale box indefinitely. Empty detections are signalled explicitly
before entering upstream pose estimation; unrelated RuntimeErrors propagate.
Regression tests exercise the production tracking branch, recovery, periodic
redetection, and frame-ID restart without GPU dependencies. The ten-minute figures
above describe the September 10 build; they have not been remeasured after these
failure-path fixes.

Validation of these fixes: 149 Python tests passed and Ruff passed.
