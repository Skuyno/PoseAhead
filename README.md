# PoseAhead

PoseAhead is a research platform for predictive latency compensation of a
video-driven full-body Unity avatar. Python owns perception and the
causal processing/evaluation pipeline; Unity receives versioned binary UDP.

## Current state

- desktop control panel: camera/NPZ/diagnostic source, processing selection,
  graceful stop, producer metrics, session artifacts, persistent local preferences,
  live session-matched Unity receipt/error feedback and pinned Unity launch;
- immutable canonical 17-joint `PoseFrame` and storage serialization;
- immutable `PoseSequence` with clock/order checks and an adapter for prepared
  Human3.6M camera-space ground-truth arrays;
- current-frame MeTRAbs PyTorch backend with estimated metric root translation;
- causal pose-guided person-box tracking with periodic YOLO redetection;
- latest-frame webcam capture and still/video/webcam benchmark tools;
- opt-in FP16 convolution storage with FP32 BatchNorm parameters and unchanged geometry,
  a calibration-preserving fast-profile UI button and per-stage camera latency;
- UDP v2 sender, Python decoder, loopback-only Unity receiver and avatar-container
  translation;
- position-based Humanoid bone driver, saved diagnostic scene and display telemetry;
- native Unity Humanoid/UDP tests, eight rendered diagnostic phases and a
  Python-to-Unity CMU replay check; real person-camera -> Unity sustained delivery
  is measured for ten minutes, while live timing requirements remain unmet;
- localhost recording/diagnostic replay without changing research timestamps;
- hardware-free Python and C# regression suites;
- pickle-free NPZ recording/replay with separate derived runs and missing observations;
- shared Raw, EMA, One-Euro, Kalman and fixed-horizon Kalman extrapolation;
- causal delayed simulation and target-aligned canonical MPJPE/root-error scoring;
- acceleration-residual jitter and window-level effective-latency APIs with coverage;
- config-driven experiments, seeded joint noise/dropout/outliers, missing-arrival bursts,
  source-rate stress and artifact checks;
- split-checked batch experiments, descriptive per-sequence summaries and pinned dataset indexes;
- paired sequence-bootstrap comparisons, validation-frozen comparator checks and Pareto summaries;
- pinned robust validation-panel comparator selection with full condition/seed coverage,
  clean error/jitter constraints and explicit non-comparable latency;
- split-safe, sharded predictor windows for 33/66/100/150 ms supervision;
- checked window loading, train-only normalization and a fixed-horizon CPU MLP training prototype;
- checksum-pinned CMU ASF/AMC import, source-ID splits and validation-only training studies;
- clip-consistent train-only noise augmentation, causal history filtering and optional
  constant-velocity residual checkpoints;
- bounded learned offline processing with explicit raw fallback and checkpoint-pinned experiments;
- experimental causal variable-rate learned processing and per-frame UDP prediction/fallback IDs;
- experimental M8 four-checkpoint adaptive processor with mature-only error tracking,
  explicit threshold policy, horizon/blending ablation and persisted decision logs;
- optional causal windowed dynamics and source-time current-forecast weight transitions,
  with immediate tracking-loss reset and separately logged requested/output decisions;
- optional same-target Kalman-CV/neural strength adaptation with paired mature errors,
  capped neural weight and explicitly identified Kalman-only fallback;
- train-only frozen ground-truth acceleration classes and descriptive class-wise validation;
- explicit prepared Human3.6M NPY/JSON import, camera extrinsics and subject split labels.

CMU-trained checkpoints and synthetic delay/noise validation are available, but a
robust learned advantage over strong baselines has not been established.
The first 72-run adaptive validation pilot also retains Kalman-CV: clean jitter
exceeds the budget and the raw acceleration gate disables noisy prediction.
The follow-up 72-run window/history/transition ablation passes the clean 10% gate,
but noisy jitter still prevents selection; neither pilot promotes an adaptive live default.
The subsequent same-target Kalman/noise-trained-h100 pilot selects a capped mixture
with lower macro error/jitter in all four Gaussian profiles. A subsequent three-noise-seed
confirmation repeats that improvement, but broader stress retains Kalman-CV because
partial-joint dropout resets degrade the adaptive output. A separate 54-cell per-joint
baseline recovery ablation corrects that regression and slightly beats Kalman-CV
on the tested rare-dropout profile. Neither matched latency nor held-out superiority
is established. The additional SPEC outlier/missing/tracking-loss panel does not
pass full-panel coverage; on frequent missing joints the neural part is disabled
and output equals Kalman-CV. A saved 60-frame MeTRAbs smoke is also Kalman-only;
the fixed comparator and live defaults remain Kalman-CV and the user's existing settings.
Live timing budgets/post-render-cap measurements, imported-character acceptance, real-camera
learned acceptance, adaptive threshold calibration/acceptance, official
Human3.6M CDF decoding, participant-level uncertainty and final held-out evaluation
remain unfinished. See
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for the ordered work.
For the predeclared 100-ms robustness continuation, see
[robust comparator selection](docs/robust_selection.md).
For the experimental adaptive model bank and validation pilot, see
[adaptive compensation](docs/adaptive_compensation.md).
For completed multi-seed/broader stress and train-only motion classes, see
[frozen adaptive confirmation](docs/adaptive_confirmation.md).

## Requirements and base setup

- Windows 11 x64;
- Python 3.10 and Poetry 2.x;
- NVIDIA GPU; the pinned experimental MeTRAbs PyTorch path currently requires CUDA;
- Unity 6.3 LTS `6000.3.23f1`.

```powershell
poetry env use C:\Users\maksa\AppData\Local\Programs\Python\Python310\python.exe
poetry install
.\scripts\setup_metrabs.ps1
poetry run poseahead doctor
poetry run poseahead check-config configs/live/default.json
```

## Desktop control panel

For a new Windows laptop, see the [Russian installation and offline demo guide](docs/laptop_demo.md).
It covers Python/Unity installation, the required recordings/checkpoints, CPU-only
dependencies and the four demo presets. This is source setup, not a ready-made EXE.

Camera speed option: choose Camera, then **«Быстрый профиль»** beside the live JSON.
This creates a separate config preserving current calibration/settings; it does
not start capture or overwrite your config. The recorded-motion ABBA check improved
aggregate throughput from 26.27 to 27.99 valid poses/s with exactly matching saved
pose/root/confidence/masks/tracking on the tested inputs. This is not a new live
measurement or end-to-end latency acceptance. Default remains `autocast`.
See [performance protocol and limits](docs/metrabs_performance.md).

Advisor meeting, 20 October: [four-page report with speech and demo steps](output/pdf/poseahead_advisor_report_20261020.pdf),
[separate three-minute speech](docs/advisor_speech_20261020.md),
[editable report](docs/advisor_report_20261020.md) and
[local browser/print version](docs/advisor_report_20261020.html).
The original topic name is retained. This is a working-prototype progress report,
not final acceptance: validation error/jitter gains are separate from unproven
latency/held-out/live superiority. All four final PDF pages were rendered and reviewed.

Earlier operator instructions: [DOCX guide](docs/demo_guide.docx),
[Russian text guide](docs/demo_guide.md).
The UI now offers four explicit presentation presets: diagnostic poses, CMU +
Kalman-CV, CMU + experimental adaptive recovery, and a saved MeTRAbs recording.
Applying a preset never starts a session or opens a camera. Quantitative saved
results and verified source hashes are in `outputs/advisor_demo_20261020/`, including
the saved 18 October ABBA performance probes.
These are repeated-validation evidence, not established latency or live superiority.

```powershell
.venv/Scripts/python.exe scripts/check_demo.py --load-models
.venv/Scripts/python.exe scripts/prepare_demo.py --include-performance --output outputs/demo_NEW
```

Both commands are offline/local checks with no UDP or camera. The preflight parses
both live profiles and the pinned Unity version, rather than only checking existence.
Preparation requires a new output directory, reproduces selection from pinned batch
artifacts and, with `--include-performance`, verifies saved ABBA arrays/checksums
without CUDA. The legacy option without this flag remains available.
The older DOCX has structural QA only; its page rendering was unavailable without
LibreOffice. The new PDF is visually verified, with no new DOCX claimed.

On Windows, double-click `StartPoseAhead.cmd` in the project root (uses the existing
`.venv`, opens no console). Or launch explicitly:

```powershell
.venv/Scripts/python.exe main.py gui
# Installed entry point: poetry run poseahead gui
```

Choose diagnostic poses, an existing NPZ recording, or a MeTRAbs camera session.
Select a baseline or an explicit processor JSON for experimental MLP checkpoints,
then start/stop with buttons. Open the pinned Unity project separately, load
`PoseAheadDiagnostic 1` and press Play; default UDP port is 5005 on both sides.
The panel reports producer data and, after a run, matching local Unity telemetry.
Sending alone is not confirmed reception; absent/stale telemetry stays unconfirmed.
Capture-to-send p95 is not full camera-to-display latency. It does not auto-select
models or assess scientific acceptance. Camera/GPU dependencies are still required
only for the camera source. Tkinter is part of the existing Python installation;
no web server, Qt install or network-setting change is needed.

Each session gets a new ignored `outputs/ui_...` directory with request/config,
worker log, producer `report.json` and combined `run_report.json`; optional raw
observations are saved on normal stop. No source recording is overwritten.
See [`desktop UI guide`](docs/desktop_ui.md).
Module boundaries and compatibility are documented in [refactoring notes](docs/refactoring.md).

Base-only wheel installation can be checked in a fresh venv, without a camera:

```powershell
.venv/Scripts/python.exe scripts/verify_base_install.py
```

This downloads build/base dependencies into an isolated environment, not the working
`.venv`; it checks all five analytic processors, NPZ, loopback UDP, lazy GUI imports
and installed CLI entry points. It does not validate CUDA or a clean operating system.

`setup_metrabs.ps1` installs the pinned CUDA inference dependencies, checks out
MeTRAbs commit `8b2b116`, downloads the 94.8 MiB EfficientNetV2-S pose model and
YOLOv8n detector, verifies SHA-256 before model loading, and performs a GPU load
smoke test. Downloads live under ignored `.vendor/`.

The MeTRAbs source is MIT licensed. The authors state that released model weights
are restricted to non-commercial use because of their training datasets. This
repository therefore treats the backend as a research/education component.

## Commands

```powershell
poetry run python scripts/benchmark_metrabs.py path/to/person.jpg
poetry run python scripts/benchmark_video.py path/to/motion.mp4
poetry run python scripts/benchmark_webcam.py --config configs/live/default.json --duration 30
poetry run python scripts/benchmark_webcam.py --config configs/live/default.json --send-udp
poetry run python scripts/stress_metrabs_video.py path/to/motion.mp4 --duration 600
```

The default `55°` field of view is only an assumption. Supplying measured
`camera_intrinsics` in the live JSON improves geometric fidelity and must match
the source frame resolution. Inputs larger than 640 pixels on their longest side
are downscaled for inference and calibrated intrinsics are scaled with them. No
floor marking or support-foot detection is used.

The original EfficientNetV2-L probe produced about 13 valid FPS. The pinned M1
fallback now uses the official EfficientNetV2-S checkpoint, YOLOv8n, and one pose
crop per frame. On the reference RTX 3070 Ti, the recorded jump sample produced
valid poses on every measured frame at roughly 22-24 FPS; a synthetic 640-pixel
stream reached 25.33 FPS with an earlier four-frame redetection configuration.
The final ten-frame tracking profile measured 23.33 FPS over ten minutes.
This documented fallback unblocks downstream work but
does not prove ground-truth metric accuracy. MeTRAbs estimates depth from one RGB
image, so its root may jitter or drift. See [`docs/metrabs.md`](docs/metrabs.md).
The full gate disposition and reproduction commands are in
[`docs/m1_feasibility.md`](docs/m1_feasibility.md).

## Regression checks

Recording, processing and scoring commands are documented in
[`docs/recording_processing.md`](docs/recording_processing.md). Prepared dataset
input conventions are in [`docs/human36m.md`](docs/human36m.md).
One-command baseline runs and synthetic sanity inputs are described in
[`docs/experiments.md`](docs/experiments.md).

```powershell
poetry run ruff check .
poetry run pytest
dotnet run --no-restore --project tests/dotnet/PoseAhead.Protocol.Smoke --configuration Release
```

Tests use simulated models/cameras except for an explicitly run MeTRAbs benchmark.
The UDP receiver binds only to `127.0.0.1`; this project does not alter VPN or
network-adapter settings.
