# PoseAhead Implementation Plan

Advisor demonstration milestone (2026-10-20): refreshed all four offline presets
and CPU checkpoint smoke; demo preflight now validates the actual pinned Unity
version and both standard/fast live JSONs. Optional `prepare_demo.py
--include-performance` also verifies the saved clean motion/still ABBA manifests
and arrays, adding pooled timings/identity and source hashes to the evidence
handout without CUDA, camera or default promotion. Nine saved batches reproduce
252 experiments / 2,025 verified files at outputs/advisor_demo_20261020.
977 Python tests and Ruff pass; four native Unity tests pass with CMU replay at
outputs/unity_native_20261020_161444_aaf6c15a. Actual Tk Stop/restart/preferences
and 15 locally decoded UDP packets pass at outputs/advisor_ui_20261020; receiver
display injections there are synthetic, not Unity acceptance. New Russian report
preserves the original topic name and contains a short first-person speech,
quantitative findings/limitations and offline demo steps. All four final PDF pages
were rendered and visually checked; editable Markdown and self-contained printable
HTML are included. No unverified new DOCX is delivered. This closes preparation
for a progress meeting, not M9/M10 scientific/live/character acceptance. No person
camera was opened, no training/test-split scoring, network/VPN/driver changes or
previous evidence overwrites. A fresh person-present live measurement, independent
latency/accuracy evidence, final character and held-out acceptance remain open.

MeTRAbs performance continuation (2026-10-18): added opt-in `fp16_conv` crop-weight
storage, leaving BatchNorm parameters/statistics FP32 and the existing geometry
path unchanged. Default stays autocast. A calibration-preserving UI speed button
creates a separate live JSON, never starts capture or overwrites the user's config.
Sequential clean ABBA motion blocks (290 frames each, 30 warmup) measure aggregate
26.2749 -> 27.9948 valid FPS (+6.55%); p95 A 46.63/47.79 vs B 43.48/45.37 ms.
The 70-frame still-video probe is 24.50 -> 29.27 FPS; do not generalize its larger
gain to motion/live. Both completed artifacts verify SHA and exact body/root/
confidence/validity/presence/detection identity, 360 unique input frames total.
The production fast-config video path also processed 279/279 measured frames after
warmup at 27.92 FPS / p95 44.12 ms, a different unpaired protocol, not live evidence.
Added source-age/backend/recording-processing/bookkeeping/packet-send timing in
camera progress/reports and stage medians in the UI. Missing sends stay null;
stage percentiles must not be summed. All 955 Python tests and Ruff pass; actual
Tk speed-button/settings/Stop/restart/UDP smoke passes without opening a camera.
No predictor retraining, held-out scoring, baseline default promotion, CUDA/driver
updates, VPN/network changes or old study overwrites. Real person-present fast
profile / post-render-cap measurement remains required; >=25 live FPS / <=50-ms
capture-to-send p95 is not declared passed. See docs/metrabs_performance.md.

Desktop usability continuation (2026-10-18): UI preferences are validated and
atomically saved locally, restored without starting a session/camera, and never
persist raw-recording consent. Malformed/unsupported preference files are kept
untouched; save failures do not interrupt a run. During replay and camera sessions,
progress now carries fresh session/port/count-matched Unity evidence. UI displays
received/applied/render/error status immediately and expires stale confirmation
using the snapshot's actual age, even if producer progress stops. Final run reports
remain historical evidence. The native Python replay test drains stdout/stderr
asynchronously to avoid Windows pipe-buffer deadlocks and asserts confirmed live
progress before replay completion. All 925 Python tests and Ruff pass; all four
native Unity tests pass (outputs/unity_native_20261018_221346_34af163f).
Actual Tk settings roundtrip/Stop/restart/UDP smoke passes in
outputs/ui_qa_20261018; synthetic receiver-display checks are labelled as such.
No real camera, model training, held-out scoring, live defaults, network/VPN or
old research artifacts were changed. Live timing/character/research acceptance
remain open; the new feedback is diagnostics, not a latency improvement claim.

Presentation preparation (2026-10-13): four explicit offline desktop presets,
read-only preflight/CPU model smoke, and a reproducible results handout from nine
verified batches (252 saved experiments / 2,025 artifacts). Selection reports are
reproduced from pinned manifests before publishing the handout; nothing is trained,
test-scored or promoted to a live default. UI columns now scroll independently,
preserving the journal and controls at smaller window sizes. Actual Tk buttons,
Stop/restart and localhost decoding pass; all four native Unity tests pass with
CMU recording replay. 876 Python tests and Ruff pass. Russian DOCX/text show guide
covers a 7–10 minute offline/live presentation, evidence boundaries and failures.
DOCX structural QA passes; visual render cannot run because LibreOffice is absent.
No camera, VPN/interface/firewall settings or old experiment files were changed.
Pending research/live/character acceptance gates below remain open; this prepares
the prototype for a demonstration, not final scientific/whole-project acceptance.

Latest continuation (2026-10-11): completed 360 new validation cells across four
predeclared panels / 2,892 verified batch artifacts. Added fixed-magnitude outliers
and delivered zero-confidence tracking-loss frames, distinct from missing arrivals.
The 126-cell SPEC E8 non-Gaussian panel improves the tested outlier means, but both
methods fail the unchanged full-panel .95 coverage gate for severe missing/gap
profiles. No winner is declared; no held-out data were scored or thresholds relaxed.
Rare-dropout correction remains selected in its narrower 54-cell ablation; Gaussian
gains remain descriptive/repeated-validation. Real weights verify exact corrected
vs original Gaussian prefix equality (640 anchors) and 400 UDP encode/decode frames.
The saved M1 MeTRAbs 60-frame smoke is valid but entirely Kalman-only: the neural
consistency guard rejects contribution. This is not real-camera accuracy evidence.
862 Python tests and Ruff pass. Source ZIP snapshots retain reproducible package
digests without data/models/envs. No camera/Unity/network settings were changed.
Next: independent/larger validation and latency-focused work, then freeze final
held-out settings; live >=25 FPS / p95<=50 ms and real-character acceptance remain.
See docs/adaptive_confirmation.md for full evidence, limitations and commands.

M8 confirmation and M9 class foundation (2026-10-11): completed 180 predeclared
validation cells / 1,446 verified batch artifacts. Frozen noisy h100/tau25/cap25
improves all Gaussian macro error/jitter means across three noise seeds (worst
ratio .956031), but broader stress retains Kalman-CV: rare per-joint dropout
increases adaptive error/jitter by 17.04%/50.73%. Missing-arrival gaps, outliers and
20-Hz source improve. No latency superiority: common-subset clean/sigma10 deltas
are -0.25/+0.77 ms; high-noise cells lack adequate shared windows.
Implemented opt-in partial baseline recovery, retaining independent usable joint
Kalman states while clearing neural support; full loss/gaps stay Raw. A new
54-cell correction panel completed / 435 verified artifacts, selecting partial
recovery: dropout error/jitter 47.24/89.64 vs Kalman's 48.16/91.57, while clean
metrics are exactly unchanged. All old configs remain unchanged; the new local
experimental config is adaptive_kalman_recovery_pilot.json, not a live default.
Missing-joint confidence now zero per SPEC; an archived dropout replay reproduces
all old pose/jitter/latency/output-decision metrics exactly. Train-only motion
classification freezes 160 GT windows at acceleration quantiles 4.775371/12.540987;
validation labels and class-wise Gaussian diagnostics are generated without test
scoring or tuning. These are repeated-validation, not participant-level results.
The final regression is now 862 Python tests / Ruff; see the latest continuation
and docs/adaptive_confirmation.md for the completed SPEC E8 panel. Independent/
larger data/latency studies remain before freezing the held-out protocol.

M8 baseline-mixture continuation (2026-10-11): implemented optional Kalman-CV
fallback with same-target neural strength adaptation, capped contribution and
causal paired neural-vs-baseline observation errors. Original Raw-based variants
remain unchanged. Root comes from Kalman; tracking loss/gaps still clear support
and return Raw. UDP/reporting distinguish Kalman-only future output from Raw.
Completed a predeclared 72-case / 577-artifact validation pilot with existing clean/noisy h100
checkpoints, history tau10/tau25 and caps .10/.25. No retraining, test scoring or
live promotion. Only h100 has a noisy checkpoint. See docs/adaptive_compensation.md.
All 757 Python tests and Ruff pass, including exact continuous-valid Kalman
fallback and same-target mixed output/paired-errors/identity/replay checks.
The pilot selects noisy h100/tau25/cap25, worst error/jitter ratio 0.955192 against
Kalman's 1.0. All four Gaussian-profile macro error/jitter means improve; sigma50
is 69.28 mm / 157.50 m_s2 versus 77.66 / 176.20. This is single-seed descriptive
validation, not held-out or latency superiority: eligibility masks differ. The M7
fixed comparator remains Kalman-CV. Next freeze/confirm the adaptive config on a
predeclared multi-seed panel and broader corruption/data before held-out inference.

M8 stability continuation (2026-10-10): added optional bounded causal trailing-window
quadratic dynamics fitting and source-time forecast-weight transitions. Dynamics
filtering does not filter matured observation-error references. Transitions mix
only current-anchor forecasts, log requested decisions/weights and preserve the
largest contributing nominal UDP horizon. Tracking loss/gaps clear weights at
once; soft gates can release weights gradually. Old configs have both options off.
Completed a new 72-run / 577-artifact validation-only mechanism ablation, with the same checkpoint
bank, gates, profiles and noise seed; no test scoring/training/live-default change.
The combined window/history/transition candidate passes the clean descriptive
gate (error -8.68%, jitter +9.69% against Kalman-CV), but worst noisy ratio 1.82017
prevents selection. Sigma10/25 forecasts now run, yet noisy jitter remains worse;
sigma50 still falls back to Raw. Kalman-CV remains selected. Next investigate a
filtered strong-baseline fallback/mixture and noise-aware bank/reliability, with a
new predeclared panel before held-out inference. See docs/adaptive_compensation.md.
Regression: 735 Python tests and Ruff pass; the original v1 replay reproduces its
archived metrics exactly. Live-camera and held-out acceptance are unchanged.

M8 update (2026-10-10): implemented the pinned four-checkpoint bank and version-1
causal threshold policy, with horizon-only and horizon/blending variants. All four
models consume each observation; probes remain separate from displayed output.
Raw fallback preserves target-zero semantics; partial blends explicitly declare
their current/future mixture and nominal forecast horizon. Per-frame decisions and
features persist in processed.npz; the shared factory/experiment/UDP paths support
adaptive custom configs without changing production defaults or Unity protocol.
Inference failures require reset before reuse. Ran and verified a prospective
72-run / 577-artifact validation-only threshold pilot (two error limits, two modes,
four Gaussian profiles, one seed, three clips). Kalman-CV remains selected: adaptive
variants violate the clean jitter budget, while the raw acceleration gate disables
all noisy forecasts. No adaptive config is accepted/promoted. Next make dynamics
features noise-tolerant and assess smoother horizon/alpha transitions in a new
predeclared panel. Calibration, broader/multi-seed evaluation, live performance and
final held-out conclusions remain open. All 694 Python tests and Ruff pass; the
original 252-run selection is unchanged. See docs/adaptive_compensation.md.

Current update (2026-10-07): a real person-bearing, raw webcam -> Unity run is now
available: outputs/ui_20261005_140958_4ed33ebe/run_report.json. It completed 600.01 s,
with 10,809 valid poses out of 11,105 processed observations (97.33%); Unity received
and applied all 10,809, with zero rejected packets/receiver/rig errors. Camera
accounting matches: 18,000 captured = 11,105 processed + 6,894 latest-slot overwrites
+ 1 pending. Host working-set and reserved-CUDA growth gates pass; Unity measured
600.26 s, with working-set growth 2.45 MiB. This closes the missing-person-feed and
sustained-delivery/stability checks, not all live acceptance: 18.01 valid FPS and
98.43-ms capture-to-send p95 miss SPEC's >=25 FPS and <=50-ms budgets. No synchronized
ground-truth accuracy or capture-to-screen latency is inferred; raw NPZ was off.

Fixed the Windows UI command-pipe startup deadlock and added detailed model/camera/
warm-up progress and safe startup cancellation. The Unity demo now scopes a 60-FPS
target/VSync-off setting to Play, restores owned previous settings and exports the
configured target separately from measured render FPS. Four native tests pass;
outputs/unity_native_20261005_143910_da62a79d. No person-bearing post-cap report
exists yet: speed improvement remains unverified, not a prerequisite for offline work.

M7 selection work: implemented a complete-panel, validation-only robust comparator
selector with input/mask/provenance checks, explicit exclusions and a baseline tie
break. Clean macro error/jitter may grow at most 10%; corrupted-profile worst error/
jitter ratios determine the descriptive selection. Differing effective-latency
eligibility masks never imply a latency victory. Prepared the pinned prospective
100-ms continuation at outputs/cmu_robust_selection_20261007: 7 existing candidates,
Gaussian sigma 0/10/25/50 mm, 3 noise seeds, 3 unchanged validation clips (252 cells).
No retraining, test scoring or live promotion. All 252 runs / 2,019 batch artifacts
completed and verified; Kalman-CV is selected, with four operating-point comparator
files frozen. The eligible noise-MLP tau10 candidate has a worst ratio 1.22375,
while tau25 violates the clean-error budget; this is not a latency/test victory.
See docs/robust_selection.md. Initial M8 feature tracking is now implemented:
bounded causal velocity/acceleration/confidence and prediction error only after
its target observations arrive, with loss/gap/clock handling. Policy, checkpoint
bank, live integration and final held-out studies remain; see
docs/adaptive_compensation.md. All 632 Python tests and Ruff pass; re-running the
artifact selector reproduced selection.json byte-for-byte. Archived study/model
inputs and production configs are unchanged.

Refactoring (2026-10-05): moved camera orchestration into the installed runtime
package and kept the script as a compatibility launcher; removed desktop runpy
coupling. Centralized lightweight processor-config validation/factory registration,
byte hashing and host-memory measurement. Hash consumers no longer import private
dataset/evaluation helpers. Separated Tk event transitions from queue/timer polling
and CLI execution from common JSON/error reporting. Preserved algorithms, NPZ/UDP,
experiment manifests and frozen data/settings. Added boundary/transition/compatibility
regressions: 537 Python tests and Ruff pass. C# smoke and all three actual Unity
tests pass; console Tk QA and a fresh base-wheel installation pass. Existing CMU
batch verifies unchanged (6 runs, 49 files). Native results:
outputs/unity_native_20261005_115627_589ce6bc; base installation:
outputs/base_install_20261005_115637_fc9db1a2; UI QA: outputs/ui_refactor_20261005_v1.
See docs/refactoring.md. Person-present live acceptance remains open; no camera or
GPU training was started by this maintenance pass.

Run evidence and base installation (2026-10-05): desktop sessions now save a combined
run_report.json alongside the unchanged producer report. Unity atomically exports
local, session-scoped received/applied counters, resident memory and built-in Game
camera render callbacks; Python refuses stale/malformed/other-session snapshots.
Batch/headless runs never claim render FPS or live acceptance. Camera benchmarks
sample host/CUDA memory after warm-up, including start/end and periods without frames;
short runs cannot pass the ten-minute memory gate. Fixed Mono returning zero resident
memory on Windows via GetProcessMemoryInfo. Native CMU worker replay verifies the
combined report, not merely local sends. A fresh non-editable base wheel installation
passes without torch/cv2/dev packages: five baselines, NPZ, actual UDP, lazy GUI imports,
CLI and pip check. This is a clean venv on the same OS, not a new GPU/driver setup.
516 Python regressions, C# smoke, actual-API compilation and three native Unity tests
pass. Native results: outputs/unity_native_20261005_023208_09dce59e; combined report:
outputs/unity_worker_da5afa68c2d845cb8535e46ab0df9032/run_report.json.
Base installation evidence: outputs/base_install_20261005_023037_4e4d1927.
Real person-camera -> Unity ten-minute acceptance remains open; no new model winner,
test-set scoring or VPN/network-setting changes. See docs/desktop_ui.md.

Desktop controls (2026-10-05): added native Tk panel and double-click launcher with
diagnostic/NPZ/camera sources, explicit baseline/custom-checkpoint selection, dynamic
forms, pinned-Editor metadata preflight, producer progress/metrics, bounded logs and
isolated session artifacts. Worker runs outside the UI thread; Stop is cooperative
and camera raw-prefix saving/cleanup is tested with simulated hardware. Actual Tk
buttons + real producer/UDP pass under console Python and pythonw; NPZ replay is
read-only. Window-only QA: outputs/ui_smoke_20261005_v7. No web server, VPN settings,
automatic research-model promotion or inferred Unity receipt. Real camera/Unity
long-run acceptance remains the next external gate. 489 Python tests pass; C# smoke
passes. See docs/desktop_ui.md.

Native Unity validation (2026-10-04, after user activation): the pinned F:/Unity/
Editor/Unity.exe now imports and runs the project. Fixed missing built-in Animation,
IMGUI, Physics and ImageConversion modules; aligned Test Framework to bundled 1.6.0.
All three native tests pass: AvatarBuilder/rotation compatibility, actual Play Mode
UDP driving across eight phases with loss/expiry/order checks, and actual Python
CMU recording replay (119 packets accepted, 114 observed rig frames). The Editor-only
suite explicitly enters Play Mode; results are in outputs/unity_native_20261004_v7.
Generated a saved scene with side-colored materials and visually inspected eight
native offline renders on RTX 3070 Ti: outputs/unity_scene_20261004_v3. Fixed fresh
untitled-scene generation and preview cleanup; existing user scenes/assets are not
overwritten. 450 Python tests, C# smoke, real-API compilation and Ruff pass. Initial
license/compile/discovery attempts remain as historical logs, not successful gates.
Next: person-present webcam -> Unity sustained acceptance and frame-rate measurements.
M3 remains open for that live gate and head/hand/foot orientation limitations.

Unity vertical-slice implementation (2026-10-04): added parent-first position-based
Humanoid driving with calibrated rest offsets, full pelvis/chest frames, inherited
rest twist, validity/degeneracy guards and ordered-frame rejection. Added a primitive
Humanoid/scene generator, telemetry, native EditMode test sources and real-assembly
API compile helper. Added localhost diagnostic/recording replay with original-time
processing and display-only clock rebasing, plus Python/C# regressions. The native
validation update above supersedes the initial activation blocker. No downgrade or
VPN/network setting changes. M3 implementation advanced, live acceptance remains
open. See docs/unity_retargeting.md for commands and gate checklist. Next: validate
with a usable person-bearing camera. Adaptive compensation/final studies remain ahead.

Noise/prefilter combination and velocity-residual pilot (2026-10-03): completed
two integrity-verified 24-run shared evaluations. Noise-trained MLP + tau 10/25 ms
has noisy error 29.58/30.41 mm and jitter 45.10/24.84 m/s², with clean error
25.40/28.31 mm. Smoothing trades lag/clean accuracy for robustness; no overall
winner or live default change. Added optional constant-velocity output baseline
with explicit version-3 checkpoint semantics and backward-compatible hold loading.
Matched clean/noisy eight-epoch training selected epoch zero in both cases:
12.68-mm clean validation target error, no learned improvement. Unfiltered analytic
baseline has display error 14.41 mm clean but 107.57 mm noisy and high jitter;
filtering does not establish a robust latency victory. 438 tests pass; Ruff clean.
See docs/cmu_noise_prefilter.md and docs/cmu_velocity_residual.md for coverage,
protocols and artifacts. Test data and live defaults remain unchanged.
Next: close Unity Humanoid retargeting/scene integration and usable-camera
acceptance; before further architecture tuning, predeclare a robust validation
selection objective and broaden corruption/data coverage. M7 is not accepted yet.

Noise-aware training pilot (2026-10-03): implemented train-only clip-consistent
Gaussian observation corruption before interpolation/windowing, with unchanged
clean future targets and version-3 provenance. Trained the 100-ms relative-last
model (seed 42, eight epochs; epoch 5 selected on unchanged clean validation).
Shared 100-ms-delay evaluation: noisy error 50.09 -> 32.37 mm, jitter 174.08 ->
72.44 m/s²; clean error worsens 20.27 -> 24.09 mm. Kalman extrapolation remains
slightly more accurate and substantially smoother on this noisy pilot. No test/live
promotion. 426 tests pass. Training: outputs/cmu_noise_training_20261003_v2;
evaluation: outputs/cmu_noise_eval_20261003. See docs/cmu_noise_training.md.
Next: controlled prefilter combination, broader noise/seeds and velocity-residual
comparison; preserve clean supervision and report latency coverage.

Prefilter robustness pilot (2026-10-01): added optional causal EMA on learned
history only (default off), with raw fallback/root/timestamps preserved and reset
tests. Completed tau 0/10/25/50 ms x clean/noisy x three validation clips (24 runs).
At tau 50, noisy learned error/jitter fall from 50.09 mm/174.08 m/s² to
32.74 mm/36.42 m/s², but clean error rises from 20.27 to 28.28 mm and clean lag
increases. Kalman extrapolation remains slightly more accurate on this noisy pilot.
No reliable noisy latency victory or live promotion. 424 tests pass. Artifacts:
outputs/cmu_prefilter_20261001; docs/cmu_prefilter.md. Next: temporally consistent
training-only noise augmentation with clean supervision and matched evaluation.

Delay/noise pilot (2026-10-01): completed and integrity-verified 36 shared-runner
validation experiments (three clips, six methods, noise sigma 0/10 mm, delay 100 ms).
Learned display MPJPE is 20.27 mm clean versus Raw 29.28, but noisy learned is 50.09
versus Raw 36.08 and Kalman extrapolation 31.90. Learned jitter rises strongly.
Noisy learned latency coverage is about 1%, so its small eligible latency estimate
is not evidence of success. No overall winner/test/live promotion. 416 tests pass.
Artifacts: outputs/cmu_delay_noise_20261001; docs/cmu_delay_noise.md.
Next: noise-robust training or causal prefiltering with clean-reference supervision,
then controlled velocity-residual experiments; preserve delay/coverage reporting.

Seed/horizon stability (2026-09-27): completed the predeclared 4-horizon x 3-seed
relative-last/global-window study on the unchanged expanded-CMU validation data.
MLP mean errors at 33/66/100/150 ms: 5.62/12.37/21.07/36.12 mm. Hold is beaten in
12/12 runs, constant velocity in 0/12. Seed ranges are descriptive, not confidence
intervals over people. All models reloaded and matched selected-epoch scores;
test and live configurations remain untouched. 413 tests pass. Artifacts:
outputs/cmu_stability_20260927; documentation: docs/cmu_seed_stability.md.
Next: shared delay/noise evaluation and a controlled velocity-residual model test.

Controlled model experiment (2026-09-27): implemented explicit relative-last input
features and optional seeded global-window mixing, preserving legacy model loading.
A fixed 2x2 experiment on identical expanded-CMU 100-ms train/validation windows
found MPJPE 30.52 mm for the original inputs (either order), 27.05 mm for relative
inputs/fixed order, and 21.00 mm for relative inputs/global mixing. Hold is 30.52 mm,
last-two-sample constant velocity 12.68 mm. First learned improvement over hold,
but no improvement over the stronger baseline and no test/live promotion. All
410 tests pass. Artifacts: outputs/cmu_ablation_20260927. See docs/cmu_model_ablation.md.
Next: repeat seeds/horizons, velocity-residual comparison and shared delay/noise
experiments; current single-seed validation evidence is not M7 completion.

Training expansion (2026-09-24): added nine clips from the existing training IDs
only, retaining all validation/test selections. The corpus now contains 22 clips
and 40,326 frames, including 19,903 train frames (formerly 3,935). Four eight-epoch
native-history runs again selected epoch zero. More diverse data alone did not
improve this MLP on this validation set. No additional participants, final balanced
cohort, test scoring or live promotion is claimed. New comparison tooling verifies
exact numeric validation-window equality and reports per-sequence/macro errors.
Artifacts: outputs/cmu_expanded_20260924. Regression suite: 406 tests pass.
Next: investigate motion-relative/velocity-based model inputs and training order
as controlled validation experiments, then compare under noise and delay.

Native CMU history update (2026-09-23): training exports now optionally reconstruct
20-ms histories from original 120-Hz observations using the same causal sampler
as the live processor. Manifest v2 pins sampling/gap settings; v1 remains supported.
All four horizons were retrained on the same 13 clips, without test scoring or live
promotion. They again selected epoch zero. At 100 ms, validation target MPJPE was
30.52 mm for hold/selected MLP and 12.68 mm for last-two-sample constant velocity.
This is not a latency/jitter comparison or a matched ablation against the old
20-Hz pilot. Artifact: outputs/cmu_native_history_20260923. Next: balanced source
data expansion and shared delay/noise comparisons, not more claims from overlapping
windows of the same small sample.

CMU pilot update (2026-09-23): a 13-clip, five-source-ID corpus (24,358 original
frames) is downloaded, checksum pinned and converted. Dataset index/window-store
validation supports a separate CMU policy, with cross-split duplicate skeleton
checks. A 20-Hz decimated exploratory run trained four horizons; all selected epoch
zero (hold last pose), while simple constant velocity had lower validation target
error. No test tuning or live model promotion. Matrix/quaternion FK agrees on the
three original smoke clips; visual/anatomical and actor-alias validation remain
open. Current artifact: outputs/cmu_pilot_20260923_v2. Full suite: 397 tests pass.
Next: balanced training expansion, time-grid handling and shared latency/noise
evaluation. See docs/cmu.md for caveats and reproduction commands.

## 1. Objective

Build a reproducible research platform and a stable Unity demo for evaluating
predictive latency compensation of a full-body avatar.

Implementation priority:

```text
correctness
-> causal processing
-> reproducibility
-> measurable baselines
-> learned prediction
-> adaptive method
-> demo polish
```

The implementation must preserve one shared processing core for live camera,
recorded replay, and offline experiments.

Current progress (updated 2026-10-07; M1 fallback measured 2026-09-10): M0's package,
CLI, config loader, and
test infrastructure are implemented; clean-machine installation remains to be
verified. M1's software feasibility work is implemented. Profiling rejected the
13 FPS EfficientNetV2-L configuration and selected the separately published
EfficientNetV2-S 256 px checkpoint, YOLOv8n, and causal pose-guided box tracking.
A power-jump recording produced valid poses on every measured frame at about
22-24 FPS; a simpler 640-pixel stream reached 25.33 FPS on an earlier profile.
The final ten-frame tracking profile measured 23.33 FPS over ten minutes. This is the
documented performance fallback permitted by the M1 gate, not a claim that all
motion reaches 25 FPS. Sixty canonical raw frames were saved and round-trip
checked. The earlier static 640x480 zero-pose probe is superseded by a person-bearing
600-second webcam -> Unity run: 10,809 valid poses, all received/applied, bounded
camera accounting and passing memory-growth gates. Live stability/delivery are
measured; throughput (18.01 FPS) and capture-to-send p95 (98.43 ms) still miss SPEC.
The 60-FPS render target is implemented but not yet remeasured with a person.
M2's pose, MeTRAbs/H36M mapping, immutable metadata, axis conversion,
and storage serialization are implemented. The September 21 update adds immutable
PoseSequence clock/order validation and a Human3.6M 32-joint camera-space array
adapter with rational event timestamps. Official Human3.6M CDF decoding and final
held-out/real-camera validation remain; CMU ASF/AMC import and validation pilots are
implemented. M3 has UDP v2, Python loopback, C# ordered receiver/root/bone drivers,
a saved diagnostic Humanoid scene and paced replay. Runtime stub regressions are
supplemented by four actual native tests and eight inspected native renders.
Real Python CMU replay and ten-minute person-camera delivery reach the native rig.
Live timing budgets and orientation limitations remain open. Regression checks
now pass 632 Python tests; the C# runtime smoke suite also passes.
Live benchmark orchestration is tested
with simulated camera/inference and actual localhost UDP, including failure
cleanup and valid-only transmission. The actual camera thread is tested with
controlled devices, including blocked reads; shutdown failures are explicit.
Pinned model files are verified before deserialization. An offline real-model
still-image inference run passed.
UDP now has loopback-only reception, observable socket failures, strict masks
and quaternions, baseline IDs 5-7, and full-field cross-language fixtures. The
overall project and the remaining hardware gates must not be marked complete yet.

M4 recording/replay and M5 causal baselines/delay are implemented. M6 includes
prepared dataset import/indexes, core metrics, single/batch runs, integrity checks
and split-safe training windows, paired comparisons and integrity-verified CMU
validation studies; final statistical/held-out coverage remains open. M7 has checked
data loading, train-only normalization, four-horizon/three-seed CMU MLP checkpoints,
noise/prefilter and velocity-residual pilots, plus causal live/offline processing
with explicit fallback. No robust learned winner, Human3.6M training or learned
real-camera acceptance is claimed. M8 and final research experiments remain ahead.

User-requested replacement: MeTRAbs now supplies an estimated monocular metric
root directly on every successful frame. The former floor/contact and body-length
estimators, their configs, commands, tests and documentation are removed. Unity's
container driver supports recentering and holds position on loss. Python and C#
tests pass; ground-truth trajectory validation and native/real-avatar scene
acceptance remain. A code-generated diagnostic mannequin/driver is now available
for the integration gate. See `docs/metrabs.md` and `docs/unity_retargeting.md`.

---

## 2. Fixed Decisions

- Python package layout: `src/poseahead`.
- Live pose backend: pinned MeTRAbs EfficientNetV2-S 256 px PyTorch with
  YOLOv8n 320 px, redetection every ten processed frames, and causal
  pose-guided crop updates between detections.
- Canonical pose: pelvis-relative 3D joint positions in meters.
- Canonical axes: right-handed camera coordinates, `+X` right, `+Y` down,
  `+Z` away from camera.
- Unity conversion: `(x, y, z) -> (x, -y, z)` plus scale/origin calibration.
- Initial transport: versioned little-endian binary UDP, one frame per datagram,
  no more than 1200 bytes.
- Learned predictor: four separate checkpoints for `33`, `66`, `100`, and
  `150 ms`.
- Current experimental dataset: CMU Motion Capture; Human3.6M access remains pending.
- Unity: Unity 6.3 LTS `6000.3.23f1` on Windows x64.
- Reference machine: Ryzen 7 5700X, RTX 3070 Ti 8 GB, 16 GB RAM.
- Minimum live performance: 25 valid pose frames/s; target: 30 FPS.

---

## 3. Target Repository Layout

```text
PoseAhead/
|-- src/poseahead/
|   |-- core/
|   |-- perception/
|   |-- processing/
|   |-- prediction/
|   |-- transport/
|   |-- recording/
|   |-- evaluation/
|   |-- datasets/
|   |-- runtime/
|   `-- cli.py
|-- unity/PoseAheadAvatar/
|-- configs/
|   |-- live/
|   |-- training/
|   `-- experiments/
|-- tests/
|   |-- unit/
|   `-- integration/
|-- data/
|-- artifacts/
|-- outputs/
|-- docs/
`-- pyproject.toml
```

`data/`, downloaded model weights, generated artifacts, and experiment outputs
must not be committed. Small deterministic test fixtures may be committed under
`tests/fixtures/`.

---

## 4. Milestones

## M0. Repository Foundation

### Work

- create the target package and test layout;
- configure `pyproject.toml`, formatting, linting, and pytest;
- add CLI entry point and configuration loader;
- add `.gitignore` rules for environments, datasets, checkpoints, Unity caches,
  and generated outputs;
- create initial architecture and setup documentation;
- record dependency versions and GPU/runtime information.

### Deliverables

- importable `poseahead` package;
- `poseahead --help` command;
- empty but executable test suite;
- documented environment setup.

### Gate

Do not start feature work until a clean environment can install the package and
run the test suite.

---

## M1. MeTRAbs Feasibility Spike

This is the first technical risk and must be tested before building the rest of
the live pipeline around it.

### Work

- pin a compatible Python, PyTorch, CUDA, Ultralytics and MeTRAbs stack;
- pin MeTRAbs source, pose checkpoint and detector checkpoint by checksum;
- run inference on a still image, recorded video, and webcam;
- inspect 2D/3D output shape, joint convention, units, confidence, and root
  behavior;
- verify that inference uses only the current/past frames;
- benchmark warm-up, median, p95, FPS, VRAM, and dropped frames;
- save a short raw sample for deterministic development.

### Acceptance

- continuous webcam inference works for 10 minutes without a crash or memory
  growth;
- at least 25 valid 3D pose frames/s on the reference machine, or a documented
  fallback decision is made;
- output can be mapped to the required 17-joint skeleton;
- no hidden future-frame dependency exists.

### Current gate disposition (2026-10-07)

- model/source/detector pins, still/video execution, H36M-17 mapping, current/past
  causality, stage profiling, motion sample, and raw PoseFrame sample: complete;
- throughput: documented fallback accepted (25.33 FPS on the simple 640-pixel
  stream; about 22-24 FPS on the 960x540 jump sample);
- ten-minute stability: exercised with looped recorded motion and kept distinct
  from webcam acceptance;
- person-bearing webcam for ten minutes: completed without crashes, matched
  sent/received/applied counters and passing RAM/VRAM-growth gates;
- live timing remains open: 18.01 valid FPS and capture-to-send p95 98.43 ms;
  the earlier recorded-motion fallback does not establish the live >=25 FPS goal;
- new 60-FPS Unity target needs a matched post-cap camera measurement;
- Human3.6M ground-truth accuracy belongs to the later evaluation/data milestones
  and is not inferred from webcam or public-video observations.

### Fallback order

1. profile detector and pose stages and remove avoidable overhead;
2. evaluate a smaller compatible MeTRAbs checkpoint if one is reproducibly available;
3. evaluate an audited ONNX/TensorRT export without changing model semantics;
4. keep recorded/mock PoseFrames available so downstream work is not blocked.

---

## M2. Canonical Pose Core

### Work

- implement `PoseFrame`, joint enumeration, hierarchy, validity, and confidence;
- define timestamp and clock-domain semantics;
- implement canonical coordinate and unit conversion;
- implement dataset/backend-to-canonical joint mappings;
- implement versioned serialization independent of UDP;
- document rest pose, root convention, missing rotations, and invalid joints.

### Tests

- joint order and parent hierarchy;
- shape, dtype, and finite-value validation;
- serialization round trip;
- coordinate conversion round trip;
- root-relative invariants;
- timestamp monotonicity checks.

### Gate

No backend-specific structure may cross the canonical-pose boundary.

### September 21 implementation update

PoseSequence rejects mixed declared clocks, duplicate/backwards frame IDs,
non-increasing event timestamps, and backwards production timestamps. Gaps and
variable-rate sources are supported. Human3.6M prepared camera-space arrays map
to the same canonical representation with labelled ground truth and synthesized
chest. Rational frame timestamps avoid cumulative rounding drift. The existing
60-frame MeTRAbs sample also passes sequence validation. Prepared numeric NPY/JSON
import now supports explicit world-to-camera extrinsics and fixed subject labels.
Official Human3.6M file decoding and licensed-data validation remain dataset work.

---

## M3. Python-to-Unity Vertical Slice

Status (2026-10-04): native Humanoid construction, ordered UDP playback, loss/expiry,
recorded CMU movement and offline scene rendering are verified. Saved colored scene:
`Assets/PoseAhead/Scenes/PoseAheadDiagnostic 1.unity`. Live person-camera acceptance,
sustained render FPS and imported-character checks remain open. Head/hand/foot local
orientations are not measured from endpoint positions; no gaze/finger tracking claim.

### Work

- define and document binary UDP packet version 2 with ordered sender sessions;
- implement Python encoder/sender and C# decoder/receiver;
- reject malformed, stale, duplicate, and out-of-order frames;
- use a latest-frame queue with capacity one;
- create the Unity 6.3 LTS project and pin its exact patch version;
- map canonical joints to a Humanoid avatar;
- derive bone rotations from positions and calibrated T-pose;
- display FPS, frame ID, active method, and processing timing.

### Tests

- Python packet round trip;
- golden packet decoded identically in Python and C#;
- malformed/truncated packet rejection;
- packet-loss and out-of-order simulation;
- coordinate conversion test with known poses.

### Acceptance

- prerecorded canonical poses animate the Unity avatar;
- live MeTRAbs poses reach Unity continuously;
- major arm, leg, torso, and head motion has the correct direction;
- queues remain bounded and stale frames are discarded.

---

## M4. Recording and Replay

Implementation update (September 21): versioned pickle-free NPZ recording,
no-overwrite atomic saving, raw webcam capture with missing observations, paced
and accelerated replay, and separate config-driven derived runs are implemented.
Live and replay use identical processor classes. Capture currently buffers a
bounded session in memory and saves on successful completion, not crash recovery.

### Work

- define a versioned recording manifest and efficient array storage;
- store immutable observations, timestamps, confidence, validity, backend
  metadata, and processing timings;
- store derived processor outputs separately by run ID;
- implement replay using original event time and accelerated deterministic mode;
- make live and replay call the same processing interface.

### Tests

- recording round trip;
- replay order and timestamp preservation;
- deterministic accelerated replay;
- live/replay processor consistency on the same input.

### Acceptance

- a camera session can be replayed without camera or Unity;
- repeated offline processing produces identical output for the same config and
  seed.

---

## M5. Causal Processing and Artificial Latency

Implementation update (September 21): all five baselines, checked-in JSON configs,
timestamp-based delay and shared causal simulation are implemented. Tests cover
reset, gaps/loss, variable-time EMA, analytic motion, Kalman covariance, delay
boundaries, future-prefix independence and live/replay equality. UDP carries the
selected method and horizon. Config parameters are initial, not validated optima.

### Work

- implement common resettable `PoseProcessor` interface;
- implement Raw, EMA, One-Euro, Kalman, and Kalman constant-velocity
  extrapolation;
- implement timestamp-based artificial observation delay;
- handle irregular frame intervals, invalid joints, and tracking gaps;
- keep method configuration out of source-code constants.

### Tests

- processor reset;
- constant and linear-motion sanity cases;
- variable-FPS behavior;
- missing-joint behavior;
- no future-frame leakage;
- exact observation cutoff at `t_eval - artificial_latency`.

### Gate

Do not start neural prediction until every baseline runs through the same offline
interface and passes causality tests.

---

## M6. Dataset and Evaluation Core

Implementation update (September 21): prepared Human3.6M import and subject labels,
target-aligned canonical MPJPE and independent root error with coverage are present.
Exact matching or explicitly selected gap-limited linear reference interpolation
is supported. Temporal APIs now implement acceleration-residual jitter and
window-level velocity-correlation latency with explicit coverage and rejection
reasons. A simulation-to-display-time bridge prevents source-time relabelling.
Synthetic tests cover known positive/negative/sub-frame delays, static/periodic
ambiguity, irregular-time acceleration and delayed-replay integration.
September 22 update: single-sequence `run-experiment` now drives all five baselines,
delay, seeded independent joint noise/dropout, display/target error, jitter,
fixed-window latency and processor runtime. Resolved configs pin input hashes,
environment/source hashes and artifact manifests; repeated runs and corruption
checks are tested. Synthetic fixtures are available from a CLI command.
Batch suites now preflight all inputs and fixed subject splits, reject duplicate
sequence/condition entries, execute sequentially and produce descriptive macro
summaries with per-run coverage. Suite/child artifacts have integrity verification.
Prepared dataset indexes pin hashes and clip-level subject/split inventories.
Training-window exports now select a verified split before generating causal
histories within each clip. Four target horizons use explicitly bounded reference
interpolation, with per-reason rejection counts and hashed NPZ shards. This is a
root-relative ground-truth data contract, not trained predictor checkpoints.
Paired comparisons now enforce matched input/time/mask coverage, report deterministic
sequence-bootstrap CIs, require validation-frozen comparators on test, and expose
SPEC decision flags plus descriptive Pareto fronts. Subject-level dependence and
multiple-comparison uncertainty are not resolved by these pointwise intervals.
Calibrated corruption profiles, subject-level sensitivity analysis,
archived source snapshots and real-data experiments
are still pending; the synthetic smoke result is not a Human3.6M conclusion.

### Work

- implement Human3.6M loader and canonical mapping;
- create immutable split manifests:
  `train=S1,S5,S6,S7`, `validation=S8`, `test=S9,S11`;
- ensure window generation happens after subject split;
- implement MPJPE, root error, primary jitter, effective latency, and runtime
  metrics;
- implement deterministic noise and tracking-loss profiles;
- implement config-driven experiment runner and resolved-config output;
- store run ID, seed, environment, Git revision, metrics, plots, and logs.

### Tests

- dataset split leakage prevention;
- metric values on analytic synthetic sequences;
- effective-latency sign and magnitude with known delays;
- deterministic noise from seed;
- invalid/static window handling and coverage.

### Acceptance

- all non-learned baselines can be evaluated from one command;
- rerunning a saved resolved config reproduces its metrics within documented
  numerical tolerance.

---

## M7. Fixed Learned Predictors

Initial implementation: checked streaming window loader, train-history-only
normalization, residual MLP, CPU Adam training with validation-only selection and
numeric NPZ checkpoints are available. Synthetic tests verify learning, deterministic
repeat weights, checkpoint integrity and rejection of test data for training/selection.
Retained CPU smoke runs for all four horizons have also trained and reloaded
numeric checkpoints on explicitly synthetic linear-motion fixtures. They use the
same architecture/training settings, are not research checkpoints, and are not
evidence of cross-subject generalization.
CMU pilot checkpoints for four horizons and three model seeds are trained and
reloaded. Learned prediction beats hold but not the stronger clean constant-velocity
baseline; noise/prefilter pilots expose accuracy/jitter tradeoffs. No licensed
Human3.6M training, final validated model family or learned live acceptance is
claimed. The pinned robust validation selection continuation is described above;
M7 is not accepted yet.
Learned checkpoints now run through the shared offline processor/experiment path,
with bounded causal history, exact-grid validation, explicit raw fallback and
checkpoint hashes in resolved configs. Status counts expose warm-up versus actual
predictions. Experimental `linear_past` history reconstruction now supports irregular
arrival/event intervals using only received observations; gaps/invalidity/capacity
limits fall back explicitly. Webcam requires that mode and sends per-frame predicted
or Raw UDP identity/horizon. Hardware-free loopback tests pass, but learned live
performance, realistic motion quality and full camera/Unity acceptance remain pending.

### Work

- implement one compact predictor architecture;
- define fixed input history and normalization from training data only;
- train separate `33/66/100/150 ms` checkpoints;
- save checkpoint, training config, split manifest, seed, and validation metrics;
- compare every checkpoint against Raw and Kalman extrapolation;
- integrate checkpoint selection into live and replay modes.

### Tests

- tensor shape and target-time construction;
- no target leakage in training windows;
- checkpoint/config compatibility;
- deterministic inference;
- unsupported horizon rejection.

### Acceptance

- four reproducible checkpoints exist;
- horizon sweep produces MPJPE, jitter, and effective-latency results;
- inference satisfies the live processing budget.

---

## M8. Adaptive Compensation

First foundation is implemented (2026-10-07): `CausalPredictabilityTracker` provides
bounded confidence, speed/acceleration and mature-only observation-error signals
with snapshot/prefix-causality, clock/gap/loss and invalid-target tests. The
2026-10-10 continuation adds the pinned bank, policy, per-frame logs, two variants
and shared factory/experiment/UDP support. Threshold calibration, dedicated UI
option/live acceptance and final comparisons remain. The regression suite passes
694 Python tests and Ruff; the local pinned config also processes the 160-frame
diagnostic sequence with the real existing checkpoints. These are functionality
checks, not a live timing/accuracy claim. See docs/adaptive_compensation.md.

### Initial method

Use only causally available features:

- recent realized prediction error;
- joint velocity and acceleration;
- direction-change rate;
- detector confidence and validity rate.

Select one of the four fixed-horizon checkpoints and optionally attenuate its
output toward the latest observation. Any blend must preserve explicit pose
timestamps.

### Work

- implement predictability feature extraction;
- implement fixed-threshold adaptive policy first;
- tune thresholds on validation only;
- implement at least one ablation-friendly variant;
- log selected horizon, predictability score, alpha, and causal feature values;
- integrate live method switching without moving research logic into Unity.

### Tests

- causality of every adaptive feature;
- boundary behavior for invalid/low-confidence observations;
- deterministic policy decisions;
- safe fallback to Raw or shortest horizon;
- no quaternion component-wise blending.

### Acceptance

- adaptive and fixed methods run on identical held-out inputs;
- all adaptive decisions can be reconstructed from saved logs;
- at least one meaningful ablation is executable.

---

## M9. Required Experiments

Run in this order:

1. artificial-latency sanity experiment;
2. fixed-horizon sweep;
3. Raw/EMA/One-Euro/Kalman/Kalman-extrapolation comparison;
4. fixed learned predictor comparison;
5. smooth/medium/high-dynamic motion-class analysis;
6. adaptive versus validation-selected fixed comparator;
7. adaptive ablation;
8. deterministic noise stress tests;
9. real webcam-to-Unity validation.

### Reporting

- report per-sequence metrics and aggregate confidence intervals;
- report invalid-window coverage for effective latency;
- show Pareto fronts for latency, root-relative MPJPE, and primary jitter;
- apply the frozen `20 ms / 10%` decision rule from TZ/SPEC;
- do not use webcam observations without synchronized ground truth as
  quantitative accuracy evidence.

### Acceptance

- every result originates from a saved config;
- final tables and plots can be regenerated by documented commands;
- test results were not used for hyperparameter or threshold selection.

---

## M10. Finalization

### Work

- finish README setup and execution instructions;
- finalize architecture, pose format, experiment, and Unity protocol docs;
- add one-command demo and benchmark entry points;
- perform the 10-minute stability run;
- verify a clean-machine installation;
- generate final VKR plots and tables;
- archive exact configs, checkpoints, logs, and environment manifests.

### Definition of Done

The project is complete only when the VKR checklist in `TZ.md` and `SPEC.md` is
satisfied and the main adaptive-versus-fixed conclusion can be reproduced from
saved artifacts.

---

## 5. Immediate Work Queue

The initial scaffolding and review fixes are implemented. Remaining immediate work:

1. base-package clean-venv installation is verified (2026-10-05); validate the
   optional GPU setup separately (the working local environment is not proof);
2. measure a short person-present Raw run after the Unity render cap, followed by
   the optional calibration-preserving fast profile with other settings unchanged;
   report valid FPS and capture-to-send p95 for each run separately. The earlier
   ten-minute stability/delivery test passed, but live timing requirements did not;
3. assess nominal scale against ground truth during the Human3.6M/evaluation work;
4. finish M6 metrics and reproducible baseline experiments on prepared data;
5. retain M3's measured sustained delivery and assess remaining pose/orientation
   limitations; render FPS is not camera FPS or capture-to-screen latency;
6. the predeclared M7 robust noise/seed panel and comparator disposition are complete;
   M8's same-target Kalman/noisy-h100 mixture now improves macro error/jitter in
   the four-profile pilot and its three-noise-seed confirmation. Broader stress
   exposed a partial-tracking reset regression; the 54-cell recovery ablation
   corrects it on the rare-dropout profile without changing clean metrics.
   Comparable latency remains open, then freeze settings before held-out studies.
   Broader corruption/data and
   four-horizon validation still remain; the selected 100-ms pilot is not M7 completion.

Keep hardware-free canonical/transport tests available while live validation is
pending. Do not treat them as feasibility acceptance or start predictor training
before the baseline and causality gates. Dataset downloads, fresh installations,
and camera validation remain distinct from the offline regression suite.
