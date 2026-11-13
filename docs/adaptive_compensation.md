# Adaptive compensation: causal model bank and threshold policy

The multi-seed/broader continuation is now in [frozen confirmation](adaptive_confirmation.md):
Gaussian gains repeat, but partial-joint dropout prevents broad adaptive selection.
Its negative result is retained and an opt-in partial-baseline recovery correction
is separately evaluated. Train-only motion classes are also available there.

### Latest development result (2026-10-11)

The same-target pilot completed **72 runs / 577 verified batch artifacts** in
`outputs/cmu_adaptive_baseline_20261011`. Its predeclared complete-panel rule selects
`kalman_noisy_h100_tau25_cap25`: existing noise-trained h100, 25-ms causal model-history
EMA, neural weight capped at .25, Kalman fallback, paired error ratio <=1,
200-ms dynamics window and 80-ms transitions. The worst corrupted error/jitter
ratio is **0.955192**, against Kalman-CV's 1.0. Selection was rechecked against
pinned artifacts. No model weights or old comparator files changed.

Three-clip equal-weight point means (single noise seed 42, no confidence intervals):

| Gaussian sigma | Kalman-CV error mm / jitter m/s2 | Selected mixture | Mean neural weight |
| --- | ---: | ---: | ---: |
| 0 mm | 25.34 / 6.20 | 24.79 / 6.17 | 0.048 |
| 10 mm | 31.90 / 36.70 | 30.47 / 34.42 | 0.118 |
| 25 mm | 47.72 / 88.66 | 43.75 / 80.40 | 0.186 |
| 50 mm | 77.66 / 176.20 | 69.28 / 157.50 | 0.231 |

Weights are frame-weighted diagnostics; metric means use equal clip weights.
Clean error/jitter change by -2.17%/-0.44%; sigma50 changes by -10.79%/-10.61%.
All four profile macro error and jitter means improve in this development panel,
not necessarily every individual clip/window. The clean-model mixture's worst
ratio is 1.042675; noisy tau10 yields 0.996868; noisy tau25/cap10 yields 0.979823.
This supports the noise-trained same-target branch rather than just re-enabling
the earlier Raw-based model (worst ratio 1.820170).

**No latency superiority is established**: selected-candidate eligibility masks
differ from Kalman's, and some noisy baseline windows have no valid lag metric.
The selector's `advantage_claim` remains `not_assessed`; there is no paired 95%-CI
three-metric or 20-ms/10% victory claim. Repeated validation, one noise seed and
three clips are insufficient for held-out/participant-level generalization.
The M7 fixed comparison baseline remains Kalman-CV; the new selection is an
experimental adaptive configuration, not a replacement fixed comparator or live
default. The local custom-JSON example now uses selected cap25 instead of its
initial cap10 diagnostic setting; archived pilot configs are immutable/unchanged.

Next predeclare and run a multi-noise-seed confirmation, broader corruption/data
checks and a comparable latency study; freeze the adaptive config before eventual
held-out evaluation. Real-camera inference cost/accuracy remains unverified.

## Same-target strong-baseline branch (2026-10-11)

Optional `fallback_method: kalman_cv` now enables an alternative strength-only
adaptive branch. The default remains `raw`, preserving existing configurations.
Kalman-CV runs on every usable observation and predicts the policy's fixed maximum
horizon. Only the neural model for that **same horizon** can contribute:
`output = (1 - alpha) * Kalman(target) + alpha * neural(target)`.
Other bank models remain causal probes; this branch does not select a shorter
horizon or blend current-time Raw into a future-time estimate. The output target
is the shared future instant even when alpha is zero. Root translation always
comes from Kalman-CV, not the neural bank; rotations remain unsupported/absent.

Additional parameters, valid only with Kalman-CV fallback:

- `fallback_parameters`: optional Kalman filter parameters, normalized/persisted;
  target_offset_ms cannot be overridden (the policy budget determines it).
- `maximum_neural_weight`: finite [0, 1], caps the requested neural contribution
  before source-time transitions. Zero yields the exact standalone Kalman-CV
  joint/root positions on a continuous fully valid stream with identical parameters.
- `neural_to_baseline_error_ratio`: optional positive finite ratio. The new pilot
  uses 1.0: require the neural forecast's recent error not to exceed Kalman's.
  Both errors use raw arrived observations at identical anchors/targets, and
  Kalman probe registration occurs only where a neural probe exists, making the
  sliding sample windows paired. Neither error is ground-truth pose accuracy.

Unknown/insufficient paired error blocks the neural contribution, not the baseline.
Version-1 confidence/dynamics/error gates still apply to the neural policy. A
valid-stream gate can release the neural weight gradually into Kalman; tracking
loss/low confidence/gaps clear transitions/filter/error support immediately and
produce original Raw with target offset zero. On reacquisition the filter restarts,
so standalone equality is not claimed across invalid/gapped streams. This is
conservative tracking-loss behavior, not a neural performance guarantee.

The causal logs distinguish policy-eligible/probe horizons, requested decision,
paired baseline features, actual alpha, baseline weight/method/target and root
policy. UDP ID 7 identifies Kalman-only future output (`prediction_status: baseline`),
ID 20 a positive neural mixture and ID 0 actual Raw on loss. Alpha zero no longer
implies target zero in this explicitly named baseline branch; strict checks guard
weight sum, same-horizon pairing, cap and identity. Layout/version are unchanged.
The experiment report labels alpha as neural weight against the future baseline
and counts baseline-only frames separately, not as neural forecasts.

### Prospective development pilot

`scripts/cmu_adaptive_baseline.py` declares 72 validation cases before scores:
six candidates, four Gaussian profiles, three unchanged clips, noise seed 42.
Candidates are Kalman-CV, the previous Raw-based combined adaptive method, and
four same-target variants: clean h100/tau10/cap25, noisy h100/tau10/cap25,
noisy h100/tau25/cap25 and noisy h100/tau25/cap10. They use 200-ms dynamics,
80-ms transitions, a 100-ms shared target, 20-m_s2 acceleration gate, three
matured observations and paired error ratio <=1. The absolute observation-error
limit is predeclared at 150 mm (not the earlier 60 mm), because observed noise
contributes to both predictor residuals; this pilot tests that new rule, not
that limit change in isolation.

The retained `outputs/cmu_noise_training_20261003_v2/model` checkpoint replaces
**only h100**; noisy models for 33/66/150 ms have not been trained. All weights
remain pinned and unchanged; no training or test scoring is done in this pilot.
One noise seed and repeated validation use preclude a held-out superiority claim.
Production/camera defaults and earlier pilot artifacts remain unchanged.
`configs/processing/adaptive_kalman_noisy_pilot.json` is a local **experimental**
custom-JSON example for the selected noisy h100/tau25/cap25. It requires the
retained generated checkpoints, is not a bundled asset and is not recommended
as a live default before independent validation.

Implementation verification: 757 Python tests and Ruff pass. Coverage includes
exact continuous-stream standalone Kalman fallback, paired probe counts, common
target/root policy, neural cap, rejecting a worse neural forecast with a permissive
absolute limit, gap/loss resets, prefix/reset determinism, UDP identity/cap/weight
checks and resolved-config experiment replay. Existing Raw-based regressions pass.
The pinned noisy example also processed 160 diagnostic frames with real weights
and UDP encode/decode: 69 Kalman-only, 71 neural mixtures and 20 loss-Raw outputs.
No camera/Unity window was opened; these counts establish functionality, not
human-pose accuracy, native avatar acceptance or live timing.
Replaying the previous Raw-based combined variant reproduced all archived pose,
jitter, latency and decision metrics exactly in
`outputs/cmu_adaptive_stability_compatibility_20261011`; implementation provenance
and measured CPU timing naturally differ. The pinned M7 fixed-comparator choice
is not replaced by this adaptive pilot's descriptive candidate selection.

```powershell
.venv/Scripts/python.exe scripts/cmu_adaptive_baseline.py `
  outputs/cmu_expanded_20260924/index.json outputs/cmu_stability_20260927 `
  outputs/cmu_noise_training_20261003_v2/model `
  outputs/cmu_adaptive_baseline_20261011 --prepare-only
.venv/Scripts/python.exe scripts/cmu_adaptive_baseline.py `
  outputs/cmu_expanded_20260924/index.json outputs/cmu_stability_20260927 `
  outputs/cmu_noise_training_20261003_v2/model `
  outputs/cmu_adaptive_baseline_20261011 --run-prepared
```

## Experimental processor (2026-10-10)

`poseahead.processing.adaptive.AdaptiveProcessor` is available through the shared
processor factory (`method: adaptive`) and custom JSON configuration. It is not a
new production default or an automatically selected UI menu method. Camera use
requires explicit `history_sampling: linear_past`, as for standalone learned models.
`configs/processing/adaptive_cmu_pilot.json` is a pinned **unselected experimental**
configuration for the retained local clean CMU bank. It requires those generated
checkpoints and is not a recommendation or clean-machine asset. The existing UI's
custom JSON option accepts it; the extrapolation dropdown does not set its budget.

The bank requires exactly four checkpoint entries with string keys `33`, `66`,
`100`, `150`; each contains `checkpoint` and `checkpoint_sha256` (manifest digest).
Relative paths resolve against the processor/experiment configuration file. Loaded
horizons must match their keys and share history length/sample spacing. Each model
uses the existing causal LearnedProcessor sampler and its own bounded history;
history buffers are not yet pooled. All four models see each observation and issue
probe forecasts even if the output is Raw. These probes are not displayed outputs.

Processing order is observe/mature earlier probes, compute current-anchor forecasts,
choose from the pre-registration feature snapshot, register probes, publish output.
No reference pose, future suffix or current-anchor future error enters selection.
After an inference/stream failure the processor refuses further use until explicit
reset; it never reuses a partially advanced bank. Missing/low-confidence input
clears model support as well as tracker support. Gaps clear support in each sampler.

### Version 1 policy (explicit caller-supplied thresholds)

The required `policy` object has these fields, with no inferred calibration:

- `version: 1`;
- `error_limits_m`: four string horizon keys with positive finite limits in metres;
- `max_horizon_ms`: 33, 66, 100 or 150, an explicit compensation budget, not an
  online measurement of camera/display latency;
- `minimum_matured`: number of matured forecasts, at most `error_history`;
- `maximum_acceleration_m_s2`: positive finite acceleration gate;
- `mode`: `horizon_only` or `horizon_blend`.

The policy requires ready kinematic support and acceleration within the limit,
then chooses the longest available horizon within the budget whose recent matured
observation error has enough samples and is within its own error limit. Unknown
error is not zero. No eligible horizon requests Raw (the optional transition
layer below can release existing weights gradually while tracking remains valid). The margin
`score = clamp(1 - recent_error / error_limit, 0, 1)` is **not a probability**.
Horizon-only uses alpha 1; horizon-blend uses alpha = score (zero-margin forecasts
are skipped). This supplies a runnable blending ablation. Thresholds and gates are
heuristics until separately validated; no hysteresis/stability guarantee is implied.

Raw fallback preserves the original positions, root and rotations and sets
target offset zero; UDP identifies it as Raw. Positive alpha holds the observed
root, drops unsupported rotations and uses Adaptive ID 20, with the **nominal
forecast horizon**, not alpha times that horizon. For 0 < alpha < 1, metadata
`output_time_semantics` explicitly records a current/future partial-compensation
mixture: it is not an estimate at one interpolated timestamp. The wire protocol
does not carry alpha or the full decision log; its horizon is nominal, not a
measured achieved reduction in delay. The uncalibrated margin is not sent in the
optional predictability-probability field. No wire version or Unity code changed.

Every processed frame carries `adaptive_decision`: horizon, alpha, reason, margin,
policy version/mode, available probe horizons and the immutable causal feature
snapshot. `processed.npz` preserves these logs. Config-driven experiment reports
include horizon counts, mean alpha, partial-compensation counts and scope labels.
Target-aligned error uses the declared nominal target even for a mixture; the
matched **display-time** metrics remain the primary comparison, not this diagnostic.
CPU time includes all four model calls but is not injected into simulated latency.

Verification: 694 Python regression tests and Ruff pass. Tests cover thresholds,
causal prefixes, four-horizon maturity/probes, confidence/loss/gaps, failure/reset,
checkpoint/grid/clock/target contracts, original rotations on raw fallback,
nominal-target blend semantics, UDP identity, decision reconstruction after NPZ
roundtrip and deterministic resolved-config experiment replay. The local pinned
config also processed 160 diagnostic frames with the real retained checkpoints;
no camera, Unity window or network settings were changed for this verification.

### Initial validation sweep

`scripts/cmu_adaptive_validation.py` freezes a 72-run CMU validation pilot before
scoring: Kalman-CV/Raw plus two error limits (30/60 mm) x two policy modes, four
Gaussian profiles (0/10/25/50 mm), one noise seed (42), three validation clips.
The bank is the unchanged clean seed-42 checkpoints for four horizons. The budget
is 100 ms, acceleration gate 20 m/s2, minimum matured count 3; these are prospective
pilot candidates, not finalized motion-class thresholds. No models are trained
and no test data are scored. The existing complete-panel rule keeps the Kalman-CV
baseline, including its clean 10% constraints and explicit latency-mask limitations.
Repeated validation use, one noise seed and small source-ID sample make this a
development pilot, not a held-out superiority claim or final robust calibration.

```powershell
.venv/Scripts/python.exe scripts/cmu_adaptive_validation.py `
  outputs/cmu_expanded_20260924/index.json `
  outputs/cmu_stability_20260927 outputs/cmu_adaptive_validation_20261010 --prepare-only
.venv/Scripts/python.exe scripts/cmu_adaptive_validation.py `
  outputs/cmu_expanded_20260924/index.json `
  outputs/cmu_stability_20260927 outputs/cmu_adaptive_validation_20261010 --run-prepared
```

Choose a new output directory to prepare another study. Completed/incomplete
batches are not overwritten. Live raw settings and the earlier robust study remain
unchanged. Final multi-seed calibration, broader corruptions, larger/independent
data, adaptive-vs-fixed inference, camera validation and held-out study remain open.

### Pilot result (2026-10-10)

All **72 runs / 577 batch artifacts** completed and verified in
`outputs/cmu_adaptive_validation_20261010`. `selection.json` retains Kalman-CV as
the selected comparator. All four adaptive variants exceed the clean jitter budget;
Raw exceeds the clean error budget. No adaptive configuration is promoted or
validation-frozen as an accepted deployment method. The original 252-run selection
was rechecked and is unchanged.

Equal-weight means over the three validation clips on the clean profile, with no
participant-level CIs (the limit is matured observation error, not MPJPE):

| Candidate | Display MPJPE, mm | Jitter, m/s2 |
| --- | ---: | ---: |
| Kalman-CV 100 ms | 25.34 | 6.20 |
| Raw | 29.28 | 4.80 |
| Adaptive, 30-mm limit, horizon-only | 21.12 | 14.57 |
| Adaptive, 30-mm limit, horizon/blend | 24.67 | 8.55 |
| Adaptive, 60-mm limit, horizon-only | 20.80 | 16.90 |
| Adaptive, 60-mm limit, horizon/blend | 22.87 | 10.12 |

Each adaptive variant compensates 6,982 of 7,281 processed clean observations
(95.89%; frame-weighted diagnostic, not the equal-clip selection statistic).
Across the noisy adaptive runs, **all 87,372 outputs are Raw**: independent
source-frame Gaussian noise at 120 Hz trips the raw finite-difference acceleration
gate. Consequently their noisy display metrics exactly match Raw. At sigma10,
that is 36.08 mm / 58.95 m/s2 versus Kalman-CV 31.90 mm / 36.70 m/s2.
Different latency eligibility masks prevent a latency-win claim; some high-noise
cells have zero eligible latency windows.

This pilot demonstrates causal functionality and a concrete failure mode, not
research acceptance. Next develop a noise-tolerant causal dynamics signal and
assess less abrupt horizon/alpha transitions, then predeclare another complete
validation/ablation panel. Do not fix the issue by selecting thresholds on test
data or silently replacing the already selected baseline. The present experimental
config is retained for reproducibility, not recommended for live output.

### Noise-tolerant dynamics and smooth transitions (2026-10-10 continuation)

Two optional processor parameters are now implemented, off by default so the
original version-1 configurations remain executable:

- `dynamics_window_ms`: zero uses the original raw finite differences; an integer
  in [20, 1000] fits a quadratic to the trailing arrived observation window.
  Speed/acceleration are derivatives of that fit at the current source timestamp,
  using actual irregular times and dimensionless time scaling for conditioning.
  The fit requires a full window and at least three samples. One sample before the
  cutoff is retained as a bracket, with a hard capacity of 256 observations. If
  capacity cannot span the window or the fit is rank deficient, features remain
  warmup/unknown, not zero acceleration. Loss/confidence/gaps clear support. This
  changes **only dynamics features**; matured prediction error still uses raw
  arrived observations. It is a causal approximation, not a motion-class label or
  calibrated predictability probability.
- `transition_time_constant_ms`: zero preserves immediate single-horizon output;
  positive finite values up to 1000 apply source-time exponential transitions to
  four nonnegative forecast weights, whose sum is at most one. Every output is
  `(1 - sum(weights)) * current_observation + sum(weight[h] * current_forecast[h])`.
  No earlier pose array is blended into the current output. Small release tails
  below 1e-6 are pruned. Missing forecasts lose their weights immediately; invalid/
  low-confidence tracking, gaps, insufficient dynamics support or reset clear all
  weights immediately. Valid tracking with a policy acceleration/error rejection
  releases existing weights gradually: this deliberately trades safety-gate
  response speed against jitter and needs validation, not a stability guarantee.

During a horizon transition, several current-anchor forecasts may contribute.
The UDP nominal horizon is the **largest positive contributing horizon**, not
the requested horizon, an alpha-weighted time or achieved latency compensation.
Metadata records `forecast_weights`, `requested_decision`, actual output alpha/
horizon and `output_time_semantics: mixed_current_anchor_forecast_weights`.
Weights and requests reconstruct the stateful transitions exactly after NPZ reload;
UDP checks their sum and nominal-horizon consistency. The decision score remains
the requested policy margin; it is not the reliability of the mixture. Root and
rotation/fallback semantics are unchanged. This layer is distinct from the existing
`history_prefilter_ms`, which filters model input history only.

The new prospective ablation uses the same clean checkpoint bank and 60-mm error
limits/20-m_s2 acceleration gate/3 matured samples/100-ms budget, without training:
Kalman-CV, Raw, original e60 blending, 200-ms dynamics window, window + 10-ms model
history EMA, and window/history + 80-ms transitions. Four Gaussian profiles and
three unchanged validation clips, one noise seed 42, give 72 runs. Keeping the
gates fixed separates mechanism changes from threshold changes; this is still an
exploratory continuation, not final robust calibration or held-out inference.

```powershell
.venv/Scripts/python.exe scripts/cmu_adaptive_stability.py `
  outputs/cmu_expanded_20260924/index.json `
  outputs/cmu_stability_20260927 outputs/cmu_adaptive_stability_20261010 --prepare-only
.venv/Scripts/python.exe scripts/cmu_adaptive_stability.py `
  outputs/cmu_expanded_20260924/index.json `
  outputs/cmu_stability_20260927 outputs/cmu_adaptive_stability_20261010 --run-prepared
```

The original pilot/artifacts and live configurations are not changed by this panel.
`configs/processing/adaptive_cmu_stability_pilot.json` exposes the combined
window/history/transition candidate through custom JSON for local diagnostics.
It is an **unselected experimental** example, not a live recommendation or bundled
checkpoint asset; the original pilot config remains unchanged.
Verification of this continuation: 735 Python tests and Ruff pass, including
exact/irregular quadratic derivatives, noise attenuation, full-window/capacity/loss
support, transition simplex/time-step invariance, mixed-horizon metadata checks,
immediate hard fallback, soft release on current probes, prefix causality, NPZ
decision/transition reconstruction and resolved-config experiment replay. Replaying
the original v1 clean e60 horizon-only run reproduced all archived pose/jitter/
latency/decision metrics exactly (`outputs/cmu_adaptive_v1_compatibility_20261010`);
source/environment/runtime artifacts naturally differ after implementation changes.
The combined local config was also exercised on 160 real-checkpoint diagnostic
frames, including tracking-loss resets and UDP encode/decode roundtrips; 126 frames
had positive compensation. This is a functionality check, not a native Unity/live
camera or measured accuracy acceptance.

### Stability ablation result

All **72 runs / 577 batch artifacts** in `outputs/cmu_adaptive_stability_20261010`
completed and verified. The complete-panel selector retains **Kalman-CV**. Only
the combined adaptive candidate meets the clean error/jitter constraints; its
worst corrupted-profile ratio is **1.82017**, versus baseline 1.0, so it is not
selected or promoted. Latency masks differ, prohibiting a latency-win claim.

Equal-weight three-clip clean-profile means, without confidence intervals:

| Candidate | Display MPJPE, mm | Jitter, m/s2 |
| --- | ---: | ---: |
| Kalman-CV | 25.34 | 6.20 |
| Original e60 horizon/blend | 22.87 | 10.12 |
| 200-ms dynamics window | 22.40 | 8.38 |
| Window + 10-ms model-history EMA | 23.49 | 7.34 |
| Window/history + 80-ms forecast-weight transitions | 23.14 | 6.80 |

The combined candidate has clean error **8.68% lower** and jitter **9.69% higher**
than Kalman-CV, just inside the descriptive 10% clean constraint. This is a
validation gate result, not a confidence-interval-based superiority claim.

| Gaussian sigma | Kalman-CV error mm / jitter m/s2 | Combined adaptive | Mean output alpha |
| --- | ---: | ---: | ---: |
| 0 mm | 25.34 / 6.20 | 23.14 / 6.80 | 0.604 |
| 10 mm | 31.90 / 36.70 | 34.47 / 66.81 | 0.284 |
| 25 mm | 47.72 / 88.66 | 52.18 / 143.36 | 0.075 |
| 50 mm | 77.66 / 176.20 | 85.94 / 292.19 | 0.000 |

Combined compensation is positive on 7,197/7,281 processed observations (98.85%)
for each of sigma0/10/25, versus zero noisy compensation in the original pilot.
Fractions/alpha above are frame-weighted diagnostics, not equal-clip selection
statistics; high fraction at sigma25 means weak compensation, not 66-ms achieved
delay reduction. Sigma50 still returns Raw on all observations. Window fitting
fixes raw acceleration-noise amplification, but activating a clean-trained model
can worsen noisy jitter (sigma10 is worse even than Raw's 58.95). Noise robustness
is therefore not established by simply enabling predictions.

Next investigate a filtered strong-baseline fallback/mixture and a noise-trained
bank or noise-aware reliability calibration, then predeclare a complete panel.
The present 60-mm raw-observation consistency limit is not noise-invariant. Keep
Kalman-CV as the selected comparator and do not tune against held-out test data.
Multi-seed/larger-data inference and real-camera cost remain separate acceptance gates.

## Implemented first step (2026-10-07)

`poseahead.prediction.predictability.CausalPredictabilityTracker` provides bounded,
auditable signals for M8. It is now used by the experimental processor above, but
the tracker itself does not choose a policy or change output. No calibrated
probability or method advantage is claimed.

Usage order:

1. `observe(observation)` consumes an arrived canonical PoseFrame and returns an
   immutable feature snapshot: confidence, mean joint speed/acceleration, recent
   matured observation-error means/counts, pending/discarded prediction counts.
2. Compute fixed-horizon predictions from already available history.
3. `record_prediction(prediction)` registers a forecast from exactly that usable
   anchor/frame/clock, with explicit 33/66/100/150-ms target metadata. Future poses
   cannot enter `observe`; duplicate anchor/horizon registrations are rejected.
4. At a subsequent observation, score only forecasts whose target time has now
   arrived. Interpolate between the two arrived observations bracketing the target;
   never read evaluation reference/ground truth or a future stream suffix.

The error is consistency with observed joints, **not ground-truth pose accuracy**.
Interpolation is an explicit online approximation. Speed, acceleration and error
use the 16 non-pelvis root-relative joints (the zero pelvis is excluded), in m/s,
m/s² and metres; root translation/rotations are not analyzed. Acceleration uses
velocity-midpoint spacing, including irregular source-frame intervals.

Missing/zero-confidence observations, configurable low confidence and gaps above
100 ms by default clear support/error history and discard pending predictions.
Unknown recent error remains `None`, not a perfect zero. Ordering and produced/source
clock checks reuse the public Raw processor validator. Explicit reset is required
for a new source clock/order and clears all diagnostic state.

Each horizon retains at most 64 pending forecasts and 8 matured errors by default;
capacities are validated, oldest pending overflow is counted, and retained position
arrays are copied. Features are snapshots rather than mutable views into future state.
Tests cover maturity, interpolation, prefix causality, all four horizons, irregular
acceleration, loss/gap/clock resets, bounded buffers and invalid targets.

## Remaining M8 work

- Select/freeze usable thresholds on a complete validation panel; a tested policy
  implementation is not validation-selected calibration or a predictability probability.
- Validate the implemented causal dynamics/history filtering and forecast-weight
  transitions; consider model-bank choice/hysteresis if the ablation warrants it.
  The optional same-target branch also supports the existing noise-trained h100;
  noisy models for the other horizons remain untrained.
- Expand the horizon-only/combined ablation to multi-seed/operating-point comparisons.
- Expose a dedicated UI option only after selecting a usable experimental config
  and verifying real-camera inference cost. Custom config execution already works.
- Validate causality/fallback/decision reproducibility and compare with the robustly
  selected fixed comparator on matched data, then frozen held-out studies.

The current 100-ms validation comparator is Kalman-CV, selected by the complete
robustness panel described in [robust selection](robust_selection.md). Four-horizon
selection/threshold calibration and the final adaptive-vs-fixed conclusion remain
open. Test data and production defaults are unchanged.
