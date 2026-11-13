# Reproducible baseline experiments (M6)

The same runner now also supports experimental `adaptive` processor configs,
four pinned checkpoints and explicit policy parameters. Per-frame decisions live
in `processed.npz`; `metrics.json.adaptive_prediction` reports choices and scope.
Target-aligned error for a partial blend is a nominal-target diagnostic, not
proof that the mixture represents a single future instant. Use display-time
metrics on matched inputs for comparisons. See [adaptive compensation](adaptive_compensation.md)
for the causal bank, two ablation modes and prospective validation pilot.

## Run the hardware-free sanity experiment

Use a new directory name if it already exists. No camera, GPU, Unity, downloads or
external network access is required.

```powershell
poetry run poseahead make-synthetic-experiment outputs/experiments/sanity
poetry run poseahead run-experiment outputs/experiments/sanity/raw.json outputs/experiments/sanity/raw_run
poetry run poseahead verify-experiment outputs/experiments/sanity/raw_run
```

The generator creates four seconds of analytic pulses at 100 Hz, a `batch.json`
manifest and five configs:
Raw, EMA, One-Euro, Kalman and Kalman CV. These signals are **not a realistic human
skeleton or Human3.6M substitute**. Their purpose is testing a known injected
100 ms delay and the metric/orchestration path. Initial filter defaults are not
tuned, and this example cannot establish which method is best on real motion.

To reproduce a completed run, use its resolved config and a new output directory:

```powershell
poetry run poseahead run-experiment outputs/experiments/sanity/raw_run/resolved_config.json outputs/experiments/sanity/raw_rerun
```

Input paths are resolved absolutely and pinned by SHA-256. Changed inputs are
rejected. Repeated metric values are deterministic in the same software environment;
runtime measurements are not. Across environments, compare with an explicitly
declared numerical tolerance, not a claim of bit-identical floating-point results.

## Configuration

```json
{
  "version": 1,
  "run_id": "raw-delay100",
  "observations": "observations.npz",
  "reference": "ground_truth.npz",
  "processor": {"method": "raw", "parameters": {}},
  "seed": 42,
  "delay_ms": 100,
  "start_ns": 200000000,
  "end_ns": 3990000000,
  "step_ns": 10000000,
  "window_frames": 380,
  "noise_std_m": 0.0,
  "dropout_probability": 0.0,
  "max_lag_ms": 200,
  "min_velocity_variation_m_s": 0.0001,
  "min_correlation": 0.6,
  "ambiguity_margin": 0.02
}
```

Input paths are relative to the config file. The inclusive evaluation grid is
`range(start_ns, end_ns + 1, step_ns)`; if end is not grid-aligned, the final sample
precedes it. Clock domains must match, and the reference must contain every grid
timestamp exactly. The runner never silently resamples reference data. Its current
limit is one million evaluation samples; long recordings should be split explicitly.

Processing consumes each available observation once, including observations before
the evaluation start so that filter history is not artificially reset at the first
display sample. Injected delay applies after observation production. Between arrivals
the last output is held. Actual processor execution duration is measured but is
not added to simulated time. Prediction horizon and delay remain independent.

The noise profile uses seeded NumPy PCG64 Gaussian noise independently on each
root-relative joint axis and independent per-frame joint dropout. The pelvis remains
at zero, root translation is not perturbed, original invalid joints remain invalid,
and no input file is edited. This is a controlled perturbation, not a calibrated
model of RGB detector errors, correlated noise or multi-frame tracking outages.

## Metrics and coverage

- `display_error`: unaligned canonical-17 MPJPE and separate root translation error
  at evaluation/display time, including the zero pelvis. Missing pairs are excluded
  and coverage is reported.
- `target_error`: error at the explicitly declared prediction target (or observation
  source time). Covers all observations processed up to the final evaluation,
  including pre-start warm-up observations. Reference interpolation is linear,
  bounded by the evaluation step; unavailable targets are counted as uncovered.
- `jitter`: acceleration residual against reference on the full evaluation grid.
- `latency`: non-overlapping fixed-size windows; median of eligible joint-window
  lags, with all per-window rejection reasons and coverage retained. Incomplete
  final windows are excluded only from latency and their sample count is reported.
- `runtime`: p50/p95 CPU duration of actual processor calls, including the first
  call. This excludes perception, transport, file I/O, metric computation and Unity.

Do not conflate target-time accuracy with displayed pose accuracy. Raw source-time
error can be zero while the delayed display still trails by 100 ms. Likewise,
lower median latency at lower coverage is not automatically an improvement.
See [temporal_metrics.md](temporal_metrics.md) for the signal-processing protocol.

The caller must supply genuine reference data with matching geometry. Sharing a
clock label is necessary but not proof of camera or ground-truth alignment.
The report explicitly does not certify caller-supplied references as ground truth.

## Artifacts and integrity

Each new output directory contains:

- `resolved_config.json`: absolute input paths, SHA-256 hashes and expanded defaults;
- `environment.json`: Python/NumPy/package/platform, Git revision and dirty status,
  plus a hash of all package Python source, including uncommitted source;
- `observations_used.npz`: exact perturbed observations used by this run;
- `processed.npz`: per-observation derived output (absent if nothing was available);
- `display_samples.npz`: evaluation-time positions/validity, source/target timestamps
  and an explicit observation-presence mask;
- `metrics.json`: metrics, protocols, coverage, runtime and window detail;
- `completion.json`: hashes of completed artifacts, written last.

`verify-experiment` checks manifest filenames, artifact hashes and consistent run
IDs. Checksums detect accidental modification; they are not cryptographic signatures
and do not authenticate artifacts against a malicious party replacing the manifest.
The display NPZ uses numeric arrays only and must be loaded with `allow_pickle=False`.

Existing output directories are never reused. Failure during publication leaves
an incomplete directory for diagnosis, without a completion marker. Do not interpret
such directories as completed results. Raw inputs and existing results are preserved.

Split-safe training-window manifests are available via [export-windows](training_windows.md).
Still pending: calibrated corruption profiles,
subject-level uncertainty analysis, validation-only parameter selection,
and licensed Human3.6M experiments. Source hashes
identify code but are not a source snapshot; preserve the corresponding repository
state alongside research results.

## Batch experiments

Newly generated synthetic fixtures include a suite manifest:

```powershell
poetry run poseahead run-batch outputs/experiments/sanity/batch.json outputs/experiments/sanity/suite
poetry run poseahead verify-batch outputs/experiments/sanity/suite
```

A real prepared-data suite uses config paths relative to its manifest:

```json
{
  "version": 1,
  "batch_id": "validation-baselines",
  "split": "validation",
  "experiments": ["walking_raw.json", "walking_ema.json", "sitting_raw.json"]
}
```

Supported splits are `train`, `validation`, `test`, and `unlabelled`. The latter is
for explicitly non-dataset runs such as synthetic sanity tests, not for silently
reclassifying labelled Human3.6M data. Each frame's dataset/subject/split label is
checked against the frozen policy. A suite cannot mix splits. Labelled observations
must match their reference's subject/split; unlabelled camera observations may use
a labelled reference when their declared clocks align. This does not authenticate
the source of manually supplied labels or guarantee camera geometry.

Before creating the suite directory, every config, input hash, reference label and
evaluation grid is checked. The suite snapshots resolved child configs with input
hashes; subsequent child loads detect replaced input files. Duplicate run IDs and
duplicate reference sequence/clock pairs within the same comparison condition are
rejected. Thus repeated seeds or crops of one labelled sequence do not count as
independent sequences. The check relies on stable sequence IDs, not content-based
motion deduplication.

Runs execute sequentially in `run-0000`, `run-0001`, etc. Summary conditions separate
processor parameters, delay, noise profile, sampling rate, latency window size and
metric thresholds. Within each condition, metric aggregation is an equal-weight
mean over eligible sequences, not pooled frames; null metrics are excluded and
eligible/total counts are reported. Every row retains pose/root/jitter/latency
coverage. Seeds and evaluation ranges are retained in child configs. Compare rows
on the same sequences/ranges before drawing conclusions; the summary does not enforce
a balanced method-by-sequence matrix or auto-select a winner.

`summary.json` contains per-run metrics and grouped descriptive means.
`batch_completion.json`, written last, hashes the summary, child configs and child
artifacts. `verify-batch` also verifies each child experiment and matching run IDs.
Failure leaves an incomplete suite without the completion marker; completed child
runs remain available. Existing directories are never reused, and resume is not
implemented. A batch is limited to 500 runs.

The batch summary itself is descriptive. The separate [paired comparison command](method_comparison.md)
now adds sequence-bootstrap intervals, frozen comparator checks, SPEC rule flags and
a point-estimate Pareto frontier. Different cameras/takes of one subject may be
correlated; subject-level sensitivity analysis and real-data conclusions remain pending.
