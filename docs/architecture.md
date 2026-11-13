# Architecture

## Current State

The optional Tk desktop panel owns one producer subprocess and consumes a bounded
event queue. It reuses live benchmark/replay and the same processor factories,
supports cooperative stop and produces isolated session artifacts. No web server,
network reconfiguration or UI-side scientific scoring is introduced. Unity remains
a separate window; one-way UDP sending is explicitly not receipt confirmation.
Only fresh, session-matched local Unity snapshots confirm receiver/applied counts
in the combined report. No network backchannel is introduced.
See [desktop controls](desktop_ui.md).

Implemented: the installable package, CLI and validated JSON config, canonical
pose and storage serialization, MeTRAbs observation adapter, latest-frame camera
capture, package-owned webcam orchestration with compatibility launchers, and UDP v2
sender/decoder/receiver. Python and C#
transport have automated tests. MeTRAbs supplies an estimated monocular metric root
used by the Unity container-position driver. Unity now has position-based Humanoid
retargeting, telemetry and a saved code-generated diagnostic scene/mannequin. Native
Humanoid/UDP tests and Python CMU replay passed in 6000.3.23f1; eight offline native
rendered phases were inspected. Person-camera ten-minute delivery and memory growth
are now measured; live timing budgets and imported-character acceptance remain open.
See [retargeting](unity_retargeting.md).
Recording/replay, causal baselines, predictor training and shared offline evaluation
are implemented. CMU conversion/training/validation pilots are available; official
Human3.6M CDF decoding, final held-out studies, adaptive policy and real-camera
learned acceptance remain unfinished. A complete robust validation panel selects
Kalman-CV as the 100-ms fixed comparator, not a global/test/live winner.
Initial causal M8 feature tracking is implemented, without processor/UI integration;
see [adaptive work](adaptive_compensation.md). No robust learned advantage is established.

## Target Components

```text
RGB frame
  -> perception backend (MeTRAbs PyTorch)
  -> canonical PoseFrame
  -> optional raw observation recording (before processing)
  -> selected causal PoseProcessor
  -> binary UDP display output
  -> Unity retargeting and visualization
```

Offline execution replaces the RGB/backend source with a dataset or recording
and uses the same `PoseFrame` and `PoseProcessor` interfaces.

## Dependency Direction

```text
core
  <- perception / processing / prediction / transport
  <- runtime / recording / evaluation
```

`core` must not import MeTRAbs, Unity, dataset, or experiment-specific types.
Backend adapters map external output into the canonical representation at the
boundary. Unity never contains training, prediction, adaptive policy, or primary
metric implementations.

`artifacts.sha256_file` is a stdlib-only leaf shared by data indexes, checkpoints,
evaluation, perception verification and research scripts. These consumers do not
import each other's private digest helpers. Byte hashes and manifest field names
are unchanged; source-code provenance hashes naturally change after refactoring.

`processing.config` owns baseline registration and lightweight processor-config
validation/path resolution. UI validation never instantiates a learned model.
Its experimental adaptive factory loads a pinned four-horizon model bank and
Python-owned causal feature tracker/threshold policy; Unity sees only the selected
output. Counterfactual probes are scored only against later arrived observations,
not evaluation truth. All decision features persist with processed recordings;
see [adaptive compensation](adaptive_compensation.md). Raw/live defaults are unchanged.
`runtime.webcam` owns live orchestration; the desktop worker calls it directly,
without runpy or dependence on a loose benchmark script. `runtime.memory` owns
resident-memory measurement; `runtime.run_report` owns Unity evidence/report merging.
Desktop event handlers are separate from Tk queue polling and can be tested without
a window, a camera or a producer process.

## Python Package Layout

```text
src/poseahead/
|-- artifacts.py
|-- core/
|-- perception/
|-- processing/
|-- prediction/
|-- transport/
|-- recording/
|-- evaluation/
|-- datasets/
|-- runtime/
|-- desktop/
`-- cli.py
```

Directories are created when their first real module is implemented; the
repository does not keep empty placeholder packages.

## Runtime Rules

- source/event timestamps are monotonic and drive all temporal logic;
- processing queues use latest-frame semantics and remain bounded;
- no live or offline processor may access a source frame from the future;
- backend-specific joint layouts never leave the perception adapter;
- generated and downloaded data lives outside `src/`.

Camera shutdown clears the pending frame and wakes consumers immediately. The
capture thread exclusively releases the device after its read loop exits;
`close()` never races `release()` against `read()`. If the driver is still blocked
after the default two-second shutdown timeout, `close()` raises an explicit
error. Python cannot safely kill a thread inside a native driver call. Once that
call returns, the worker discards the late frame and releases the device. Closing
before start and repeated close calls are supported; restarting a closed camera
requires a new instance. The desktop isolates the entire producer in its own
subprocess and can force-stop that exact process after explicit user confirmation,
with possible loss of unsaved observations. This does not provide independent
capture-process recovery or automatic device restart.
