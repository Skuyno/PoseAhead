# Humanoid retargeting and diagnostic scene

## Implemented / verified status (2026-10-04)

Added position-based Humanoid retargeting, an Editor scene generator using our own
primitive mannequin, display telemetry and Python-to-localhost paced replay.
The C# smoke suite compiles all runtime scripts against explicit Unity type stubs
and exercises the real receiver/root/retarget components. Python regressions check
wire fixture equality, real socket delivery, clock rebasing and original-frame
immutability. These hardware-free checks are separate from the native runs below.

The user supplied `F:/Unity/Editor/Unity.exe`. Its product version is exactly
`6000.3.23f1_09d2ecc7fb28`, matching the project pin. Compilation of runtime and
Editor scripts against the actual installed Unity assemblies completed with zero
errors/warnings. After the user activated the license, actual native import/tests
also ran. Import exposed a missing Animation module; the manifest now explicitly
enables it and the IMGUI/Physics/ImageConversion modules needed by the scene/preview,
and pins bundled Test Framework 1.6.0.

The first headless EditMode launch exited with code **198** before tests/import:
`No valid Unity Editor license found. Please activate your license.` No results XML
was produced. This is a retained historical failure, not the current blocker.
Credentials/licensing settings were not changed by the agent. No project downgrade,
Editor install or VPN/network setting modification was performed. The earlier
executable search missed this F-drive installation; the old E-drive directory is a stub.

Native results after activation:

- three tests passed, zero failed/skipped in `outputs/unity_native_20261004_v7/editmode.xml`;
- valid Humanoid AvatarBuilder result, 12 bone bindings and native quaternion compatibility;
- actual UDP/Play Mode driving for all eight phases, segment directions/lengths,
  root translation, missing wrist, old-packet rejection, expiry and receiver shutdown;
- actual Python replay of one second of CMU `13_01.npz`: 119 accepted packets,
  114 observed applied frames, no rejects, final frame 118 and 12 applied bones;
- all eight native offline renders inspected on RTX 3070 Ti, with left orange/right
  cyan materials: `outputs/unity_scene_20261004_v3/preview.json` and PNGs;
- saved colored scene: `Assets/PoseAhead/Scenes/PoseAheadDiagnostic 1.unity`.

The native suite is discovered as **EditMode** because its factory uses Editor
APIs; coroutine tests explicitly enter real Play Mode and exit it in teardown.
It is not an empty PlayMode discovery or a stub-only check. An early attempted
`-testPlatform PlayMode` discovered zero tests and is not counted as passed.
Python replay uses an OS-chosen loopback test port, not the demo's port 5005.
The NPZ/source clocks and research results remain unchanged. Packet/display counts
are integration observations, not an FPS/latency benchmark.

The historical checks above did not establish person-camera acceptance. A later
600-second raw camera -> Unity run now verifies sustained delivery/application;
see the live update below. Remaining live timing, motion/orientation quality,
ground-contact limitations and imported-character checks are distinct. Research
model selection is not changed by a successful demo.

## Live update (2026-10-07)

`outputs/ui_20261005_140958_4ed33ebe/run_report.json` records 10,809 sent, received
and applied poses, zero rejected packets/receiver/rig errors, and a 600.26-second
Unity session. Unity working-set growth was 2.45 MiB. The producer passed memory
growth checks, but delivered only 18.01 valid poses/s with capture-to-send p95
98.43 ms. This is functional/stability evidence, not a throughput/latency pass or
synchronized accuracy evidence.

The demo now scopes a 60-FPS target and VSync-off setting to Play, restoring
previous owned settings on disable; overlapping demo instances are supported.
Telemetry reports target settings separately from measured Game-camera callbacks.
Four native tests passed in `outputs/unity_native_20261005_143910_da62a79d/`, including
scope restoration and real Python replay with the limit active. Those tests are
headless: no post-cap live speed or 60 render FPS is inferred. A short matched
person-camera measurement remains pending; see [desktop controls](desktop_ui.md).

## Mapping and unobservable rotations

`PoseHumanoidDriver` calibrates from the avatar's current imported T-pose, with a
valid Humanoid Avatar and no Animator Controller. It does not silently force an
arbitrary current pose into a T-pose. Assign its receiver, Animator and a fixed
pose-space/container Transform, with positive uniform scale. Root motion is disabled.
Calibration retains rest orientations and bone lengths; packets do not stretch
bones or replace their local positions.

| Driven Humanoid bone | Canonical segment | Additional orientation information |
| --- | --- | --- |
| Hips | pelvis -> spine (0 -> 1) | right hip minus left hip (14 - 11) |
| Spine | spine -> chest (1 -> 2) | rest local twist inherited from current parent |
| Chest | chest -> neck (2 -> 3) | right shoulder minus left shoulder (8 - 5) |
| Neck | neck -> head (3 -> 4) | rest local twist inherited from current parent |
| Left / Right UpperArm | shoulder -> elbow (5 -> 6 / 8 -> 9) | same rest-twist rule |
| Left / Right LowerArm | elbow -> wrist (6 -> 7 / 9 -> 10) | same rest-twist rule |
| Left / Right UpperLeg | hip -> knee (11 -> 12 / 14 -> 15) | same rest-twist rule |
| Left / Right LowerLeg | knee -> ankle (12 -> 13 / 15 -> 16) | same rest-twist rule |

Positions are converted `(x, y, z) -> (x, -y, z)` into pose space, then oriented
into Unity world space. Pelvis/chest use orthogonalized right/up body frames, so
they can follow body turns. Other segments apply a shortest-axis swing to the
current parent's orientation plus calibrated rest local rotation, retaining the
unobserved rest twist. The exactly antiparallel case uses a deterministic
perpendicular axis. There is no temporal smoothing or prediction in this layer.

Head, hands, feet, fingers and clavicles retain their local pose/inherit parent
motion. In particular, neck-to-head direction is not a measurement of head yaw or
gaze. Wrist/ankle endpoints cannot measure hand/foot orientation. Optional incoming
joint/root quaternions are deliberately not consumed by this position-only driver.
CMU/MeTRAbs landmark approximations are not anatomical Humanoid joint calibration.
No IK, foot locking, ground-contact correction or anatomical joint limits are added.

Application is parent-first and recomputed from the rest calibration, not integrated
from previous bone rotations. Missing/invalid/zero-confidence or short/degenerate
segments hold their last **local** bone rotation. Valid descendants may still be
driven. Default confidence floor is 0.1 and segment-length floor is 1 mm; these are
diagnostic guards, not validation-tuned tracking thresholds. A torso frame also
needs a nondegenerate lateral axis. Expired packets hold the last rendered pose.
Duplicate/old sessions/frames cannot change the rig; later sender sessions accept
reset frame IDs. After a sender host reboot/clock-epoch change, restart the whole
Unity scene to reset receiver and driver ordering state together.

## Generate a self-contained diagnostic scene

1. Open `unity/PoseAheadAvatar` in pinned **6000.3.23f1**. Do not downgrade the project.
2. Open the saved colored scene above, or run menu
   `PoseAhead > Create diagnostic Humanoid scene` to create a new one.
3. The generator creates a Humanoid Avatar asset and primitive mannequin, an avatar
   container with receiver/root driver, the bone driver, a camera, light, floor and
   telemetry. It validates Avatar.isValid/isHuman before assigning the Avatar.
4. Save paths under `Assets/PoseAhead/Scenes` are unique; existing scenes/assets are
   not overwritten. Creation is additive and preserves the previous active scene.
   A pristine empty untitled scene can be reused; otherwise save any open untitled
   scene first. The generator never discards or silently saves user scene contents.
   Open the newly generated scene alone for Play Mode to avoid a second receiver
   or unrelated cameras from another loaded scene.
5. Enter Play Mode, then send the deterministic diagnostic sequence:

```powershell
.venv/Scripts/python.exe scripts/replay_unity.py --diagnostic
```

The eight one-second phases are T-pose, raised left arm, bent left elbow toward
camera-away, raised left knee, whole-body yaw, root translation, missing left
wrist, then restored T-pose. This deliberately uses abrupt holds for direction/
tracking checks; it is not an animation-quality or research dataset. Primitive
geometry is generated from code, not a downloaded character. It can be replaced
later with an imported Humanoid using the same driver.

The overlay reports current packet frame/method/target offset, root estimate flag,
receiver health, accepted/rejected packets, applied/bound bone count and instantaneous
update FPS (not render FPS). It does not report cross-clock one-way latency or camera FPS. The
Recenter button resets only translation. Actual movement comes from the separate
root driver and may be a monocular estimate, not measured room position.

## Replay existing recordings

```powershell
.venv/Scripts/python.exe scripts/replay_unity.py --recording outputs/cmu_expanded_20260924/13_01.npz --duration 10
# Optional --processor-config uses the same frozen processor/config as other runners.
# Learned CMU/camera replay normally needs history_sampling: linear_past.
```

The command uses localhost only (port 5005, optionally `--port`) and actual recorded
availability spacing at speed 1. Original frames/times go to the causal processor;
only outgoing display copies receive a constant clock offset into the current
sender's monotonic domain. Source/production interval and prediction target offset
are preserved. This permits replay after a different host boot without modifying
the recording or fabricating send times. Rebased copies are not saved or scored as
research data. There is no automatic looping, accelerated playback, model fitting
or metrics claim. UDP sending alone does not prove reception/rendering.

The diagnostic telemetry component also atomically writes a local snapshot to
`Temp/PoseAheadTelemetry-<actual-port>.json` every 0.5 seconds while playing.
Desktop sessions merge only fresh, exact-session snapshots into `run_report.json`.
Counts distinguish received packets from poses applying at least one bone; rejected
packets are receiver-lifetime totals. Session RAM uses current resident memory;
Windows reads GetProcessMemoryInfo because Mono's WorkingSet64 can return zero.
Render callbacks count only on-screen Game cameras in this scene, once per Unity
frame, in the current built-in pipeline. Batch/headless FPS is never accepted;
callbacks do not measure monitor presentation or capture-to-screen latency.
An SRP migration would require a separate render instrumentation adapter.

## Verification commands and remaining gate

```powershell
dotnet run --project tests/dotnet/PoseAhead.Protocol.Smoke -c Release
.venv/Scripts/python.exe -m pytest tests/integration/test_unity_replay.py -q

# Native Unity tests; each run creates a new output directory and validates XML counts.
# Without -Recording, only the two self-contained tests run; external replay is skipped.
./scripts/test_unity.ps1 -UnityEditor 'F:/Unity/Editor/Unity.exe'
./scripts/test_unity.ps1 -UnityEditor 'F:/Unity/Editor/Unity.exe' -Recording 'outputs/cmu_expanded_20260924/13_01.npz'

# Requires the real pinned Editor installation; no stubs are used here.
dotnet build tests/dotnet/PoseAhead.Unity.ApiCompile "-p:UnityManagedPath=C:/Program Files/Unity/Hub/Editor/6000.3.23f1/Editor/Data/Managed"
# Verified installation on this machine: -p:UnityManagedPath=F:/Unity/Editor/Data/Managed
```

In Unity Test Runner, run the PoseAhead **EditMode** suite. To enable the optional
external replay test, set `POSEAHEAD_UNITY_REPLAY_RECORDING` before starting the
Editor; the PowerShell wrapper scopes this environment value to its launch only.
This test now runs the actual desktop worker and asserts session-matched combined
report counts, measured Windows resident memory and absent headless render FPS.
No dataset/Python environment is downloaded or installed by the test.

For reproducible offline native PNGs, run the Editor with `-batchmode -quit`,
`-projectPath unity/PoseAheadAvatar`,
`-executeMethod PoseAhead.Editor.DiagnosticScenePreview.Run`, and
`-poseAheadOutput` pointing to a fresh output directory. **Omit `-nographics`**:
this gate needs a real GPU. The generator creates unique scene/material/avatar
assets, applies eight real Python wire fixtures to an unsaved preview, writes PNGs
and `preview.json`, then restores the saved T-pose without saving preview poses.
Existing preview PNGs/report are refused, not overwritten.

Next, run the real-camera sender with a person fully visible. Check sides, torso
turn, bone lengths, loss/recovery and stream expiry; record Unity version,
receiver/applied counts and sustained render FPS separately from camera valid FPS.
The native checks above do not replace that camera acceptance gate.

API choices follow the pinned [GetBoneTransform contract](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/Animator.GetBoneTransform.html)
and [AvatarBuilder contract](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/AvatarBuilder.BuildHumanAvatar.html).
