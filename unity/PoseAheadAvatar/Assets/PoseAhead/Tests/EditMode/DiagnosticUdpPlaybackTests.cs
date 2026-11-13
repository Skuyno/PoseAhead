using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using NUnit.Framework;
using PoseAhead.Editor;
using PoseAhead.Retargeting;
using PoseAhead.Transport;
using UnityEngine;
using UnityEngine.TestTools;

namespace PoseAhead.Tests
{
    public sealed class DiagnosticUdpPlaybackTests
    {
        [System.Serializable]
        private sealed class PythonReplayReport
        {
            public int sent_frames = 0;
            public string sender_session_id = null;
        }
        [System.Serializable]
        private sealed class WorkerRequest
        {
            public string source = "recording", recording;
            public double duration_seconds = 2;
            public int udp_port;
        }
        [System.Serializable]
        private sealed class CombinedReport
        {
            public PythonReplayReport producer = null;
            public PoseDemoTelemetry.Snapshot unity = null;
        }
        [System.Serializable]
        private sealed class WorkerProgress
        {
            public string kind = null;
            public string sender_session_id = null;
            public int sent_frames = 0;
            public ProgressUnity unity = null;
        }
        [System.Serializable]
        private sealed class ProgressUnity
        {
            public string status = null;
            public string session_id = null;
            public int udp_port = 0;
            public int received_pose_frames = 0;
        }
        private static readonly (HumanBodyBones start, HumanBodyBones end, int a, int b)[] Segments = {
            (HumanBodyBones.Hips, HumanBodyBones.Spine, 0, 1),
            (HumanBodyBones.Spine, HumanBodyBones.Chest, 1, 2),
            (HumanBodyBones.Chest, HumanBodyBones.Neck, 2, 3),
            (HumanBodyBones.Neck, HumanBodyBones.Head, 3, 4),
            (HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm, 5, 6),
            (HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftHand, 6, 7),
            (HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm, 8, 9),
            (HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand, 9, 10),
            (HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg, 11, 12),
            (HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot, 12, 13),
            (HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg, 14, 15),
            (HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot, 15, 16),
        };

        private static Vector3 Point(PosePacket packet, int index)
        {
            return new Vector3(packet.JointPositions[index * 3], -packet.JointPositions[index * 3 + 1],
                               packet.JointPositions[index * 3 + 2]);
        }

        [UnityTest]
        public IEnumerator ActualUdpDrivesEightPhasesAndHoldsExpiredPose()
        {
            // The factory uses Editor APIs. Run this Editor-only test in actual
            // Play Mode explicitly rather than silently discovering zero player tests.
            yield return new EnterPlayMode();
            var mannequin = DiagnosticSceneBuilder.CreateHumanoid();
            var animator = mannequin.GetComponent<Animator>();
            Avatar avatar = animator.avatar;
            var container = new GameObject("DiagnosticUdpTest");
            container.SetActive(false);
            mannequin.transform.SetParent(container.transform, false);
            var receiver = container.AddComponent<PoseUdpReceiver>();
            typeof(PoseUdpReceiver).GetField("port", BindingFlags.NonPublic | BindingFlags.Instance)
                .SetValue(receiver, 0); // OS-chosen loopback port: never contend with the demo receiver.
            var rootDriver = container.AddComponent<PoseRootDriver>();
            rootDriver.Configure(receiver, container.transform);
            var driver = mannequin.AddComponent<PoseHumanoidDriver>();
            driver.Configure(receiver, animator, container.transform);
            var locals = new Dictionary<Transform, Vector3>();
            foreach (var bone in mannequin.GetComponentsInChildren<Transform>(true))
                locals.Add(bone, bone.localPosition);
            byte[][] fixtures = DiagnosticPoseFixtures.LoadPackets();
            using (var sender = new UdpClient(AddressFamily.InterNetwork))
            try
            {
                mannequin.SetActive(true);
                container.SetActive(true);
                yield return null;
                Assert.That(driver.IsCalibrated, Is.True, driver.LastError);
                Assert.That(receiver.IsReceiving, Is.True);
                var socket = (UdpClient)typeof(PoseUdpReceiver)
                    .GetField("client", BindingFlags.NonPublic | BindingFlags.Instance).GetValue(receiver);
                var endpoint = (IPEndPoint)socket.Client.LocalEndPoint;
                Quaternion lastLowerArm = Quaternion.identity;
                for (int phase = 0; phase < fixtures.Length; phase++)
                {
                    Assert.That(PosePacketDecoder.TryDecode(fixtures[phase], out var packet, out var error),
                                Is.True, error);
                    sender.Send(fixtures[phase], fixtures[phase].Length, endpoint);
                    float deadline = Time.realtimeSinceStartup + 3f;
                    while (driver.AppliedFrameId != packet.FrameId || receiver.AcceptedPackets < phase + 1)
                    {
                        Assert.That(Time.realtimeSinceStartup, Is.LessThan(deadline), "UDP phase " + phase);
                        yield return null;
                    }
                    // The runner can resume before LateUpdate; check after another complete frame.
                    yield return null;
                    Assert.That(receiver.LatestPacket, Is.Not.Null);
                    Assert.That(driver.AppliedBoneCount, Is.EqualTo(phase == 6 ? 11 : 12));
                    Assert.That(Vector3.Distance(container.transform.position,
                        phase == 5 ? new Vector3(.3f, .15f, -.2f) : Vector3.zero), Is.LessThan(1e-5f));
                    foreach (var pair in locals)
                        Assert.That(Vector3.Distance(pair.Key.localPosition, pair.Value), Is.LessThan(1e-5f),
                                    "Bone/geometry length changed: " + pair.Key.name);
                    foreach (var segment in Segments)
                    {
                        if (!packet.JointValidity[segment.a] || !packet.JointValidity[segment.b]) continue;
                        Vector3 actual = (animator.GetBoneTransform(segment.end).position -
                                          animator.GetBoneTransform(segment.start).position).normalized;
                        Vector3 expected = (Point(packet, segment.b) - Point(packet, segment.a)).normalized;
                        Assert.That(Vector3.Distance(actual, expected), Is.LessThan(1e-4f),
                                    "Phase " + phase + ", " + segment.start);
                    }
                    Quaternion lowerArm = animator.GetBoneTransform(HumanBodyBones.LeftLowerArm).localRotation;
                    if (phase == 6)
                        Assert.That(Quaternion.Angle(lowerArm, lastLowerArm), Is.LessThan(.01f));
                    lastLowerArm = lowerArm;
                }
                sender.Send(fixtures[0], fixtures[0].Length, endpoint); // Delayed old packet.
                float rejectionDeadline = Time.realtimeSinceStartup + 3f;
                while (receiver.RejectedPackets == 0)
                {
                    Assert.That(Time.realtimeSinceStartup, Is.LessThan(rejectionDeadline));
                    yield return null;
                }
                var finalRotations = new Dictionary<Transform, Quaternion>();
                foreach (var bone in locals.Keys) finalRotations.Add(bone, bone.localRotation);
                float expiryDeadline = Time.realtimeSinceStartup + 3f;
                while (receiver.LatestPacket != null)
                {
                    Assert.That(Time.realtimeSinceStartup, Is.LessThan(expiryDeadline));
                    yield return null;
                }
                yield return null;
                Assert.That(driver.AppliedFrameId, Is.EqualTo(140));
                Assert.That(receiver.AcceptedPackets, Is.EqualTo(8));
                foreach (var pair in finalRotations)
                    Assert.That(Quaternion.Angle(pair.Key.localRotation, pair.Value), Is.LessThan(.01f));
                Assert.That(receiver.LastError, Is.Null);
                container.SetActive(false);
                Assert.That(receiver.IsReceiving, Is.False);
                Assert.That(receiver.LatestPacket, Is.Null);
            }
            finally
            {
                Object.DestroyImmediate(container);
                Object.DestroyImmediate(avatar);
            }
        }

        [UnityTearDown]
        public IEnumerator LeavePlayMode()
        {
            if (Application.isPlaying) yield return new ExitPlayMode();
        }

        [UnityTest, Category("ExternalReplay")]
        public IEnumerator PythonRecordingReplayReachesNativeAvatar()
        {
            string recording = System.Environment.GetEnvironmentVariable("POSEAHEAD_UNITY_REPLAY_RECORDING");
            if (string.IsNullOrEmpty(recording))
                Assert.Ignore("Set POSEAHEAD_UNITY_REPLAY_RECORDING to opt into a local Python/NPZ integration run.");
            yield return new EnterPlayMode();
            // EnterPlayMode reloads the domain: reconstruct local paths afterward.
            recording = System.Environment.GetEnvironmentVariable("POSEAHEAD_UNITY_REPLAY_RECORDING");
            string repository = System.IO.Path.GetFullPath(System.IO.Path.Combine(Application.dataPath, "../../.."));
            string python = System.IO.Path.Combine(repository, ".venv/Scripts/python.exe");
            Assert.That(System.IO.File.Exists(python), Is.True, "Local Python environment is required.");
            Assert.That(System.IO.File.Exists(recording), Is.True, "Recording must already exist; no download by this test.");
            var mannequin = DiagnosticSceneBuilder.CreateHumanoid();
            var animator = mannequin.GetComponent<Animator>();
            Avatar avatar = animator.avatar;
            var container = new GameObject("PythonReplayTest");
            container.SetActive(false);
            mannequin.transform.SetParent(container.transform, false);
            var receiver = container.AddComponent<PoseUdpReceiver>();
            typeof(PoseUdpReceiver).GetField("port", BindingFlags.NonPublic | BindingFlags.Instance)
                .SetValue(receiver, 0);
            var rootDriver = container.AddComponent<PoseRootDriver>();
            rootDriver.Configure(receiver, container.transform);
            var driver = mannequin.AddComponent<PoseHumanoidDriver>();
            driver.Configure(receiver, animator, container.transform);
            var telemetry = container.AddComponent<PoseDemoTelemetry>();
            telemetry.Configure(receiver, driver, rootDriver);
            System.Diagnostics.Process process = null;
            try
            {
                mannequin.SetActive(true);
                container.SetActive(true);
                yield return null;
                Assert.That(driver.IsCalibrated, Is.True, driver.LastError);
                var socket = (UdpClient)typeof(PoseUdpReceiver)
                    .GetField("client", BindingFlags.NonPublic | BindingFlags.Instance).GetValue(receiver);
                int port = ((IPEndPoint)socket.Client.LocalEndPoint).Port;
                string output = System.IO.Path.Combine(repository, "outputs", "unity_worker_" +
                    System.Guid.NewGuid().ToString("N"));
                System.IO.Directory.CreateDirectory(output);
                string optionsPath = System.IO.Path.Combine(output, "worker_options.json");
                System.IO.File.WriteAllText(optionsPath, JsonUtility.ToJson(new WorkerRequest {
                    recording = recording, udp_port = port }));
                process = System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo {
                    FileName = python,
                    Arguments = "-m poseahead.desktop.worker --repository \"" + repository +
                        "\" --output \"" + output + "\" --options-file \"" + optionsPath + "\"",
                    WorkingDirectory = repository, UseShellExecute = false, CreateNoWindow = true,
                    RedirectStandardOutput = true, RedirectStandardError = true });
                Assert.That(process, Is.Not.Null);
                // Drain both redirected pipes while Unity keeps running. Progress can exceed
                // the Windows pipe buffer before replay finishes; reading only after exit hangs.
                var stdoutTask = process.StandardOutput.ReadToEndAsync();
                var stderrTask = process.StandardError.ReadToEndAsync();
                var observedFrames = new HashSet<ulong>();
                bool poseChanged = false;
                float deadline = Time.realtimeSinceStartup + 30f;
                while (!process.HasExited)
                {
                    Assert.That(Time.realtimeSinceStartup, Is.LessThan(deadline), "Python replay timed out.");
                    yield return null;
                    var packet = receiver.LatestPacket;
                    if (packet == null || driver.AppliedFrameId != packet.FrameId || driver.AppliedBoneCount == 0)
                        continue;
                    observedFrames.Add(packet.FrameId);
                    poseChanged |= Quaternion.Angle(animator.GetBoneTransform(HumanBodyBones.LeftUpperArm)
                        .localRotation, Quaternion.identity) > .1f;
                    foreach (var segment in Segments)
                    {
                        // Full torso frames orthogonalize lateral/up axes; CMU need not be exactly orthogonal.
                        if (segment.start == HumanBodyBones.Hips || segment.start == HumanBodyBones.Chest ||
                            !packet.JointValidity[segment.a] || !packet.JointValidity[segment.b] ||
                            packet.JointConfidence[segment.a] < .1f || packet.JointConfidence[segment.b] < .1f)
                            continue;
                        Vector3 expected = Point(packet, segment.b) - Point(packet, segment.a);
                        if (expected.magnitude < .001f) continue;
                        Vector3 actual = animator.GetBoneTransform(segment.end).position -
                                         animator.GetBoneTransform(segment.start).position;
                        Assert.That(Vector3.Distance(actual.normalized, expected.normalized), Is.LessThan(1e-4f),
                                    "Python recording segment " + segment.start);
                    }
                }
                while (!stdoutTask.IsCompleted || !stderrTask.IsCompleted)
                {
                    Assert.That(Time.realtimeSinceStartup, Is.LessThan(deadline), "Python pipes did not close.");
                    yield return null;
                }
                string stdout = stdoutTask.GetAwaiter().GetResult();
                string stderr = stderrTask.GetAwaiter().GetResult();
                Assert.That(process.ExitCode, Is.Zero, stderr);
                var report = JsonUtility.FromJson<PythonReplayReport>(System.IO.File.ReadAllText(
                    System.IO.Path.Combine(output, "report.json")));
                string combinedJson = System.IO.File.ReadAllText(System.IO.Path.Combine(output, "run_report.json"));
                var combined = JsonUtility.FromJson<CombinedReport>(combinedJson);
                Assert.That(combined.unity.session_id, Is.EqualTo(report.sender_session_id), stdout);
                Assert.That(combined.unity.received_pose_frames, Is.EqualTo(report.sent_frames));
                Assert.That(combined.unity.applied_pose_frames, Is.GreaterThan(1));
                Assert.That(combined.unity.target_frame_rate, Is.EqualTo(60));
                Assert.That(combined.unity.v_sync_count, Is.Zero);
                Assert.That(combined.unity.demo_frame_limit_active, Is.True);
                Assert.That(combined.unity.headless, Is.True, "Native runner must not count as render acceptance.");
                Assert.That(combined.unity.working_set_available, Is.True, "Windows native resident-memory API failed.");
                Assert.That(combined.unity.working_set_mib, Is.GreaterThan(0));
                Assert.That(combinedJson, Does.Contain("\"render_fps\": null"));
                bool sawConfirmedProgress = false;
                const string eventPrefix = "POSEAHEAD_EVENT ";
                foreach (string line in stdout.Split('\n'))
                {
                    if (!line.StartsWith(eventPrefix)) continue;
                    var progress = JsonUtility.FromJson<WorkerProgress>(line.Substring(eventPrefix.Length));
                    if (progress.kind != "progress" || progress.unity == null ||
                        progress.unity.status != "observed") continue;
                    Assert.That(progress.sender_session_id, Is.EqualTo(report.sender_session_id));
                    Assert.That(progress.unity.session_id, Is.EqualTo(report.sender_session_id));
                    Assert.That(progress.unity.udp_port, Is.EqualTo(port));
                    Assert.That(progress.unity.received_pose_frames, Is.LessThanOrEqualTo(progress.sent_frames));
                    sawConfirmedProgress = true;
                }
                Assert.That(sawConfirmedProgress, Is.True, "No matched Unity receipt before replay finished.");
                Assert.That(report.sent_frames, Is.GreaterThan(1));
                yield return null;
                Assert.That(receiver.AcceptedPackets, Is.EqualTo(report.sent_frames));
                Assert.That(receiver.RejectedPackets, Is.Zero);
                Assert.That(observedFrames.Count, Is.GreaterThan(1));
                Assert.That(poseChanged, Is.True, "Rig never moved from the rest pose.");
                Assert.That(driver.AppliedBoneCount, Is.EqualTo(12));
                Assert.That(receiver.LastError, Is.Null);
                Debug.Log("PoseAhead Python -> native Unity: accepted=" + receiver.AcceptedPackets +
                          ", observed frames=" + observedFrames.Count + ", applied frame=" + driver.AppliedFrameId);
            }
            finally
            {
                // Only the child started by this test may be stopped on failure.
                if (process != null)
                {
                    if (!process.HasExited) process.Kill();
                    process.Dispose();
                }
                Object.DestroyImmediate(container);
                Object.DestroyImmediate(avatar);
            }
        }
    }
}
