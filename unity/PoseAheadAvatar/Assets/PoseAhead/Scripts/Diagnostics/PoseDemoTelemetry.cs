using System;
using System.Diagnostics;
using System.IO;
using PoseAhead.Retargeting;
using PoseAhead.Transport;
using UnityEngine;

namespace PoseAhead
{
    // Local diagnostic evidence. Sender clocks are never subtracted from Unity time.
    public sealed class PoseDemoTelemetry : MonoBehaviour
    {
        [SerializeField] private PoseUdpReceiver receiver = null;
        [SerializeField] private PoseHumanoidDriver humanoid = null;
        [SerializeField] private PoseRootDriver root = null;
        private ulong measuredSession;
        private double sessionStarted, nextWrite;
        private int renderedFrames, lastRenderedFrame = -1;
        private string destination;
        private double? firstMemory;
        private double peakMemory;
        private DemoFrameRateScope frameRateScope;
        public string ExportError { get; private set; }
        private static double Now => (double)Stopwatch.GetTimestamp() / Stopwatch.Frequency;

        [Serializable]
        public sealed class Snapshot
        {
            public int schema_version = 1;
            public string session_id;
            public long written_unix_ms;
            public int udp_port, received_pose_frames, applied_pose_frames;
            public int rejected_packets_receiver_total, rendered_frames;
            public int target_frame_rate, v_sync_count;
            public bool demo_frame_limit_active;
            public double duration_s, working_set_mib, working_set_growth_mib, peak_working_set_mib;
            public bool receiving, headless, working_set_available;
            public string receiver_error, humanoid_error;
        }

        private void OnEnable()
        {
            if (Application.isPlaying && frameRateScope == null)
                frameRateScope = new DemoFrameRateScope();
            Camera.onPostRender += AfterRender;
        }
        private void OnDisable()
        {
            Camera.onPostRender -= AfterRender;
            frameRateScope?.Dispose();
            frameRateScope = null;
        }

        private void AfterRender(Camera camera)
        {
            // Only on-screen Game cameras in this scene, at most once per Unity frame.
            // Built-in pipeline only: SRP needs a separate adapter, not fabricated FPS.
            if (measuredSession == 0 || camera.cameraType != CameraType.Game ||
                camera.targetTexture != null || camera.gameObject.scene != gameObject.scene ||
                lastRenderedFrame == Time.frameCount) return;
            lastRenderedFrame = Time.frameCount;
            renderedFrames++;
        }

        private void Update()
        {
            if (receiver == null) return;
            receiver.SessionCounts(out ulong session, out _);
            if (session != 0 && session != measuredSession)
            {
                measuredSession = session; sessionStarted = Now;
                renderedFrames = 0; lastRenderedFrame = -1;
                firstMemory = null; peakMemory = 0;
            }
            if (session == 0 || Now < nextWrite) return;
            nextWrite = Now + .5;
            // This checkout's Temp directory is disposable, not a tracked scene/asset.
            if (destination == null)
                destination = Path.GetFullPath(Path.Combine(Application.dataPath,
                    "../Temp/PoseAheadTelemetry-" + receiver.ListenPort + ".json"));
            WriteSnapshot(destination);
        }

        public Snapshot CaptureSnapshot()
        {
            ulong session = 0; int accepted = 0;
            receiver?.SessionCounts(out session, out accepted);
            if (session != measuredSession)
            {
                measuredSession = session; sessionStarted = Now;
                renderedFrames = 0; lastRenderedFrame = -1;
                firstMemory = null; peakMemory = 0;
            }
            bool memoryAvailable = HostMemory.TryRead(out double memory);
            if (memoryAvailable)
            {
                if (!firstMemory.HasValue) firstMemory = memory;
                peakMemory = Math.Max(peakMemory, memory);
            }
            return new Snapshot {
                    session_id = session.ToString(System.Globalization.CultureInfo.InvariantCulture),
                    written_unix_ms = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                    udp_port = receiver?.ListenPort ?? 0, received_pose_frames = accepted,
                    applied_pose_frames = humanoid != null && humanoid.AppliedSessionId == session ?
                        humanoid.AppliedPoseFrames : 0,
                    rejected_packets_receiver_total = receiver?.RejectedPackets ?? 0,
                    rendered_frames = renderedFrames,
                    target_frame_rate = Application.targetFrameRate,
                    v_sync_count = QualitySettings.vSyncCount,
                    demo_frame_limit_active = frameRateScope != null &&
                        Application.targetFrameRate == DemoFrameRateScope.TargetFramesPerSecond &&
                        QualitySettings.vSyncCount == 0,
                    duration_s = measuredSession == session ? Math.Max(0, Now - sessionStarted) : 0,
                    working_set_mib = memory, working_set_available = memoryAvailable,
                    working_set_growth_mib = firstMemory.HasValue ? memory - firstMemory.Value : 0,
                    peak_working_set_mib = peakMemory,
                    receiving = receiver != null && receiver.IsReceiving,
                    headless = Application.isBatchMode ||
                        SystemInfo.graphicsDeviceType == UnityEngine.Rendering.GraphicsDeviceType.Null,
                    receiver_error = receiver?.LastError, humanoid_error = humanoid?.LastError,
                };
        }

        public void WriteSnapshot(string path)
        {
            // Atomic replacement: readers never accept half a JSON document.
            string temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path)));
                File.WriteAllText(temporary, JsonUtility.ToJson(CaptureSnapshot()));
                if (File.Exists(path)) File.Replace(temporary, path, null);
                else File.Move(temporary, path);
                ExportError = null;
            }
            catch (IOException error) { ExportError = error.Message; }
            catch (UnauthorizedAccessException error) { ExportError = error.Message; }
            finally
            {
                try { if (File.Exists(temporary)) File.Delete(temporary); }
                catch (IOException error) { ExportError = error.Message; }
                catch (UnauthorizedAccessException error) { ExportError = error.Message; }
            }
        }

        public void Configure(PoseUdpReceiver source, PoseHumanoidDriver driver, PoseRootDriver rootDriver)
        { receiver = source; humanoid = driver; root = rootDriver; }

        private void OnGUI()
        {
            var packet = receiver != null ? receiver.LatestPacket : null;
            string status = packet == null ? "No fresh pose; last bone rotations held" :
                $"Frame {packet.FrameId} | {packet.Algorithm} | target +{packet.TargetOffsetMs} ms\n" +
                $"Root: {(packet.RootPosition == null ? "missing" : packet.RootPositionEstimated ? "monocular estimate" : "provided")}";
            string error = receiver != null ? receiver.LastError : "Receiver not assigned";
            if (humanoid != null && humanoid.LastError != null) error = humanoid.LastError;
            GUI.Box(new Rect(10, 10, 550, 140), "PoseAhead diagnostic Humanoid");
            GUI.Label(new Rect(20, 35, 530, 90), status + "\n" +
                $"Receive: {(receiver != null && receiver.IsReceiving ? "running" : "stopped")} | " +
                $"accepted {receiver?.AcceptedPackets ?? 0}, rejected {receiver?.RejectedPackets ?? 0}\n" +
                $"Bones: {humanoid?.AppliedBoneCount ?? 0}/{humanoid?.BoundBoneCount ?? 0} | " +
                $"update FPS {(Time.unscaledDeltaTime > 0 ? 1 / Time.unscaledDeltaTime : 0):F1} " +
                $"| target {Application.targetFrameRate}" +
                (ExportError == null ? "" : "\nTelemetry file: " + ExportError) +
                (error == null ? "" : "\nError: " + error));
            if (root != null && GUI.Button(new Rect(20, 160, 160, 28), "Recenter translation")) root.Recenter();
        }
    }
}
