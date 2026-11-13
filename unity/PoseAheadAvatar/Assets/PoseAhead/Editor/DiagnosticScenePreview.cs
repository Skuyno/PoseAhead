using System;
using System.IO;
using PoseAhead.Retargeting;
using PoseAhead.Transport;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace PoseAhead.Editor
{
    // Batch entry point. Generates a saved scene, then renders unsaved diagnostic poses.
    // PNGs prove native geometry/rendering, not camera tracking or live latency.
    public static class DiagnosticScenePreview
    {
        [Serializable]
        private sealed class PhaseReport
        {
            public string name;
            public int phase;
            public ulong frameId;
            public int appliedBones;
            public Vector3 containerPosition;
            public string image;
        }

        [Serializable]
        private sealed class PreviewReport
        {
            public string unityVersion;
            public string scene;
            public string graphicsDevice;
            public string scope = "Native offline diagnostic fixture rendering; not live-camera acceptance.";
            public PhaseReport[] phases;
        }

        public static void Run()
        {
            string[] args = Environment.GetCommandLineArgs();
            int outputIndex = Array.IndexOf(args, "-poseAheadOutput");
            if (outputIndex < 0 || outputIndex + 1 >= args.Length)
                throw new ArgumentException("Pass -poseAheadOutput with an empty output directory.");
            string output = Path.GetFullPath(args[outputIndex + 1]);
            Directory.CreateDirectory(output);
            if (Directory.GetFiles(output, "phase_*.png").Length != 0 || File.Exists(Path.Combine(output, "preview.json")))
                throw new IOException("Preview artifacts already exist; choose a new output directory.");
            if (SystemInfo.graphicsDeviceType == UnityEngine.Rendering.GraphicsDeviceType.Null)
                throw new InvalidOperationException("Rendering requires a graphics device; omit -nographics.");

            var scene = DiagnosticSceneBuilder.BuildScene();
            try
            {
                Camera camera = null;
                PoseHumanoidDriver driver = null;
                PoseRootDriver rootDriver = null;
                foreach (var root in scene.GetRootGameObjects())
                {
                    if (camera == null) camera = root.GetComponentInChildren<Camera>();
                    if (driver == null) driver = root.GetComponentInChildren<PoseHumanoidDriver>();
                    if (rootDriver == null) rootDriver = root.GetComponent<PoseRootDriver>();
                }
                if (camera == null || driver == null || rootDriver == null)
                    throw new InvalidOperationException("Diagnostic scene components are missing.");
                driver.CalibrateRestPose();
                string[] names = { "tpose", "raised_arm", "bent_elbow", "raised_knee", "body_yaw",
                                   "root_translation", "missing_wrist", "restored_tpose" };
                byte[][] fixtures = DiagnosticPoseFixtures.LoadPackets();
                var report = new PreviewReport {
                    unityVersion = Application.unityVersion, scene = scene.path,
                    graphicsDevice = SystemInfo.graphicsDeviceName, phases = new PhaseReport[fixtures.Length] };
                for (int phase = 0; phase < fixtures.Length; phase++)
                {
                    if (!PosePacketDecoder.TryDecode(fixtures[phase], out var packet, out var error))
                        throw new InvalidDataException(error);
                    rootDriver.ApplyPacket(packet);
                    if (!driver.ApplyPacket(packet)) throw new InvalidOperationException("Fixture was not applied.");
                    string image = "phase_" + phase + "_" + names[phase] + ".png";
                    WriteNew(Path.Combine(output, image), Render(camera));
                    report.phases[phase] = new PhaseReport { name = names[phase], phase = phase,
                        frameId = packet.FrameId, appliedBones = driver.AppliedBoneCount,
                        containerPosition = rootDriver.transform.position, image = image };
                }
                WriteNew(Path.Combine(output, "preview.json"),
                         System.Text.Encoding.UTF8.GetBytes(JsonUtility.ToJson(report, true)));
                Debug.Log("PoseAhead: rendered eight diagnostic phases to " + output);
            }
            finally
            {
                // Do not save applied preview poses over the generated T-pose scene.
                if (SceneManager.sceneCount > 1) EditorSceneManager.CloseScene(scene, true);
                else EditorSceneManager.OpenScene(scene.path, OpenSceneMode.Single);
            }
        }

        private static byte[] Render(Camera camera)
        {
            var target = new RenderTexture(960, 720, 24, RenderTextureFormat.ARGB32);
            var texture = new Texture2D(960, 720, TextureFormat.RGB24, false);
            RenderTexture oldActive = RenderTexture.active;
            RenderTexture oldTarget = camera.targetTexture;
            float oldAspect = camera.aspect;
            try
            {
                target.Create();
                camera.targetTexture = target;
                camera.aspect = 960f / 720f;
                camera.Render();
                RenderTexture.active = target;
                texture.ReadPixels(new Rect(0, 0, 960, 720), 0, 0);
                texture.Apply();
                return texture.EncodeToPNG();
            }
            finally
            {
                camera.targetTexture = oldTarget;
                camera.aspect = oldAspect;
                RenderTexture.active = oldActive;
                target.Release();
                UnityEngine.Object.DestroyImmediate(target);
                UnityEngine.Object.DestroyImmediate(texture);
            }
        }

        private static void WriteNew(string path, byte[] data)
        {
            using (var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write))
                stream.Write(data, 0, data.Length);
        }
    }
}
