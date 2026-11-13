using System;
using NUnit.Framework;
using PoseAhead.Editor;
using PoseAhead.Retargeting;
using PoseAhead.Transport;
using UnityEngine;

namespace PoseAhead.Tests
{
    public sealed class DiagnosticHumanoidTests
    {
        [Test]
        public void DemoFrameRateLimitRestoresSettingsAndHandlesOverlap()
        {
            int target = Application.targetFrameRate, vSync = QualitySettings.vSyncCount;
            try
            {
                Application.targetFrameRate = 144; QualitySettings.vSyncCount = 2;
                using (var first = new DemoFrameRateScope())
                using (var second = new DemoFrameRateScope())
                {
                    Assert.That(Application.targetFrameRate, Is.EqualTo(60));
                    Assert.That(QualitySettings.vSyncCount, Is.Zero);
                    first.Dispose(); first.Dispose();
                    Assert.That(Application.targetFrameRate, Is.EqualTo(60));
                }
                Assert.That(Application.targetFrameRate, Is.EqualTo(144));
                Assert.That(QualitySettings.vSyncCount, Is.EqualTo(2));
                using (var scope = new DemoFrameRateScope())
                {
                    Application.targetFrameRate = 120; QualitySettings.vSyncCount = 1;
                }
                Assert.That(Application.targetFrameRate, Is.EqualTo(120));
                Assert.That(QualitySettings.vSyncCount, Is.EqualTo(1));
            }
            finally
            { Application.targetFrameRate = target; QualitySettings.vSyncCount = vSync; }
        }

        [Test]
        public void BuiltHumanoidCalibratesAndNumericRotationsAgreeWithUnity()
        {
            var root = DiagnosticSceneBuilder.CreateHumanoid();
            Avatar avatar = root.GetComponent<Animator>().avatar;
            var sourceObject = new GameObject("DisabledTestReceiver");
            sourceObject.SetActive(false); // No socket opened by this EditMode test.
            try
            {
                var driver = root.AddComponent<PoseHumanoidDriver>();
                var receiver = sourceObject.AddComponent<PoseUdpReceiver>();
                root.SetActive(true);
                var animator = root.GetComponent<Animator>();
                Assert.That(avatar.isValid && avatar.isHuman);
                driver.Configure(receiver, animator, root.transform);
                driver.CalibrateRestPose();
                Assert.That(driver.BoundBoneCount, Is.EqualTo(12));
                var swing = PoseRetargetMath.Swing(System.Numerics.Vector3.UnitX, System.Numerics.Vector3.UnitY);
                Vector3 direction = new Quaternion(swing.X, swing.Y, swing.Z, swing.W) * Vector3.right;
                Assert.That(Vector3.Distance(direction, Vector3.up), Is.LessThan(1e-5f));
                // Decode a real Python packet fixture; do not bypass internal packet setters.
                string fixture = System.IO.Path.Combine(Application.dataPath, "PoseAhead/Tests/Fixtures/retarget_tpose.base64");
                Assert.That(PosePacketDecoder.TryDecode(Convert.FromBase64String(System.IO.File.ReadAllText(fixture)),
                                                       out var packet, out var error), Is.True, error);
                Assert.That(driver.ApplyPacket(packet), Is.True);
                Assert.That(driver.AppliedBoneCount, Is.EqualTo(12));
                var arm = animator.GetBoneTransform(HumanBodyBones.LeftUpperArm);
                var elbow = animator.GetBoneTransform(HumanBodyBones.LeftLowerArm);
                Assert.That(Vector3.Distance((elbow.position - arm.position).normalized, Vector3.right),
                            Is.LessThan(1e-4f));
                Assert.That(driver.ApplyPacket(packet), Is.False); // Duplicate does not drift.
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
                UnityEngine.Object.DestroyImmediate(sourceObject);
                UnityEngine.Object.DestroyImmediate(avatar);
            }
        }
    }
}
