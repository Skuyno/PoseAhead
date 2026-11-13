using System;
using System.Numerics;
using PoseAhead.Retargeting;
using PoseAhead.Transport;
using U = UnityEngine;

internal static class RetargetTests
{
    private static void Check(bool value, string error)
    { if (!value) throw new InvalidOperationException(error); }

    private static void Near(Vector3 a, Vector3 b, string error)
    { Check(Vector3.Distance(a, b) < 1e-4f, error + ": " + a + " != " + b); }

    private static PosePacket Packet()
    {
        var p = new PosePacket { SessionId = 1, FrameId = 0, JointPositions = new float[51],
            JointValidity = new bool[17], JointConfidence = new float[17] };
        for (int i = 0; i < 17; i++) { p.JointValidity[i] = true; p.JointConfidence[i] = 1; }
        return p;
    }

    private static void Point(PosePacket p, int joint, Vector3 display)
    { p.JointPositions[joint * 3] = display.X; p.JointPositions[joint * 3 + 1] = -display.Y;
      p.JointPositions[joint * 3 + 2] = display.Z; }

    internal static void Run()
    {
        var p = Packet();
        Point(p, 5, new Vector3(1, 2, 3)); Point(p, 6, new Vector3(1, 3, 3));
        Check(PoseRetargetMath.TryDirection(p, 5, 6, .1f, .001f, out var d), "Valid direction rejected.");
        Near(d, Vector3.UnitY, "Camera Y reflection differs");
        p.JointValidity[6] = false;
        Check(!PoseRetargetMath.TryDirection(p, 5, 6, .1f, .001f, out _), "Invalid joint accepted.");
        p.JointValidity[6] = true; p.JointConfidence[6] = .09f;
        Check(!PoseRetargetMath.TryDirection(p, 5, 6, .1f, .001f, out _), "Low confidence accepted.");
        p.JointConfidence[6] = 0;
        Check(!PoseRetargetMath.TryDirection(p, 5, 6, 0, .001f, out _), "Zero confidence accepted.");
        p.JointConfidence[6] = 1; Point(p, 6, new Vector3(1, 2, 3));
        Check(!PoseRetargetMath.TryDirection(p, 5, 6, .1f, .001f, out _), "Zero-length segment accepted.");
        Point(p, 6, new Vector3(float.NaN, 0, 0));
        Check(!PoseRetargetMath.TryDirection(p, 5, 6, .1f, .001f, out _), "Nonfinite point accepted.");
        Check(!PoseRetargetMath.TryBodyFrame(Vector3.UnitY, Vector3.UnitY, .001f, out _),
              "Collinear body frame accepted.");
        var yaw = Quaternion.CreateFromAxisAngle(Vector3.UnitY, .7f);
        var right = Vector3.Transform(Vector3.UnitX, yaw);
        Check(PoseRetargetMath.TryBodyFrame(right, Vector3.UnitY, .001f, out var body),
              "Valid body frame rejected.");
        Near(Vector3.Transform(Vector3.UnitX, body), right, "Body right axis");
        Near(Vector3.Transform(Vector3.UnitZ, body), Vector3.Transform(Vector3.UnitZ, yaw), "Body facing");
        foreach (var from in new[] { Vector3.UnitX, Vector3.UnitY, Vector3.UnitZ })
            foreach (var to in new[] { from, -from, Vector3.Normalize(new Vector3(.2f, -.3f, .8f)) })
                Near(Vector3.Transform(from, PoseRetargetMath.Swing(from, to)), to, "Swing direction");
        Driver();
    }

    private static void Driver()
    {
        var avatar = new U.Animator();
        U.Transform Bone(U.HumanBodyBones id, U.HumanBodyBones? parent, Vector3 local)
        {
            var t = new U.Transform { localPosition = new U.Vector3(local.X, local.Y, local.Z),
                parent = parent.HasValue ? avatar.Bones[parent.Value] : null };
            avatar.Bones.Add(id, t); return t;
        }
        var hips = Bone(U.HumanBodyBones.Hips, null, new Vector3(0, 1, 0));
        Bone(U.HumanBodyBones.Spine, U.HumanBodyBones.Hips, new Vector3(0, .2f, 0));
        Bone(U.HumanBodyBones.Chest, U.HumanBodyBones.Spine, new Vector3(0, .2f, 0));
        Bone(U.HumanBodyBones.Neck, U.HumanBodyBones.Chest, new Vector3(0, .2f, 0));
        Bone(U.HumanBodyBones.Head, U.HumanBodyBones.Neck, new Vector3(0, .15f, 0));
        Bone(U.HumanBodyBones.LeftUpperArm, U.HumanBodyBones.Chest, new Vector3(.2f, .1f, 0));
        Bone(U.HumanBodyBones.LeftLowerArm, U.HumanBodyBones.LeftUpperArm, new Vector3(.3f, 0, 0));
        Bone(U.HumanBodyBones.LeftHand, U.HumanBodyBones.LeftLowerArm, new Vector3(.25f, 0, 0));
        Bone(U.HumanBodyBones.RightUpperArm, U.HumanBodyBones.Chest, new Vector3(-.2f, .1f, 0));
        Bone(U.HumanBodyBones.RightLowerArm, U.HumanBodyBones.RightUpperArm, new Vector3(-.3f, 0, 0));
        Bone(U.HumanBodyBones.RightHand, U.HumanBodyBones.RightLowerArm, new Vector3(-.25f, 0, 0));
        Bone(U.HumanBodyBones.LeftUpperLeg, U.HumanBodyBones.Hips, new Vector3(.1f, 0, 0));
        Bone(U.HumanBodyBones.LeftLowerLeg, U.HumanBodyBones.LeftUpperLeg, new Vector3(0, -.4f, 0));
        Bone(U.HumanBodyBones.LeftFoot, U.HumanBodyBones.LeftLowerLeg, new Vector3(0, -.4f, 0));
        Bone(U.HumanBodyBones.RightUpperLeg, U.HumanBodyBones.Hips, new Vector3(-.1f, 0, 0));
        Bone(U.HumanBodyBones.RightLowerLeg, U.HumanBodyBones.RightUpperLeg, new Vector3(0, -.4f, 0));
        Bone(U.HumanBodyBones.RightFoot, U.HumanBodyBones.RightLowerLeg, new Vector3(0, -.4f, 0));
        var driver = new PoseHumanoidDriver();
        driver.Configure(new PoseUdpReceiver(), avatar, new U.Transform());
        driver.CalibrateRestPose();
        Check(driver.IsCalibrated && driver.BoundBoneCount == 12, "Humanoid binding incomplete.");
        var packet = Packet();
        var canonical = new[] { U.HumanBodyBones.Hips, U.HumanBodyBones.Spine,
            U.HumanBodyBones.Chest, U.HumanBodyBones.Neck, U.HumanBodyBones.Head,
            U.HumanBodyBones.LeftUpperArm, U.HumanBodyBones.LeftLowerArm, U.HumanBodyBones.LeftHand,
            U.HumanBodyBones.RightUpperArm, U.HumanBodyBones.RightLowerArm, U.HumanBodyBones.RightHand,
            U.HumanBodyBones.LeftUpperLeg, U.HumanBodyBones.LeftLowerLeg, U.HumanBodyBones.LeftFoot,
            U.HumanBodyBones.RightUpperLeg, U.HumanBodyBones.RightLowerLeg, U.HumanBodyBones.RightFoot };
        for (int joint = 0; joint < 17; joint++)
        {
            var t = avatar.Bones[canonical[joint]];
            Point(packet, joint, new Vector3(t.position.x - hips.position.x,
                t.position.y - hips.position.y, t.position.z - hips.position.z));
        }
        Check(driver.ApplyPacket(packet) && driver.AppliedBoneCount == 12, "T-pose not applied.");
        Check(driver.AppliedSessionId == packet.SessionId && driver.AppliedPoseFrames == 1,
              "Applied-pose telemetry differs from the first frame.");
        Check(!driver.ApplyPacket(packet), "Duplicate applied.");
        Check(driver.AppliedPoseFrames == 1, "Duplicate incremented applied telemetry.");
        var original = (float[])packet.JointPositions.Clone();
        var turn = Quaternion.CreateFromAxisAngle(Vector3.UnitY, (float)Math.PI / 2);
        packet.FrameId++;
        for (int joint = 0; joint < 17; joint++)
            Point(packet, joint, Vector3.Transform(new Vector3(original[joint * 3],
                -original[joint * 3 + 1], original[joint * 3 + 2]), turn));
        driver.ApplyPacket(packet);
        var legRotation = avatar.Bones[U.HumanBodyBones.LeftLowerLeg].rotation;
        Near(Vector3.Transform(Vector3.UnitX,
             new Quaternion(legRotation.x, legRotation.y, legRotation.z, legRotation.w)),
             -Vector3.UnitZ, "Leg rest twist did not inherit body yaw");
        packet.FrameId++; packet.JointPositions = original;
        driver.ApplyPacket(packet);
        // Keep all lengths: raise left arm, bend elbow toward +Z.
        packet.FrameId++;
        Point(packet, 6, new Vector3(.2f, .8f, 0));
        Point(packet, 7, new Vector3(.2f, .8f, .25f));
        driver.ApplyPacket(packet);
        Vector3 Segment(U.HumanBodyBones a, U.HumanBodyBones b)
        {
            var x = avatar.Bones[a].position; var y = avatar.Bones[b].position;
            return Vector3.Normalize(new Vector3(y.x - x.x, y.y - x.y, y.z - x.z));
        }
        Near(Segment(U.HumanBodyBones.LeftUpperArm, U.HumanBodyBones.LeftLowerArm),
             Vector3.UnitY, "Upper arm wrong");
        Near(Segment(U.HumanBodyBones.LeftLowerArm, U.HumanBodyBones.LeftHand),
             Vector3.UnitZ, "Lower arm wrong after parent rotation");
        var local = avatar.Bones[U.HumanBodyBones.LeftLowerArm].localRotation;
        packet.FrameId++; packet.JointValidity[7] = false;
        Point(packet, 6, new Vector3(.2f, .5f, .3f));
        driver.ApplyPacket(packet);
        var held = avatar.Bones[U.HumanBodyBones.LeftLowerArm].localRotation;
        Check(Math.Abs(local.x - held.x) < 1e-5 && Math.Abs(local.w - held.w) < 1e-5,
              "Invalid descendant did not hold local rotation.");
        Check(!driver.ApplyPacket(null), "Expired packet applied.");
        packet.SessionId = 2; packet.FrameId = 0;
        Check(driver.ApplyPacket(packet), "New sender session rejected.");
        Check(driver.AppliedSessionId == packet.SessionId && driver.AppliedPoseFrames == 1,
              "Applied-pose telemetry did not reset on sender restart.");
        packet.SessionId = 1; packet.FrameId = 1000;
        Check(!driver.ApplyPacket(packet), "Old sender session accepted.");
        var lowerArm = avatar.Bones[U.HumanBodyBones.LeftLowerArm].localPosition;
        Check(lowerArm.x == .3f && lowerArm.y == 0 && lowerArm.z == 0,
              "Retargeting changed a calibrated bone length.");
        avatar.runtimeAnimatorController = new object();
        bool rejected = false;
        try { driver.CalibrateRestPose(); } catch (InvalidOperationException) { rejected = true; }
        Check(rejected && !driver.IsCalibrated, "Animated non-rest rig accepted.");
    }
}
