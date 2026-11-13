using System;
using System.Collections.Generic;
using PoseAhead.Transport;
using UnityEngine;
using NQuaternion = System.Numerics.Quaternion;
using NVector3 = System.Numerics.Vector3;

namespace PoseAhead.Retargeting
{
    // Calibrate once in the avatar's imported T-pose, without an animation controller.
    // Applies world orientations parent-first; never edits bone lengths/local positions.
    public sealed class PoseHumanoidDriver : MonoBehaviour
    {
        [SerializeField] private PoseUdpReceiver receiver = null;
        [SerializeField] private Animator animator = null;
        [SerializeField] private Transform poseSpace = null;
        [SerializeField] private float minimumConfidence = 0.1f;
        [SerializeField] private float minimumBoneLengthMeters = 0.001f;
        private readonly List<Binding> bindings = new List<Binding>();
        private ulong session, frame;
        private bool hasFrame;

        public bool IsCalibrated { get; private set; }
        public string LastError { get; private set; }
        public int AppliedBoneCount { get; private set; }
        public ulong AppliedFrameId { get; private set; }
        public ulong AppliedSessionId { get; private set; }
        public int AppliedPoseFrames { get; private set; }
        public int BoundBoneCount => bindings.Count;

        private sealed class Binding
        {
            public Transform Bone;
            public int Start, End;
            public int Left = -1, Right = -1;
            public NVector3 RestDirection, RestAxis;
            public NQuaternion RestRotation, RestLocalRotation, RestFrame;
        }

        private Transform Space => poseSpace != null ? poseSpace : transform;
        private static NVector3 Numeric(Vector3 v) => new NVector3(v.x, v.y, v.z);
        private static NQuaternion Numeric(Quaternion q) => new NQuaternion(q.x, q.y, q.z, q.w);
        private static Quaternion Unity(NQuaternion q) => new Quaternion(q.X, q.Y, q.Z, q.W);

        public void Configure(PoseUdpReceiver source, Animator avatar, Transform space)
        {
            receiver = source; animator = avatar; poseSpace = space;
            bindings.Clear(); IsCalibrated = false; hasFrame = false;
        }

        private void Start()
        {
            if (IsCalibrated) return;
            try { CalibrateRestPose(); }
            catch (Exception error)
            {
                LastError = error.Message;
                Debug.LogError("PoseAhead humanoid calibration: " + LastError, this);
                enabled = false;
            }
        }

        public void CalibrateRestPose()
        {
            IsCalibrated = false;
            bindings.Clear(); hasFrame = false; AppliedBoneCount = 0;
            if (receiver == null || animator == null || animator.avatar == null ||
                !animator.avatar.isValid || !animator.avatar.isHuman)
                throw new InvalidOperationException("Assign a receiver and a valid Humanoid Animator.");
            if (animator.runtimeAnimatorController != null)
                throw new InvalidOperationException("Remove the Animator Controller; calibrate in T-pose.");
            if (float.IsNaN(minimumConfidence) || float.IsInfinity(minimumConfidence) ||
                minimumConfidence < 0 || minimumConfidence > 1 ||
                float.IsNaN(minimumBoneLengthMeters) || float.IsInfinity(minimumBoneLengthMeters) ||
                minimumBoneLengthMeters <= 0)
                throw new InvalidOperationException("Invalid retarget confidence or length threshold.");
            Vector3 scale = Space.lossyScale;
            if (!PoseRetargetMath.IsFinite(Numeric(scale)) || scale.x <= 0 ||
                Math.Abs(scale.x - scale.y) > 1e-4f ||
                Math.Abs(scale.x - scale.z) > 1e-4f)
                throw new InvalidOperationException("Pose space needs positive uniform scale.");

            // Full pelvis/chest frames recover yaw/roll from two independent axes.
            Add(HumanBodyBones.Hips, HumanBodyBones.Spine, 0, 1,
                HumanBodyBones.LeftUpperLeg, HumanBodyBones.RightUpperLeg, 11, 14);
            Add(HumanBodyBones.Spine, HumanBodyBones.Chest, 1, 2);
            Add(HumanBodyBones.Chest, HumanBodyBones.Neck, 2, 3,
                HumanBodyBones.LeftUpperArm, HumanBodyBones.RightUpperArm, 5, 8);
            Add(HumanBodyBones.Neck, HumanBodyBones.Head, 3, 4);
            Add(HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm, 5, 6);
            Add(HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftHand, 6, 7);
            Add(HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm, 8, 9);
            Add(HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand, 9, 10);
            Add(HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg, 11, 12);
            Add(HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot, 12, 13);
            Add(HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg, 14, 15);
            Add(HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot, 15, 16);
            animator.applyRootMotion = false;
            IsCalibrated = true; LastError = null;
        }

        private void Add(HumanBodyBones boneId, HumanBodyBones childId, int start, int end,
                         HumanBodyBones leftId = HumanBodyBones.LastBone,
                         HumanBodyBones rightId = HumanBodyBones.LastBone,
                         int left = -1, int right = -1)
        {
            Transform bone = animator.GetBoneTransform(boneId);
            Transform child = animator.GetBoneTransform(childId);
            // Spine/chest/neck can be optional in imported Humanoids. Fallback end
            // points keep torso chains usable without fabricating an absent bone.
            if (bone == null) return;
            if (child == null && childId == HumanBodyBones.Chest)
                child = animator.GetBoneTransform(HumanBodyBones.Neck) ??
                        animator.GetBoneTransform(HumanBodyBones.Head);
            if (child == null && childId == HumanBodyBones.Neck)
                child = animator.GetBoneTransform(HumanBodyBones.Head);
            if (child == null) throw new InvalidOperationException("Missing end bone for " + boneId);
            NQuaternion inverse = NQuaternion.Inverse(Numeric(Space.rotation));
            var binding = new Binding { Bone = bone, Start = start, End = end,
                RestRotation = inverse * Numeric(bone.rotation),
                RestLocalRotation = Numeric(bone.localRotation) };
            if (!PoseRetargetMath.TryUnit(NVector3.Transform(Numeric(child.position - bone.position),
                                                           inverse), 1e-6f,
                                         out binding.RestDirection))
                throw new InvalidOperationException("Zero-length rest bone: " + boneId);
            binding.RestAxis = NVector3.Transform(binding.RestDirection,
                                                 NQuaternion.Inverse(binding.RestRotation));
            if (left >= 0)
            {
                var l = animator.GetBoneTransform(leftId);
                var r = animator.GetBoneTransform(rightId);
                if (l == null || r == null || !PoseRetargetMath.TryBodyFrame(
                    NVector3.Transform(Numeric(r.position - l.position), inverse),
                    binding.RestDirection, 1e-6f, out binding.RestFrame))
                    throw new InvalidOperationException("Degenerate rest body frame: " + boneId);
                binding.Left = left; binding.Right = right;
            }
            bindings.Add(binding);
        }

        private void LateUpdate() { ApplyPacket(receiver != null ? receiver.LatestPacket : null); }

        public bool ApplyPacket(PosePacket packet)
        {
            if (!IsCalibrated || packet == null || (hasFrame && (packet.SessionId < session ||
                (packet.SessionId == session && packet.FrameId <= frame)))) return false;
            session = packet.SessionId; frame = packet.FrameId; hasFrame = true;
            AppliedBoneCount = 0;
            NQuaternion spaceRotation = Numeric(Space.rotation);
            foreach (Binding binding in bindings)
            {
                NQuaternion rotation;
                if (binding.Left >= 0)
                {
                    if (!PoseRetargetMath.TryBodyFrame(packet, binding.Left, binding.Right,
                        binding.Start, binding.End, minimumConfidence, minimumBoneLengthMeters,
                        out var bodyFrame)) continue;
                    rotation = bodyFrame * NQuaternion.Inverse(binding.RestFrame) * binding.RestRotation;
                }
                else
                {
                    if (!PoseRetargetMath.TryDirection(packet, binding.Start, binding.End,
                        minimumConfidence, minimumBoneLengthMeters, out var direction)) continue;
                    // Inherit the current parent's orientation, retain rest local
                    // twist, then swing only the observable segment axis. No
                    // incremental integration from the bone's previous rotation.
                    var baseline = binding.Bone.parent != null ?
                        NQuaternion.Inverse(spaceRotation) * Numeric(binding.Bone.parent.rotation) *
                        binding.RestLocalRotation : binding.RestRotation;
                    var axis = NVector3.Transform(binding.RestAxis, baseline);
                    rotation = PoseRetargetMath.Swing(axis, direction) * baseline;
                }
                binding.Bone.rotation = Space.rotation * Unity(NQuaternion.Normalize(rotation));
                AppliedBoneCount++;
            }
            if (AppliedSessionId != session) AppliedPoseFrames = 0;
            AppliedSessionId = session;
            if (AppliedBoneCount > 0) AppliedPoseFrames++;
            AppliedFrameId = frame;
            return true;
        }
    }
}
