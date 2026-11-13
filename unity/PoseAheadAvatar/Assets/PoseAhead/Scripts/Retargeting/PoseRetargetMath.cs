using System;
using System.Numerics;
using PoseAhead.Transport;

namespace PoseAhead.Retargeting
{
    // No Unity dependency: conversion and analytic orientation, not a predictor/filter.
    public static class PoseRetargetMath
    {
        public static bool IsFinite(Vector3 value) =>
            Finite(value.X) && Finite(value.Y) && Finite(value.Z);

        private static bool Finite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);

        public static bool TryPoint(PosePacket packet, int joint, float minimumConfidence,
                                    out Vector3 point)
        {
            point = default;
            if (packet == null || joint < 0 || joint >= 17 ||
                packet.JointPositions == null || packet.JointPositions.Length != 51 ||
                packet.JointValidity == null || packet.JointValidity.Length != 17 ||
                packet.JointConfidence == null || packet.JointConfidence.Length != 17 ||
                !Finite(minimumConfidence) || minimumConfidence < 0 || minimumConfidence > 1 ||
                !packet.JointValidity[joint]) return false;
            float confidence = packet.JointConfidence[joint];
            if (!Finite(confidence) || confidence <= 0 || confidence < minimumConfidence) return false;
            int offset = joint * 3;
            point = new Vector3(packet.JointPositions[offset], -packet.JointPositions[offset + 1],
                                packet.JointPositions[offset + 2]);
            return IsFinite(point);
        }

        public static bool TryDirection(PosePacket packet, int start, int end,
                                        float minimumConfidence, float minimumLength,
                                        out Vector3 direction)
        {
            direction = default;
            if (!TryPoint(packet, start, minimumConfidence, out var a) ||
                !TryPoint(packet, end, minimumConfidence, out var b)) return false;
            return TryUnit(b - a, minimumLength, out direction);
        }

        public static bool TryUnit(Vector3 value, float minimumLength, out Vector3 unit)
        {
            unit = default;
            if (!IsFinite(value) || !Finite(minimumLength) || minimumLength <= 0) return false;
            float length = value.Length();
            if (!Finite(length) || length < minimumLength) return false;
            unit = value / length;
            return true;
        }

        // Body frame: anatomical right, up, cross(right, up). No guessed camera yaw.
        public static bool TryBodyFrame(Vector3 right, Vector3 up, float minimumLength,
                                        out Quaternion rotation)
        {
            rotation = Quaternion.Identity;
            if (!TryUnit(up, minimumLength, out var y)) return false;
            if (!TryUnit(right - Vector3.Dot(right, y) * y, minimumLength, out var x)) return false;
            var z = Vector3.Cross(x, y);
            var matrix = new Matrix4x4(x.X, x.Y, x.Z, 0, y.X, y.Y, y.Z, 0,
                                      z.X, z.Y, z.Z, 0, 0, 0, 0, 1);
            rotation = Quaternion.Normalize(Quaternion.CreateFromRotationMatrix(matrix));
            return true;
        }

        public static bool TryBodyFrame(PosePacket packet, int left, int right, int lower, int upper,
                                        float minimumConfidence, float minimumLength,
                                        out Quaternion rotation)
        {
            rotation = Quaternion.Identity;
            if (!TryPoint(packet, left, minimumConfidence, out var l) ||
                !TryPoint(packet, right, minimumConfidence, out var r) ||
                !TryPoint(packet, lower, minimumConfidence, out var a) ||
                !TryPoint(packet, upper, minimumConfidence, out var b)) return false;
            return TryBodyFrame(r - l, b - a, minimumLength, out rotation);
        }

        // Single-axis swing retains the calibrated rest twist. The antipodal case
        // uses a deterministic perpendicular axis; twist is not a measured quantity.
        public static Quaternion Swing(Vector3 fromUnit, Vector3 toUnit)
        {
            if (!TryUnit(fromUnit, 1e-6f, out var from) ||
                !TryUnit(toUnit, 1e-6f, out var to))
                throw new ArgumentException("Swing requires finite nonzero directions.");
            float dot = Math.Max(-1f, Math.Min(1f, Vector3.Dot(from, to)));
            if (dot < -0.999999f)
            {
                var reference = Math.Abs(from.X) < 0.9f ? Vector3.UnitX : Vector3.UnitY;
                var axis = Vector3.Normalize(Vector3.Cross(from, reference));
                return new Quaternion(axis, 0);
            }
            return Quaternion.Normalize(new Quaternion(Vector3.Cross(from, to), 1 + dot));
        }
    }
}
