using System;
using System.Buffers.Binary;

namespace PoseAhead.Transport
{
    public enum PoseAlgorithm : byte
    {
        Raw = 0,
        Hold = 1,
        Linear = 2,
        DoubleExponential = 3,
        Kalman = 4,
        Ema = 5,
        OneEuro = 6,
        KalmanExtrapolation = 7,
        Predictor33Ms = 10,
        Predictor66Ms = 11,
        Predictor100Ms = 12,
        Predictor150Ms = 13,
        Adaptive = 20,
    }

    public sealed class PosePacket
    {
        public ulong SessionId { get; internal set; }
        public ulong FrameId { get; internal set; }
        public ulong SourceTimestampNs { get; internal set; }
        public ulong ProducedTimestampNs { get; internal set; }
        public ulong SendTimestampNs { get; internal set; }
        public short TargetOffsetMs { get; internal set; }
        public PoseAlgorithm Algorithm { get; internal set; }
        public float? Predictability { get; internal set; }
        public float[] RootPosition { get; internal set; }
        public bool RootPositionEstimated { get; internal set; }
        public float[] RootRotation { get; internal set; }
        public float[] JointPositions { get; internal set; }
        public float[] JointRotations { get; internal set; }
        public float[] JointConfidence { get; internal set; }
        public bool[] JointValidity { get; internal set; }
    }

    public static class PosePacketDecoder
    {
        public const int JointCount = 17;
        public const int MaximumPacketBytes = 1200;
        private const int HeaderBytes = 52;
        private const byte RootPositionFlag = 1 << 0;
        private const byte RootRotationFlag = 1 << 1;
        private const byte JointRotationsFlag = 1 << 2;
        private const byte PredictabilityFlag = 1 << 3;
        private const byte EstimatedRootFlag = 1 << 4;
        private const byte KnownFlags = 0x1F;

        public static bool TryDecode(byte[] data, out PosePacket packet, out string error)
        {
            packet = null;
            error = null;
            if (data == null || data.Length < HeaderBytes + 4 || data.Length > MaximumPacketBytes)
            {
                error = "Packet length is outside protocol bounds.";
                return false;
            }

            ReadOnlySpan<byte> bytes = data;
            ReadOnlySpan<byte> payload = bytes.Slice(0, bytes.Length - 4);
            uint expectedCrc = BinaryPrimitives.ReadUInt32LittleEndian(bytes.Slice(bytes.Length - 4));
            if (ComputeCrc32(payload) != expectedCrc)
            {
                error = "CRC mismatch.";
                return false;
            }
            if (payload[0] != (byte)'P' || payload[1] != (byte)'A' ||
                payload[2] != (byte)'U' || payload[3] != (byte)'D')
            {
                error = "Invalid magic.";
                return false;
            }
            if (payload[4] != 2)
            {
                error = "Unsupported protocol version.";
                return false;
            }

            byte flags = payload[5];
            if ((flags & ~KnownFlags) != 0 || payload[6] != JointCount)
            {
                error = "Unsupported flags or joint count.";
                return false;
            }
            if ((flags & EstimatedRootFlag) != 0 && (flags & RootPositionFlag) == 0)
            {
                error = "Estimated root flag requires a root position.";
                return false;
            }
            if (!Enum.IsDefined(typeof(PoseAlgorithm), payload[7]))
            {
                error = "Unsupported algorithm ID.";
                return false;
            }
            if (BinaryPrimitives.ReadUInt16LittleEndian(payload.Slice(50, 2)) != 0)
            {
                error = "Reserved field is not zero.";
                return false;
            }

            var decoded = new PosePacket
            {
                SessionId = BinaryPrimitives.ReadUInt64LittleEndian(payload.Slice(8, 8)),
                FrameId = BinaryPrimitives.ReadUInt64LittleEndian(payload.Slice(16, 8)),
                SourceTimestampNs = BinaryPrimitives.ReadUInt64LittleEndian(payload.Slice(24, 8)),
                ProducedTimestampNs = BinaryPrimitives.ReadUInt64LittleEndian(payload.Slice(32, 8)),
                SendTimestampNs = BinaryPrimitives.ReadUInt64LittleEndian(payload.Slice(40, 8)),
                TargetOffsetMs = BinaryPrimitives.ReadInt16LittleEndian(payload.Slice(48, 2)),
                Algorithm = (PoseAlgorithm)payload[7],
                RootPositionEstimated = (flags & EstimatedRootFlag) != 0,
            };
            if (decoded.SessionId == 0 || decoded.ProducedTimestampNs < decoded.SourceTimestampNs ||
                decoded.SendTimestampNs < decoded.ProducedTimestampNs)
            {
                error = "Packet timestamps are not monotonic.";
                return false;
            }

            int offset = HeaderBytes;
            try
            {
                if ((flags & PredictabilityFlag) != 0)
                {
                    decoded.Predictability = ReadFloat(payload, ref offset);
                    if (decoded.Predictability < 0f || decoded.Predictability > 1f)
                    {
                        throw new FormatException("Predictability is outside [0, 1].");
                    }
                }
                if ((flags & RootPositionFlag) != 0)
                {
                    decoded.RootPosition = ReadFloats(payload, ref offset, 3);
                }
                if ((flags & RootRotationFlag) != 0)
                {
                    decoded.RootRotation = ReadFloats(payload, ref offset, 4);
                    ValidateQuaternions(decoded.RootRotation);
                }
                decoded.JointPositions = ReadFloats(payload, ref offset, JointCount * 3);
                for (int axis = 0; axis < 3; axis++)
                {
                    if (Math.Abs(decoded.JointPositions[axis]) > 1e-6f)
                        throw new FormatException("Pelvis must be the origin.");
                }
                if ((flags & JointRotationsFlag) != 0)
                {
                    decoded.JointRotations = ReadFloats(payload, ref offset, JointCount * 4);
                    ValidateQuaternions(decoded.JointRotations);
                }
                decoded.JointConfidence = ReadFloats(payload, ref offset, JointCount);
                foreach (float confidence in decoded.JointConfidence)
                {
                    if (confidence < 0f || confidence > 1f)
                    {
                        throw new FormatException("Joint confidence is outside [0, 1].");
                    }
                }

                int validityBytes = (JointCount + 7) / 8;
                if (offset + validityBytes != payload.Length)
                {
                    throw new FormatException("Packet payload has an invalid length.");
                }
                if ((payload[offset + validityBytes - 1] & 0xFE) != 0)
                    throw new FormatException("Unused validity bits must be zero.");
                decoded.JointValidity = new bool[JointCount];
                for (int joint = 0; joint < JointCount; joint++)
                {
                    decoded.JointValidity[joint] =
                        (payload[offset + joint / 8] & (1 << (joint % 8))) != 0;
                }
            }
            catch (FormatException exception)
            {
                error = exception.Message;
                return false;
            }

            packet = decoded;
            return true;
        }

        private static void ValidateQuaternions(float[] values)
        {
            for (int offset = 0; offset < values.Length; offset += 4)
            {
                double squaredNorm = 0;
                for (int axis = 0; axis < 4; axis++)
                {
                    double component = values[offset + axis];
                    squaredNorm += component * component;
                }
                if (Math.Abs(Math.Sqrt(squaredNorm) - 1.0) > 1e-4)
                    throw new FormatException("Rotations must be unit quaternions.");
            }
        }

        private static float[] ReadFloats(ReadOnlySpan<byte> payload, ref int offset, int count)
        {
            var values = new float[count];
            for (int index = 0; index < count; index++)
            {
                values[index] = ReadFloat(payload, ref offset);
            }
            return values;
        }

        private static float ReadFloat(ReadOnlySpan<byte> payload, ref int offset)
        {
            if (offset + 4 > payload.Length)
            {
                throw new FormatException("Packet payload is truncated.");
            }
            int bits = BinaryPrimitives.ReadInt32LittleEndian(payload.Slice(offset, 4));
            offset += 4;
            float value = BitConverter.Int32BitsToSingle(bits);
            if (float.IsNaN(value) || float.IsInfinity(value))
            {
                throw new FormatException("Packet contains a non-finite float.");
            }
            return value;
        }

        private static uint ComputeCrc32(ReadOnlySpan<byte> data)
        {
            uint crc = 0xFFFFFFFFu;
            foreach (byte value in data)
            {
                crc ^= value;
                for (int bit = 0; bit < 8; bit++)
                {
                    uint mask = (uint)-(int)(crc & 1u);
                    crc = (crc >> 1) ^ (0xEDB88320u & mask);
                }
            }
            return ~crc;
        }
    }
}
