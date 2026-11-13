using System;
using System.Buffers.Binary;
using System.IO;
using PoseAhead.Transport;

internal static class DecoderTests
{
    private static void Check(bool value, string message)
    {
        if (!value) throw new InvalidOperationException(message);
    }

    private static byte[] WithCrc(byte[] payload)
    {
        byte[] result = new byte[payload.Length + 4];
        Array.Copy(payload, result, payload.Length);
        uint crc = 0xFFFFFFFFu;
        foreach (byte value in payload)
        {
            crc ^= value;
            for (int bit = 0; bit < 8; bit++)
                crc = (crc >> 1) ^ ((crc & 1u) == 1u ? 0xEDB88320u : 0u);
        }
        BinaryPrimitives.WriteUInt32LittleEndian(result.AsSpan(payload.Length), ~crc);
        return result;
    }

    private static byte[] SelectFields(byte[] full, byte flags)
    {
        byte[] header = full.AsSpan(0, 52).ToArray();
        header[5] = flags;
        using (var stream = new MemoryStream())
        {
            stream.Write(header, 0, header.Length);
            if ((flags & 8) != 0) stream.Write(full, 52, 4);
            if ((flags & 1) != 0) stream.Write(full, 56, 12);
            if ((flags & 2) != 0) stream.Write(full, 68, 16);
            stream.Write(full, 84, 204);
            if ((flags & 4) != 0) stream.Write(full, 288, 272);
            stream.Write(full, 560, 71);
            return WithCrc(stream.ToArray());
        }
    }

    internal static void Run()
    {
        byte[] full = Convert.FromBase64String(File.ReadAllText(
            Path.Combine(AppContext.BaseDirectory, "udp_v2_full_golden.base64")).Trim());
        Check(full.Length == 635, "Full packet length differs from Python.");
        Check(PosePacketDecoder.TryDecode(full, out var packet, out var error), error);
        Check(packet.Algorithm == PoseAlgorithm.KalmanExtrapolation, "New baseline ID differs.");
        Check(packet.TargetOffsetMs == 100 && packet.Predictability == 0.75f, "Target metadata differs.");
        byte[] estimated = full.AsSpan(0, full.Length - 4).ToArray();
        estimated[5] |= 16;
        Check(PosePacketDecoder.TryDecode(WithCrc(estimated), out packet, out error) &&
              packet.RootPositionEstimated, "Estimated-root marker was lost.");
        Check(!PosePacketDecoder.TryDecode(SelectFields(full, 16), out _, out _),
              "Estimated-root marker without root was accepted.");
        for (byte flags = 0; flags < 16; flags++)
        {
            byte[] bytes = SelectFields(full, flags);
            Check(PosePacketDecoder.TryDecode(bytes, out packet, out error), error);
            Check((packet.RootPosition != null) == ((flags & 1) != 0), "Root flag differs.");
            Check((packet.RootRotation != null) == ((flags & 2) != 0), "Root rotation flag differs.");
            Check((packet.JointRotations != null) == ((flags & 4) != 0), "Joint rotation flag differs.");
            Check(packet.Predictability.HasValue == ((flags & 8) != 0), "Predictability flag differs.");
            for (int joint = 0; joint < 17; joint++)
            {
                Check(packet.JointValidity[joint] == (joint % 2 == 0), "Validity order differs.");
                for (int axis = 0; axis < 3; axis++)
                    Check(Math.Abs(packet.JointPositions[joint * 3 + axis] - joint * 0.03f) < 1e-6f,
                          "Position order differs.");
                if (packet.JointRotations != null)
                    Check(packet.JointRotations[joint * 4 + 3] == 1, "Quaternion order differs.");
            }
        }
        foreach (byte algorithm in new byte[] { 5, 6, 7 })
        {
            byte[] payload = full.AsSpan(0, full.Length - 4).ToArray();
            payload[7] = algorithm;
            Check(PosePacketDecoder.TryDecode(WithCrc(payload), out packet, out error), error);
            Check((byte)packet.Algorithm == algorithm, "Baseline ID was remapped.");
        }
        foreach (int offset in new int[] { 80, 300 })
        {
            byte[] payload = full.AsSpan(0, full.Length - 4).ToArray();
            Array.Clear(payload, offset, 4); // identity quaternion becomes zero
            Check(!PosePacketDecoder.TryDecode(WithCrc(payload), out _, out _),
                  "Zero quaternion was accepted.");
        }
        foreach (int offset in new int[] { 5, 7, 50, 630 })
        {
            byte[] payload = full.AsSpan(0, full.Length - 4).ToArray();
            payload[offset] = 255;
            Check(!PosePacketDecoder.TryDecode(WithCrc(payload), out _, out _),
                  "Invalid flags, algorithm, reserved field or padding were accepted.");
        }
        for (int length = 52; length < full.Length - 4; length++)
            Check(!PosePacketDecoder.TryDecode(WithCrc(full.AsSpan(0, length).ToArray()), out _, out _),
                  "Truncated payload with valid CRC was accepted.");
    }
}
