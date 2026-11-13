using System;
using System.IO;
using PoseAhead.Transport;

internal static class Program
{
    private static int Main()
    {
        string encoded = File.ReadAllText(
            Path.Combine(AppContext.BaseDirectory, "udp_v2_golden.base64")).Trim();
        byte[] bytes = Convert.FromBase64String(encoded);
        Require(PosePacketDecoder.TryDecode(bytes, out PosePacket packet, out string error), error);
        Require(packet.FrameId == 1, "Frame ID differs from Python golden packet.");
        Require(packet.SessionId == 42, "Session ID differs from Python golden packet.");
        Require(packet.SourceTimestampNs == 100, "Source timestamp differs.");
        Require(packet.ProducedTimestampNs == 200, "Produced timestamp differs.");
        Require(packet.SendTimestampNs == 300, "Send timestamp differs.");
        Require(packet.Algorithm == PoseAlgorithm.Raw, "Algorithm differs.");
        Require(packet.RootPosition.Length == 3, "Root position is missing.");
        Require(packet.RootPosition[0] == 1f && packet.RootPosition[1] == 2f &&
                packet.RootPosition[2] == 3f, "Root position differs.");
        Require(packet.JointPositions.Length == 51, "Joint position count differs.");
        Require(packet.JointConfidence.Length == 17, "Confidence count differs.");
        for (int joint = 0; joint < 17; joint++)
        {
            Require(packet.JointValidity[joint], $"Joint {joint} should be valid.");
        }

        bytes[50] ^= 0xFF;
        Require(!PosePacketDecoder.TryDecode(bytes, out _, out _), "Corrupt CRC was accepted.");
        DecoderTests.Run();
        ReceiverTests.Run();
        RootTranslationTests.Run();
        RetargetTests.Run();
        FrameRateTests.Run();
        Console.WriteLine("C# UDP v2, root placement and humanoid retarget regressions: PASS");
        return 0;
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }
}
