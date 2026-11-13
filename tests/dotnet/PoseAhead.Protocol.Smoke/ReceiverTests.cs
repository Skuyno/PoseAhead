using System;
using System.Buffers.Binary;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Threading;
using PoseAhead.Transport;

internal static class ReceiverTests
{
    private const BindingFlags PrivateInstance = BindingFlags.Instance | BindingFlags.NonPublic;

    private static void Check(bool value, string message)
    {
        if (!value) throw new InvalidOperationException(message);
    }

    private static byte[] PacketBytes(ulong session, ulong frame)
    {
        byte[] packet = Convert.FromBase64String(File.ReadAllText(
            Path.Combine(AppContext.BaseDirectory, "udp_v2_golden.base64")).Trim());
        BinaryPrimitives.WriteUInt64LittleEndian(packet.AsSpan(8, 8), session);
        BinaryPrimitives.WriteUInt64LittleEndian(packet.AsSpan(16, 8), frame);
        uint crc = 0xFFFFFFFFu;
        for (int i = 0; i < packet.Length - 4; i++)
        {
            crc ^= packet[i];
            for (int bit = 0; bit < 8; bit++)
                crc = (crc >> 1) ^ ((crc & 1u) == 1u ? 0xEDB88320u : 0u);
        }
        BinaryPrimitives.WriteUInt32LittleEndian(packet.AsSpan(packet.Length - 4), ~crc);
        return packet;
    }

    private static PosePacket Packet(ulong session, ulong frame)
    {
        Check(PosePacketDecoder.TryDecode(PacketBytes(session, frame), out var packet, out var error), error);
        return packet;
    }

    private static void Invoke(PoseUdpReceiver receiver, string method)
    {
        typeof(PoseUdpReceiver).GetMethod(method, PrivateInstance).Invoke(receiver, null);
    }

    private static void Send(PoseUdpReceiver receiver, ulong session, ulong frame)
    {
        var socket = (UdpClient)typeof(PoseUdpReceiver).GetField("client", PrivateInstance).GetValue(receiver);
        int port = ((IPEndPoint)socket.Client.LocalEndPoint).Port;
        int previous = receiver.AcceptedPackets + receiver.RejectedPackets;
        using (var sender = new UdpClient())
        {
            byte[] bytes = PacketBytes(session, frame);
            sender.Send(bytes, bytes.Length, "127.0.0.1", port);
        }
        Check(SpinWait.SpinUntil(() => receiver.AcceptedPackets + receiver.RejectedPackets > previous, 2000),
              "Receiver did not process the loopback datagram.");
    }

    internal static void Run()
    {
        var buffer = new LatestPoseBuffer(0.5);
        Check(buffer.Offer(Packet(42, 101), 0), "First packet rejected.");
        buffer.SessionCounts(out ulong countedSession, out int count);
        Check(countedSession == 42 && count == 1, "First session telemetry differs.");
        Check(!buffer.Offer(Packet(42, 100), 0.1), "Old packet accepted.");
        Check(!buffer.Offer(Packet(42, 101), 0.1), "Duplicate packet accepted.");
        Check(buffer.GetLatest(0.2).FrameId == 101, "Old packet displaced latest.");
        Check(buffer.GetLatest(0.6) == null, "Expired pose remained available.");
        Check(!buffer.Offer(Packet(42, 100), 0.7), "Expiry forgot the high-water mark.");
        Check(buffer.Offer(Packet(43, 0), 1), "New session with reset frame ID rejected.");
        Check(!buffer.Offer(Packet(42, 9999), 1.1), "Delayed previous session accepted.");
        buffer.SessionCounts(out countedSession, out count);
        Check(countedSession == 43 && count == 1 && buffer.Accepted == 2,
              "Telemetry did not reset per session or counted rejected packets.");
        Check(buffer.GetLatest(1.2).FrameId == 0, "Session restart did not reset frame order.");

        var receiver = new PoseUdpReceiver();
        typeof(PoseUdpReceiver).GetField("port", PrivateInstance).SetValue(receiver, 0);
        Invoke(receiver, "OnEnable");
        try
        {
            var boundClient = (UdpClient)typeof(PoseUdpReceiver)
                .GetField("client", PrivateInstance).GetValue(receiver);
            Check(((IPEndPoint)boundClient.Client.LocalEndPoint).Address.Equals(IPAddress.Loopback),
                  "Receiver must bind exclusively to loopback.");
            bool duplicateEnableRejected = false;
            try { Invoke(receiver, "OnEnable"); }
            catch (TargetInvocationException exception)
            {
                duplicateEnableRejected = exception.InnerException is InvalidOperationException;
            }
            Check(duplicateEnableRejected, "Duplicate enable replaced a running receive thread.");
            Send(receiver, 42, 101);
            Send(receiver, 42, 100);
            Invoke(receiver, "Update");
            Check(receiver.LatestPacket.FrameId == 101, "Real receiver applied older packet.");
            Send(receiver, 43, 0);
            Send(receiver, 42, 9999);
            Invoke(receiver, "Update");
            Check(receiver.LatestPacket.SessionId == 43 && receiver.LatestPacket.FrameId == 0,
                  "Real receiver failed sender restart.");
            Invoke(receiver, "OnDisable");
            Check(receiver.LatestPacket == null, "Disable retained a stale pose.");
            Invoke(receiver, "OnEnable");
            Send(receiver, 1, 0);
            Invoke(receiver, "Update");
            Check(receiver.LatestPacket.SessionId == 1, "Explicit reset retained old clock epoch.");
            Check(receiver.IsReceiving && receiver.LastError == null, "Receiver health is incorrect.");
            var restartedClient = (UdpClient)typeof(PoseUdpReceiver)
                .GetField("client", PrivateInstance).GetValue(receiver);
            restartedClient.Close(); // unexpected failure must be observable, not silent
            Check(SpinWait.SpinUntil(() => !receiver.IsReceiving, 2000), "Failed socket kept running.");
            Check(receiver.LastError != null, "Unexpected socket failure was not reported.");
        }
        finally { Invoke(receiver, "OnDisable"); }
    }
}
