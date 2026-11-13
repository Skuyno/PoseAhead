using System;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

namespace PoseAhead.Transport
{
    public sealed class PoseUdpReceiver : MonoBehaviour
    {
        [SerializeField] private int port = 5005;
        [SerializeField] private float maximumPoseAgeSeconds = 0.5f;
        private UdpClient client;
        private Thread receiveThread;
        private LatestPoseBuffer buffer;
        private volatile bool stopping;
        private volatile string receiveError;

        public PosePacket LatestPacket { get; private set; }
        public int AcceptedPackets => buffer?.Accepted ?? 0;
        public int RejectedPackets => buffer?.Rejected ?? 0;
        public string LastError => receiveError;
        public bool IsReceiving => receiveThread?.IsAlive == true && !stopping;
        public int ListenPort => client?.Client.LocalEndPoint is IPEndPoint endpoint ? endpoint.Port : port;
        public void SessionCounts(out ulong session, out int count)
        {
            session = 0; count = 0;
            buffer?.SessionCounts(out session, out count);
        }
        private static double Now => (double)Stopwatch.GetTimestamp() / Stopwatch.Frequency;

        private void OnEnable()
        {
            if (receiveThread?.IsAlive == true)
                throw new InvalidOperationException("Previous UDP receive thread is still running.");
            // An explicit receiver restart also resets the sender boot/clock epoch.
            buffer = new LatestPoseBuffer(maximumPoseAgeSeconds);
            LatestPacket = null;
            receiveError = null;
            client = new UdpClient(new IPEndPoint(IPAddress.Loopback, port));
            client.Client.ReceiveTimeout = 250;
            stopping = false;
            UdpClient ownedClient = client;
            LatestPoseBuffer ownedBuffer = buffer;
            receiveThread = new Thread(() => ReceiveLoop(ownedClient, ownedBuffer))
            {
                IsBackground = true,
                Name = "PoseAhead UDP receiver",
            };
            receiveThread.Start();
        }

        private void Update() { LatestPacket = buffer?.GetLatest(Now); }

        private void OnDisable()
        {
            stopping = true;
            client?.Close();
            if (receiveThread != null && !receiveThread.Join(1000))
                throw new InvalidOperationException("UDP receive thread did not stop.");
            receiveThread = null;
            client = null;
            LatestPacket = null;
        }

        private void ReceiveLoop(UdpClient socket, LatestPoseBuffer destination)
        {
            var endpoint = new IPEndPoint(IPAddress.Any, 0);
            while (!stopping)
            {
                try
                {
                    byte[] bytes = socket.Receive(ref endpoint);
                    if (IPAddress.IsLoopback(endpoint.Address) &&
                        PosePacketDecoder.TryDecode(bytes, out PosePacket decoded, out _))
                        destination.Offer(decoded, Now);
                    else destination.RejectMalformed();
                }
                catch (SocketException exception)
                {
                    if (stopping) return;
                    if (exception.SocketErrorCode != SocketError.TimedOut)
                    {
                        receiveError = exception.SocketErrorCode.ToString();
                        return;
                    }
                }
                catch (ObjectDisposedException)
                {
                    if (!stopping) receiveError = "Socket disposed unexpectedly.";
                    return;
                }
            }
        }
    }
}
