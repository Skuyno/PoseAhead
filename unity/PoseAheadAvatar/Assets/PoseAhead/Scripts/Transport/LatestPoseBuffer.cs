using System;

namespace PoseAhead.Transport
{
    // Caller supplies its own monotonic clock; sender/receiver clocks are never subtracted.
    public sealed class LatestPoseBuffer
    {
        private readonly object gate = new object();
        private readonly double maximumAgeSeconds;
        private PosePacket latest;
        private ulong sessionId;
        private ulong frameId;
        private bool hasHighWater;
        private double receivedAt;
        private int rejected;
        private int accepted;
        private int sessionAccepted;

        public LatestPoseBuffer(double maximumAgeSeconds = 0.5)
        {
            if (double.IsNaN(maximumAgeSeconds) || double.IsInfinity(maximumAgeSeconds) ||
                maximumAgeSeconds <= 0) throw new ArgumentOutOfRangeException(nameof(maximumAgeSeconds));
            this.maximumAgeSeconds = maximumAgeSeconds;
        }

        public int Rejected { get { lock (gate) return rejected; } }
        public int Accepted { get { lock (gate) return accepted; } }

        public void SessionCounts(out ulong session, out int count)
        { lock (gate) { session = sessionId; count = sessionAccepted; } }

        public bool Offer(PosePacket packet, double localReceivedAt)
        {
            lock (gate)
            {
                if (hasHighWater && (packet.SessionId < sessionId ||
                    (packet.SessionId == sessionId && packet.FrameId <= frameId)))
                {
                    rejected++;
                    return false;
                }
                if (!hasHighWater || packet.SessionId != sessionId) sessionAccepted = 0;
                sessionAccepted++;
                sessionId = packet.SessionId;
                frameId = packet.FrameId;
                hasHighWater = true;
                latest = packet;
                receivedAt = localReceivedAt;
                accepted++;
                return true;
            }
        }

        public PosePacket GetLatest(double now)
        {
            lock (gate)
            {
                if (latest != null && now - receivedAt > maximumAgeSeconds) latest = null;
                return latest;
            }
        }

        public void RejectMalformed() { lock (gate) rejected++; }
    }
}
