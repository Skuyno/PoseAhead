namespace PoseAhead.Transport
{
    // Camera-space translation relative to the first available root in each sender session.
    // This is display placement, not a prediction/filter or an accuracy metric.
    public sealed class RootTranslationState
    {
        private ulong session;
        private ulong frame;
        private bool hasFrame;
        private float[] origin;

        public void Reset() { hasFrame = false; origin = null; }

        public bool TryGetOffset(PosePacket packet, out float[] offset, out bool newOrigin)
        {
            offset = null;
            newOrigin = false;
            if (packet == null) return false;
            if (hasFrame && (packet.SessionId < session ||
                (packet.SessionId == session && packet.FrameId <= frame))) return false;
            if (!hasFrame || packet.SessionId != session) origin = null;
            session = packet.SessionId;
            frame = packet.FrameId;
            hasFrame = true;
            if (packet.RootPosition == null) return false; // hold display position on tracking loss
            if (origin == null)
            {
                origin = (float[])packet.RootPosition.Clone();
                newOrigin = true;
            }
            float dx = packet.RootPosition[0] - origin[0];
            float dy = packet.RootPosition[1] - origin[1];
            float dz = packet.RootPosition[2] - origin[2];
            offset = new float[] {
                dx,
                -dy,
                dz };
            return true;
        }
    }
}
