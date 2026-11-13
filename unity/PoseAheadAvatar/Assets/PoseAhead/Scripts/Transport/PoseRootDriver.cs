using System;
using UnityEngine;

namespace PoseAhead.Transport
{
    // Attach to an avatar container (not an individual humanoid bone).
    // Bone retargeting is handled separately by PoseHumanoidDriver.
    public sealed class PoseRootDriver : MonoBehaviour
    {
        [SerializeField] private PoseUdpReceiver receiver = null;
        [SerializeField] private Transform target = null;
        [SerializeField] private float sceneUnitsPerMeter = 1f;
        private readonly RootTranslationState state = new RootTranslationState();
        private Vector3 sceneOrigin;

        public void Configure(PoseUdpReceiver source, Transform destination)
        {
            receiver = source; target = destination; Recenter();
        }

        private void OnEnable() { Recenter(); }

        public void Recenter()
        {
            state.Reset();
            sceneOrigin = (target != null ? target : transform).position;
        }

        private void LateUpdate()
        {
            ApplyPacket(receiver != null ? receiver.LatestPacket : null);
        }

        // Shared by live playback and deterministic Editor previews.
        public bool ApplyPacket(PosePacket packet)
        {
            if (float.IsNaN(sceneUnitsPerMeter) || float.IsInfinity(sceneUnitsPerMeter) ||
                sceneUnitsPerMeter <= 0) throw new InvalidOperationException("Invalid scene scale.");
            if (!state.TryGetOffset(packet, out var offset, out bool newOrigin)) return false;
            Transform destination = target != null ? target : transform;
            if (newOrigin) sceneOrigin = destination.position;
            destination.position = new Vector3(
                sceneOrigin.x + offset[0] * sceneUnitsPerMeter,
                sceneOrigin.y + offset[1] * sceneUnitsPerMeter,
                sceneOrigin.z + offset[2] * sceneUnitsPerMeter);
            return true;
        }
    }
}
