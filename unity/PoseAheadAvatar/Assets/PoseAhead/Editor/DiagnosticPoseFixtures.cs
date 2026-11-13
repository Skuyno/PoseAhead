using System;
using System.IO;
using PoseAhead.Transport;
using UnityEngine;

namespace PoseAhead.Editor
{
    public static class DiagnosticPoseFixtures
    {
        [Serializable]
        private sealed class WireFixtures
        {
            public int version = 0;
            public string[] packets = null;
        }

        public static byte[][] LoadPackets()
        {
            string path = Path.Combine(Application.dataPath, "PoseAhead/Tests/Fixtures/diagnostic_packets.json");
            var fixtures = JsonUtility.FromJson<WireFixtures>(File.ReadAllText(path));
            if (fixtures == null || fixtures.version != 1 || fixtures.packets == null || fixtures.packets.Length != 8)
                throw new InvalidDataException("Expected eight version-1 diagnostic wire fixtures.");
            var packets = new byte[fixtures.packets.Length][];
            for (int i = 0; i < packets.Length; i++)
            {
                packets[i] = Convert.FromBase64String(fixtures.packets[i]);
                if (!PosePacketDecoder.TryDecode(packets[i], out _, out var error))
                    throw new InvalidDataException("Diagnostic fixture " + i + ": " + error);
            }
            return packets;
        }
    }
}
