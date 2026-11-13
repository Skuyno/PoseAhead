using System;
using System.Reflection;
using PoseAhead.Transport;
using UnityEngine;

internal static class RootTranslationTests
{
    private static readonly BindingFlags Fields = BindingFlags.Instance | BindingFlags.NonPublic;

    private static PosePacket Packet(ulong session, ulong frame, float[] root) =>
        new PosePacket { SessionId = session, FrameId = frame, RootPosition = root,
                         RootPositionEstimated = root != null };

    private static void Check(bool value, string error)
    {
        if (!value) throw new InvalidOperationException(error);
    }

    internal static void Run()
    {
        var state = new RootTranslationState();
        Check(state.TryGetOffset(Packet(1, 0, new float[] { 0, 0, 3 }), out _, out bool origin)
              && origin, "Initial root did not establish origin.");
        Check(state.TryGetOffset(Packet(1, 1, new float[] { 1, 0.5f, 2 }), out var offset, out origin)
              && !origin, "Movement did not produce an offset.");
        Check(offset[0] == 1 && offset[1] == -0.5f && offset[2] == -1, "Axis conversion differs.");
        Check(!state.TryGetOffset(Packet(1, 1, new float[] { 99, 99, 99 }), out _, out _),
              "Duplicate frame affected placement.");
        Check(!state.TryGetOffset(Packet(1, 2, null), out _, out _), "Missing root moved avatar.");
        Check(state.TryGetOffset(Packet(2, 0, new float[] { 9, 9, 9 }), out offset, out origin)
              && origin && offset[0] == 0, "New sender session did not reanchor.");
        Check(!state.TryGetOffset(Packet(1, 999, new float[] { 99, 99, 99 }), out _, out _),
              "Old session affected placement.");

        var receiver = new PoseUdpReceiver(); // no socket/camera is opened in this test
        var driver = new PoseRootDriver();
        var target = new Transform { position = new Vector3(10, 20, 30) };
        typeof(PoseRootDriver).GetField("receiver", Fields).SetValue(driver, receiver);
        typeof(PoseRootDriver).GetField("target", Fields).SetValue(driver, target);
        driver.Recenter();
        var update = typeof(PoseRootDriver).GetMethod("LateUpdate", Fields);
        var latest = typeof(PoseUdpReceiver).GetProperty("LatestPacket");
        latest.SetValue(receiver, Packet(1, 0, new float[] { 0, 0, 3 }));
        update.Invoke(driver, null);
        latest.SetValue(receiver, Packet(1, 1, new float[] { 1, 0.5f, 2 }));
        update.Invoke(driver, null);
        Check(target.position.x == 11 && target.position.y == 19.5f && target.position.z == 29,
              "Actual driver did not apply root movement to target.");
        latest.SetValue(receiver, null);
        update.Invoke(driver, null);
        Check(target.position.x == 11, "Expired stream changed placement.");
        latest.SetValue(receiver, Packet(2, 0, new float[] { 9, 9, 9 }));
        update.Invoke(driver, null);
        Check(target.position.x == 11 && target.position.y == 19.5f && target.position.z == 29,
              "Restart teleported the target.");
        driver.Recenter();
        latest.SetValue(receiver, Packet(2, 1, new float[] { 10, 9, 9 }));
        update.Invoke(driver, null);
        Check(target.position.x == 11, "Recenter moved target before new anchor.");
    }
}
