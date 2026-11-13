using System;
using PoseAhead;
using UnityEngine;

internal static class FrameRateTests
{
    public static void Run()
    {
        int originalTarget = Application.targetFrameRate, originalVSync = QualitySettings.vSyncCount;
        try
        {
            Application.targetFrameRate = 144; QualitySettings.vSyncCount = 2;
            var first = new DemoFrameRateScope();
            var second = new DemoFrameRateScope();
            Require(Application.targetFrameRate == 60 && QualitySettings.vSyncCount == 0,
                    "Demo render limit not applied.");
            first.Dispose(); first.Dispose();
            Require(Application.targetFrameRate == 60, "Overlapping demo lost its limit.");
            second.Dispose();
            Require(Application.targetFrameRate == 144 && QualitySettings.vSyncCount == 2,
                    "Previous render settings not restored.");
            using (var scope = new DemoFrameRateScope())
            {
                Application.targetFrameRate = 120;
                QualitySettings.vSyncCount = 1;
            }
            Require(Application.targetFrameRate == 120 && QualitySettings.vSyncCount == 1,
                    "External render-setting change was overwritten.");
        }
        finally
        { Application.targetFrameRate = originalTarget; QualitySettings.vSyncCount = originalVSync; }
    }

    private static void Require(bool condition, string message)
    { if (!condition) throw new InvalidOperationException(message); }
}
