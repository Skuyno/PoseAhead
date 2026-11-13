using System;
using UnityEngine;

namespace PoseAhead
{
    // Process-local demo budget: leave GPU capacity for the camera pose model.
    // Count overlapping demo instances and restore only settings we still own.
    public sealed class DemoFrameRateScope : IDisposable
    {
        public const int TargetFramesPerSecond = 60;
        private static int users, previousTarget, previousVSync;
        private bool disposed;

        public DemoFrameRateScope()
        {
            if (users == 0)
            {
                previousTarget = Application.targetFrameRate;
                previousVSync = QualitySettings.vSyncCount;
                QualitySettings.vSyncCount = 0;
                Application.targetFrameRate = TargetFramesPerSecond;
            }
            users++;
        }

        public void Dispose()
        {
            if (disposed) return;
            disposed = true;
            if (--users != 0) return;
            // Another script may have deliberately changed both settings.
            if (Application.targetFrameRate != TargetFramesPerSecond ||
                QualitySettings.vSyncCount != 0) return;
            Application.targetFrameRate = previousTarget;
            QualitySettings.vSyncCount = previousVSync;
        }
    }
}
