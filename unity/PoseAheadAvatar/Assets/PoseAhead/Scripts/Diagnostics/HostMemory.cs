using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace PoseAhead
{
    // Mono's Process.WorkingSet64 can return zero on Windows. Use the native API
    // there; never treat an unsupported zero as measured resident memory.
    internal static class HostMemory
    {
        [StructLayout(LayoutKind.Sequential)]
        private struct Counters
        {
            public uint cb, pageFaultCount;
            public UIntPtr peakWorkingSet, workingSet, quotaPeakPaged, quotaPaged;
            public UIntPtr quotaPeakNonPaged, quotaNonPaged, pagefile, peakPagefile;
        }
        [DllImport("kernel32.dll")]
        private static extern IntPtr GetCurrentProcess();
        [DllImport("psapi.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool GetProcessMemoryInfo(IntPtr process, ref Counters counters, uint cb);

        public static bool TryRead(out double mib)
        {
            mib = 0;
            if (Environment.OSVersion.Platform == PlatformID.Win32NT)
            {
                var counters = new Counters { cb = (uint)Marshal.SizeOf<Counters>() };
                if (!GetProcessMemoryInfo(GetCurrentProcess(), ref counters, counters.cb)) return false;
                mib = counters.workingSet.ToUInt64() / 1048576.0;
            }
            else
            {
                using (var process = Process.GetCurrentProcess())
                { process.Refresh(); mib = process.WorkingSet64 / 1048576.0; }
            }
            return mib > 0;
        }
    }
}
