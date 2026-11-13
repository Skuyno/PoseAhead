"""Visible desktop QA against actual producer/loopback UDP, never a real camera."""

import argparse
import ctypes
import json
import socket
import time
import traceback
from ctypes import wintypes
from pathlib import Path

from poseahead.desktop.app import METHODS, SOURCES, DesktopApp
from poseahead.desktop.demo import DEMO_PRESETS
from poseahead.desktop.preferences import PreferenceStore
from poseahead.transport import decode_udp_packet


def main():
    import tkinter as tk
    from tkinter import messagebox

    import win32gui
    import win32ui
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new QA artifact directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[1]
    root = tk.Tk()
    app = DesktopApp(root, repository, remember_settings=False)
    app.preferences = PreferenceStore(args.output / "desktop_settings.json")
    app.live_profile_directory = args.output / "live_profiles"
    errors, captured, reports = [], [], []
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.setblocking(False)
    app.port.set(str(receiver.getsockname()[1]))
    deadline = time.monotonic() + 20
    phase = 0
    stop_requested = False

    def image(name):
        root.update()
        # Capture this exact Tk HWND even when another app covers it. Never grab the desktop.
        window = win32gui.GetParent(root.winfo_id()) or root.winfo_id()
        _, _, width, height = win32gui.GetClientRect(window)
        handle = win32gui.GetDC(window)
        source = win32ui.CreateDCFromHandle(handle)
        memory = source.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(source, width, height)
        previous = memory.SelectObject(bitmap)
        try:
            win32gui.RedrawWindow(window, None, None, 0x0001 | 0x0004 | 0x0080 | 0x0100)
            printer = ctypes.windll.user32.PrintWindow
            printer.argtypes = (wintypes.HWND, wintypes.HDC, wintypes.UINT)
            printer.restype = wintypes.BOOL
            if not printer(window, memory.GetSafeHdc(), 1):
                raise RuntimeError("PrintWindow failed for our Tk window.")
            pixels = bitmap.GetBitmapBits(True)
            picture = Image.frombuffer("RGB", (width, height), pixels, "raw", "BGRX", 0, 1)
            picture.save(args.output / name)
        finally:
            memory.SelectObject(previous)
            win32gui.DeleteObject(bitmap.GetHandle())
            memory.DeleteDC()
            source.DeleteDC()
            win32gui.ReleaseDC(window, handle)

    def tick():
        nonlocal phase, stop_requested
        try:
            if time.monotonic() > deadline:
                raise RuntimeError("Desktop smoke timed out.")
            while True:
                try:
                    captured.append(decode_udp_packet(receiver.recvfrom(1200)[0]))
                except BlockingIOError:
                    break
            if phase == 0:
                for label, preset in DEMO_PRESETS.items():
                    app.demo_preset.set(label)
                    app.demo_button.invoke()
                    assert not app.controller.busy  # applying never starts anything
                    options = app.options()
                    assert options.source != "camera" and not options.record_raw
                    if preset == "cmu_adaptive":
                        assert options.method == "config"
                        assert "recovery" in options.processor_config
                    image("preset_" + preset + ".png")
                app.demo_preset.set(next(iter(DEMO_PRESETS)))
                app.demo_button.invoke()
                app.save_settings()
                app.duration.set("999")
                app.restore_settings()
                assert float(app.duration.get()) == 8
                assert not app.record_raw.get()
                image("idle.png")
                app.handle_event({"kind": "progress", "unity": {
                    "status": "observed", "written_unix_ms": time.time() * 1000,
                    "received_pose_frames": 12, "applied_pose_frames": 10,
                    "render_fps": 60, "receiving": True,
                }})
                assert "принято 12" in app.unity_status.get()
                image("synthetic_live_telemetry.png")
                app.unity_deadline = time.monotonic() - 1
                app.expire_unity_status()
                assert "перестал подтверждать" in app.unity_status.get()
                image("synthetic_stale_telemetry.png")
                app.source.set(
                    next(label for label, source in SOURCES.items() if source == "camera")
                )
                app.method.set(
                    next(label for label, method in METHODS.items() if method == "config")
                )
                phase = 1
            elif phase == 1:
                root.update_idletasks()
                assert str(app.live_widget.cget("state")) == "normal"
                assert str(app.processor_widget.cget("state")) == "normal"
                assert str(app.horizon_widget.cget("state")) == "disabled"
                original_live = Path(app.live_config.get()).read_bytes()
                app.fast_camera_button.invoke()
                fast = json.loads(Path(app.live_config.get()).read_text(encoding="utf-8"))
                assert fast["crop_precision"] == "fp16_conv"
                assert (repository / "configs/live/default.json").read_bytes() == original_live
                assert not app.controller.busy
                image("camera_settings.png")
                app.source.set(next(iter(SOURCES)))
                app.method.set(next(iter(METHODS)))
                app.duration.set("-1")
                previous_error = messagebox.showerror
                try:
                    messagebox.showerror = lambda *args, **kwargs: errors.append(args[1])
                    app.start_button.invoke()
                finally:
                    messagebox.showerror = previous_error
                assert len(errors) == 1 and not app.controller.busy
                app.duration.set("2")
                app.start_button.invoke()
                phase = 2
            elif phase == 2 and app.metrics["sent"].get() not in ("—", "0"):
                image("running.png")
                assert str(app.start_button.cget("state")) == "disabled"
                assert str(app.stop_button.cget("state")) == "normal"
                app.stop_button.invoke()
                stop_requested = True
                phase = 3
            elif (
                phase == 3
                and not app.controller.busy
                and app.finished_seen
                and str(app.start_button.cget("state")) == "normal"
            ):
                report = json.loads(
                    (app.controller.output_directory / "report.json").read_text(encoding="utf-8")
                )
                assert report["status"] == "stopped" and report["sent_frames"] > 0
                assert not app.error_seen
                reports.append(report)
                image("stopped.png")
                app.duration.set(".5")
                app.start_button.invoke()
                phase = 4
            elif (
                phase == 4
                and not app.controller.busy
                and app.finished_seen
                and str(app.start_button.cget("state")) == "normal"
            ):
                report = json.loads(
                    (app.controller.output_directory / "report.json").read_text(encoding="utf-8")
                )
                assert report["status"] == "completed" and report["sent_frames"] >= 2
                assert len(captured) == reports[0]["sent_frames"] + report["sent_frames"]
                reports.append(report)
                image("completed.png")
                result = {
                    "status": "passed",
                    "actual_udp_packets": len(captured),
                    "stop_button_used": stop_requested,
                    "validation_errors": errors,
                    "reports": reports,
                    "camera_started": False,
                    "preferences_roundtrip": "passed_in_isolated_qa_file",
                    "receiver_display_check": "synthetic events only; not actual Unity evidence",
                    "scope": "Native Tk window + real producer; no camera/Unity acceptance.",
                }
                with (args.output / "report.json").open("x", encoding="utf-8") as stream:
                    json.dump(result, stream, ensure_ascii=False, indent=2)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                root.destroy()
                return
            root.after(100, tick)
        except Exception as error:
            traceback.print_exc()
            errors.append(error)
            app.controller.force_stop()
            root.destroy()

    root.after(400, tick)
    try:
        root.mainloop()
    finally:
        receiver.close()
        app.controller.force_stop()
    if any(isinstance(error, Exception) for error in errors):
        raise RuntimeError(str(errors[-1]))


if __name__ == "__main__":
    main()
