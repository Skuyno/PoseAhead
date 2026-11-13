"""Validated UI requests and ownership of one isolated producer process."""

from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from poseahead.config import load_live_config
from poseahead.processing.config import METHODS, read_processor_config

EVENT_PREFIX = "POSEAHEAD_EVENT "


def unity_launch_command(repository, editor):
    """Read executable metadata before opening the pinned project; never downgrade it."""
    repository, editor = Path(repository).resolve(), Path(editor).resolve()
    if not editor.is_file() or editor.name.lower() != "unity.exe":
        raise ValueError("Выберите Unity.exe самого Editor, не Unity Hub.")
    project = repository / "unity/PoseAheadAvatar"
    lines = (
        (project / "ProjectSettings/ProjectVersion.txt").read_text(encoding="utf-8").splitlines()
    )
    expected = next(
        (line.split(":", 1)[1].strip() for line in lines if line.startswith("m_EditorVersion:")),
        None,
    )
    if expected is None:
        raise ValueError("В проекте не закреплена версия Unity.")
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(repository / "scripts/unity_editor_version.ps1"),
            "-UnityEditor",
            str(editor),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        timeout=5,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    version = result.stdout.strip()
    if version != expected and not version.startswith(expected + "_"):
        raise ValueError(
            f"Нужен Unity {expected}; выбран {version or 'неизвестная версия'}. "
            "Проект не открыт и не изменён."
        )
    return [str(editor), "-projectPath", str(project)]


@dataclass(frozen=True)
class SessionOptions:
    source: str = "diagnostic"
    method: str = "raw"
    duration_seconds: float = 30.0
    camera_index: int = 0
    udp_port: int = 5005
    horizon_ms: int = 100
    recording: str = ""
    live_config: str = ""
    processor_config: str = ""
    record_raw: bool = False

    def validate(self):
        if self.source not in ("diagnostic", "recording", "camera"):
            raise ValueError("Неизвестный источник движения.")
        if self.method not in (*METHODS, "config"):
            raise ValueError("Неизвестный метод обработки.")
        if (
            type(self.duration_seconds) not in (int, float)
            or not math.isfinite(self.duration_seconds)
            or not 0 < self.duration_seconds <= 3600
        ):
            raise ValueError("Длительность должна быть от 0 до 3600 секунд (не включая 0).")
        if type(self.camera_index) is not int or self.camera_index < 0:
            raise ValueError("Индекс камеры должен быть целым неотрицательным числом.")
        if type(self.udp_port) is not int or not 1 <= self.udp_port <= 65535:
            raise ValueError("UDP-порт должен быть от 1 до 65535.")
        if type(self.horizon_ms) is not int or self.horizon_ms not in (33, 66, 100, 150):
            raise ValueError("Выберите горизонт 33, 66, 100 или 150 мс.")
        if type(self.record_raw) is not bool or self.record_raw and self.source != "camera":
            raise ValueError("Запись новых наблюдений доступна только для камеры.")
        if self.source == "recording" and not Path(self.recording).is_file():
            raise ValueError("Выберите существующую запись NPZ.")
        if self.source == "camera":
            config = load_live_config(self.live_config)
            if config.udp_host != "127.0.0.1":
                raise ValueError("Desktop-панель поддерживает только UDP на 127.0.0.1.")
        if self.method == "config":
            path = Path(self.processor_config)
            if not path.is_file():
                raise ValueError("Выберите JSON-конфигурацию процессора.")
            values = read_processor_config(path)
            if (
                self.source == "camera"
                and values["method"] in ("learned", "adaptive")
                and values["parameters"].get("history_sampling") != "linear_past"
            ):
                raise ValueError("Для MLP с камерой требуется history_sampling: linear_past.")


class SessionController:
    """Never kill Unity, cameras owned by another app, or an unrelated Python process."""

    def __init__(self, repository: Path):
        self.repository = Path(repository).resolve()
        self.events = queue.Queue(maxsize=200)
        self.process = None
        self.reader = None
        self.output_directory = None
        self.started_at = None
        self.stop_sent = False

    @property
    def busy(self):
        return (
            self.process is not None
            and self.process.poll() is None
            or self.reader is not None
            and self.reader.is_alive()
        )

    def _publish(self, event):
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            self.events.put_nowait(event)

    def start(self, options: SessionOptions):
        if self.busy:
            raise RuntimeError("Сессия уже запущена; сначала остановите её.")
        options.validate()
        if not (self.repository / "pyproject.toml").is_file():
            raise ValueError("Нужна рабочая папка проекта PoseAhead.")
        while not self.events.empty():
            self.events.get_nowait()
        directory = (
            self.repository
            / "outputs"
            / ("ui_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
        )
        directory.mkdir(parents=True, exist_ok=False)
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        # Supports a source checkout even when launched with a different working directory.
        env["PYTHONPATH"] = str(self.repository / "src") + os.pathsep + env.get("PYTHONPATH", "")
        # A pythonw launcher has no console streams. The worker needs redirected
        # stdin/stdout, so use its sibling python.exe with CREATE_NO_WINDOW.
        interpreter = Path(sys.executable)
        if os.name == "nt" and interpreter.name.lower() == "pythonw.exe":
            interpreter = interpreter.with_name("python.exe")
        self.process = subprocess.Popen(
            [
                str(interpreter),
                "-u",
                "-m",
                "poseahead.desktop.worker",
                "--repository",
                str(self.repository),
                "--output",
                str(directory),
                "--options",
                json.dumps(asdict(options), ensure_ascii=False),
            ],
            cwd=self.repository,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self.output_directory = directory
        self.started_at = time.monotonic()
        self.stop_sent = False
        owned_process = self.process
        self.reader = threading.Thread(
            target=self._read,
            args=(owned_process, directory),
            daemon=True,
            name="poseahead-ui-output",
        )
        self.reader.start()

    def _read(self, owned_process, directory):
        log = None
        try:
            log = (directory / "worker.log").open("x", encoding="utf-8")
        except OSError as error:
            self._publish({"kind": "log", "message": "Не удалось сохранить журнал: " + str(error)})
        logged_bytes = 0
        try:
            for line in owned_process.stdout:
                if log is not None and logged_bytes < 4 * 1024**2:
                    try:
                        log.write(line)
                        log.flush()
                        logged_bytes += len(line.encode("utf-8"))
                    except OSError as error:
                        self._publish({"kind": "log", "message": "Журнал: " + str(error)})
                        try:
                            log.close()
                        except OSError:
                            pass
                        log = None
                if line.startswith(EVENT_PREFIX):
                    try:
                        event = json.loads(line[len(EVENT_PREFIX) :])
                        if not isinstance(event, dict) or "kind" not in event:
                            raise ValueError("missing event kind")
                        self._publish(event)
                    except (ValueError, TypeError):
                        self._publish({"kind": "log", "message": line[:4096].rstrip()})
                else:
                    self._publish({"kind": "log", "message": line[:4096].rstrip()})
            code = owned_process.wait()
            self._publish({"kind": "exit", "code": code})
        finally:
            if log is not None:
                try:
                    log.close()
                except OSError:
                    pass
            owned_process.stdout.close()
            owned_process.stdin.close()

    def request_stop(self):
        if self.process is None or self.process.poll() is not None or self.stop_sent:
            return
        try:
            self.process.stdin.write("stop\n")
            self.process.stdin.flush()
            self.stop_sent = True
        except (BrokenPipeError, OSError, ValueError):
            pass  # Already ending; the reader delivers the actual result/exit status.

    def force_stop(self):
        # The UI must explicitly confirm that unsaved observations may be lost.
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
