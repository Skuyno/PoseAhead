"""Small, local, atomic UI preferences; never persist recording consent or auto-start."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

from poseahead.processing.config import METHODS

FIELDS = {"source", "method", "duration_seconds", "camera_index", "udp_port", "horizon_ms",
          "recording", "live_config", "processor_config", "unity_editor"}
PATH_FIELDS = {"recording", "live_config", "processor_config", "unity_editor"}


def validate_preferences(values):
    """Validate data only: do not read configs, open devices, or load checkpoints."""
    if not isinstance(values, dict) or not set(values) <= FIELDS:
        raise ValueError("Неизвестные поля настроек панели.")
    for key, value in values.items():
        valid = True
        if key == "source":
            valid = isinstance(value, str) and value in ("diagnostic", "recording", "camera")
        elif key == "method":
            valid = isinstance(value, str) and value in (*METHODS, "config")
        elif key == "duration_seconds":
            valid = (type(value) in (int, float) and math.isfinite(value) and 0 < value <= 3600)
        elif key == "camera_index":
            valid = type(value) is int and value >= 0
        elif key == "udp_port":
            valid = type(value) is int and 1 <= value <= 65535
        elif key == "horizon_ms":
            valid = type(value) is int and value in (33, 66, 100, 150)
        elif key in PATH_FIELDS:
            valid = isinstance(value, str) and len(value) <= 8192 and "\0" not in value
        if not valid:
            raise ValueError("Некорректная сохранённая настройка: " + key)
    return dict(values)


class PreferenceStore:
    def __init__(self, path):
        self.path = Path(path)
        self.blocked = False

    def load(self):
        """Ignore bad/unsupported data without destroying the original preferences."""
        self.blocked = False
        try:
            if not self.path.exists():
                return {}, None
            if self.path.stat().st_size > 65536:
                raise ValueError("Файл настроек слишком большой.")
            values = json.loads(self.path.read_text(encoding="utf-8-sig"))
            if (not isinstance(values, dict) or set(values) != {"version", "settings"}
                    or type(values["version"]) is not int or values["version"] != 1):
                raise ValueError("Неизвестная версия файла настроек панели.")
            return validate_preferences(values["settings"]), None
        except (OSError, ValueError, TypeError):
            self.blocked = True
            return {}, ("Сохранённые настройки не прочитаны; используются начальные. "
                        "Исходный файл не изменён: " + str(self.path))

    def save(self, values):
        if self.blocked:
            raise ValueError("Повреждённый или несовместимый файл настроек сохранён без изменений. "
                             "Переименуйте его для повторного сохранения: " + str(self.path))
        values = validate_preferences(values)
        data = json.dumps({"version": 1, "settings": values},
                          ensure_ascii=False, indent=2, allow_nan=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent,
                prefix=self.path.name + ".", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(data + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
