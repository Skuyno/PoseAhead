"""Explicit presentation presets; never start a camera or change research defaults."""

from pathlib import Path

from poseahead.artifacts import sha256_file
from poseahead.desktop.session import SessionOptions
from poseahead.processing.config import read_processor_config

DEMO_PRESETS = {
    "1 · Проверка аватара · без камеры": "diagnostic",
    "2 · CMU · Калман + 100 мс": "cmu_baseline",
    "3 · CMU · экспериментальная нейросеть": "cmu_adaptive",
    "4 · Сохранённый MeTRAbs · без камеры": "metrabs_recording",
}
DEMO_NOTES = {
    "diagnostic": "8 секунд тестовых поз, включая перемещение и пропуск сустава. "
                  "Не оценка качества.",
    "cmu_baseline": "CMU 13/29, до 30 секунд. Это запись, не камера и не замер live-задержки.",
    "cmu_adaptive": "Тот же CMU 13/29. Нейросеть может отключаться защитой; это не доказательство "
                    "снижения задержки. Live-настройки не меняются.",
    "metrabs_recording": "60 сохранённых поз MeTRAbs, около 2,5 секунд. Не новое измерение камеры.",
}


def demo_options(repository, preset, *, udp_port=5005):
    """Resolve all dependencies before the UI applies a preset. No torch/camera imports."""
    if preset not in DEMO_NOTES:
        raise ValueError("Неизвестный сценарий показа.")
    repository = Path(repository).resolve()
    options = {"udp_port": udp_port, "duration_seconds": 8.0}
    if preset.startswith("cmu_"):
        options.update(source="recording", method="kalman_cv", horizon_ms=100,
                       duration_seconds=30.0,
                       recording=str(repository / "outputs/cmu_expanded_20260924/13_29.npz"))
    elif preset == "metrabs_recording":
        options.update(source="recording", duration_seconds=10.0,
                       recording=str(repository / "outputs/fixtures/m1_raw_recording.npz"))
    if preset == "cmu_adaptive":
        path = repository / "configs/processing/adaptive_kalman_recovery_pilot.json"
        values = read_processor_config(path)
        if values["method"] != "adaptive":
            raise ValueError("Сценарий требует adaptive-конфигурацию.")
        bank = values["parameters"]["checkpoints"]
        if set(bank) != {"33", "66", "100", "150"}:
            raise ValueError("Сценарий требует четыре закреплённых checkpoint.")
        for entry in bank.values():
            directory = Path(entry["checkpoint"])
            if sha256_file(directory / "manifest.json") != entry["checkpoint_sha256"]:
                raise ValueError("Checkpoint изменился. Сценарий не применён.")
            for name in ("model.npz", "normalizer.json"):
                if not (directory / name).is_file():
                    raise ValueError(f"Для сценария отсутствует {directory / name}.")
        options.update(method="config", processor_config=str(path))
    result = SessionOptions(**options)
    result.validate()
    return result
