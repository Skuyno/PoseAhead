# Рефакторинг · 2026-10-05

Изменены границы ответственности и повторяющийся код. Математика фильтров,
предсказания, causal history, разбиение данных и настройки экспериментов не менялись.
NPZ и UDP v2 сохраняют прежние форматы. Существующие результаты не перезаписывались.

## Что упрощено

- Камерный benchmark перенесён из `scripts/` в `poseahead.runtime.webcam`.
  Desktop вызывает его напрямую; прежняя команда `scripts/benchmark_webcam.py`
  остаётся совместимым launcher'ом. Доступен и `python -m poseahead.runtime.webcam`.
  Проектная проверка desktop опирается на `pyproject.toml`, не на наличие скрипта.
- Единый `processing.config` проверяет схему, допустимый метод и пути checkpoint
  для desktop, загрузчика процессоров и экспериментов. Чтение конфигурации не
  загружает модель. Baseline-методы и их фабрики определены в одном реестре.
- `artifacts.sha256_file` заменяет три реализации и межмодульные импорты приватных
  `_digest`. Hashing больше не требует импорта dataset/evaluation/model-слоёв.
  Названия полей manifest и алгоритм вычисления byte checksum сохранены.
- Замер памяти вынесен из сборки отчёта в `runtime.memory` и переиспользуется
  записанным stress-прогоном. Windows ABI/DLL настраиваются один раз, не при
  каждом образце. `None` по-прежнему означает отсутствие измерения.
- Обработка событий desktop отделена от опроса очереди и таймера Tk.
  Переходы loading/progress/finished/error/exit проверяются отдельными тестами.
- JSON-команды CLI используют общий вывод и обработку ошибок. Сохраняются
  привычные префиксы ошибок, код выхода 2 и отказ перезаписывать результаты.
  Optional импорты остаются внутри соответствующих команд.

Исследовательские protocol-скрипты оставлены: это воспроизводимые рецепты прошлых
прогонов, а не автоматически устаревший код. Они используют общую функцию hashing.
Не добавлялись пустые слои, базовые классы или зависимости.

## Проверка

Полный набор: 537 Python-тестов, Ruff и C# smoke проходят.

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src scripts tests
.venv/Scripts/python.exe scripts/verify_base_install.py
.venv/Scripts/python.exe scripts/smoke_desktop_ui.py --output outputs/new_ui_check
./scripts/test_unity.ps1 -UnityEditor F:/Unity/Editor/Unity.exe -Recording outputs/cmu_expanded_20260924/13_01.npz
.venv/Scripts/python.exe main.py verify-batch outputs/cmu_noise_eval_20261003/batch
```

Новый wheel проходит базовую установку в изолированном venv без torch/cv2/dev
пакетов: `outputs/base_install_20261005_115637_fc9db1a2/report.json`.
Packaged camera CLI `--help` дополнительно выполнен в этом venv без включения камеры.
Три native-теста Unity прошли: `outputs/unity_native_20261005_115627_589ce6bc/`.
Tk start/stop/restart и визуальная проверка собственного окна:
`outputs/ui_refactor_20261005_v1/`.
Существующий CMU batch прошёл read-only проверку: 6 запусков и 49 файлов.

Эти проверки не заменяют 10-минутную приёмку с человеком перед камерой и не
доказывают улучшения accuracy/FPS от рефакторинга.
