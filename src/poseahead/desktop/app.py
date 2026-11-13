"""Native desktop control panel. Avatar rendering remains in the Unity window."""

from __future__ import annotations

import os
import queue
import subprocess
import time
from pathlib import Path

from poseahead.desktop.camera_profile import fast_camera_profile
from poseahead.desktop.demo import DEMO_NOTES, DEMO_PRESETS, demo_options
from poseahead.desktop.preferences import FIELDS, PreferenceStore
from poseahead.desktop.session import SessionController, SessionOptions, unity_launch_command

SOURCES = {
    "Тестовые движения · 8 секунд": "diagnostic",
    "Запись движения · NPZ": "recording",
    "Камера · MeTRAbs": "camera",
}
METHODS = {
    "Без обработки · Raw": "raw",
    "Сглаживание · EMA": "ema",
    "Адаптивный фильтр · One-Euro": "one_euro",
    "Фильтр Калмана": "kalman",
    "Калман + экстраполяция": "kalman_cv",
    "Из JSON · включая MLP": "config",
}


class DesktopApp:
    def __init__(self, root, repository, *, remember_settings=True):
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk = tk, ttk
        self.root = root
        self.repository = Path(repository).resolve()
        self.live_profile_directory = self.repository / "outputs/live_profiles"
        self.controller = SessionController(self.repository)
        self.preferences = (PreferenceStore(self.repository / "outputs/desktop_settings.json")
                            if remember_settings else None)
        self.unity_deadline = None
        self.closing = False
        self.error_seen = False
        self.finished_seen = False
        self.stop_time = None
        self.inputs = []
        self.groups = {}
        self.last_metrics = {}
        root.title("PoseAhead — панель управления")
        root.geometry("1160x850")
        root.minsize(1060, 730)
        root.configure(bg="#111827")
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#111827")
        style.configure("Card.TFrame", background="#1b2638")
        style.configure("TLabel", background="#111827", foreground="#e5edf7", font=("Segoe UI", 10))
        style.configure("Card.TLabel", background="#1b2638", foreground="#e5edf7")
        style.configure("Muted.TLabel", foreground="#a0aec0")
        style.configure("Title.TLabel", font=("Segoe UI", 23, "bold"))
        style.configure("Heading.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure(
            "Value.TLabel",
            background="#1b2638",
            foreground="#66d9d0",
            font=("Segoe UI", 21, "bold"),
        )
        style.configure(
            "TButton",
            padding=(12, 8),
            font=("Segoe UI", 10),
            foreground="#edf3fb",
            background="#2d3d54",
            borderwidth=0,
        )
        style.map(
            "TButton",
            background=[("active", "#3c536f"), ("disabled", "#243247")],
            foreground=[("disabled", "#7a899d")],
        )
        style.configure("Start.TButton", background="#16796e", font=("Segoe UI", 11, "bold"))
        style.map("Start.TButton", background=[("active", "#1c9588"), ("disabled", "#243247")])
        style.configure("TEntry", fieldbackground="#233148", foreground="#edf3fb", padding=7)
        style.configure(
            "TCombobox",
            fieldbackground="#233148",
            background="#2d3d54",
            foreground="#edf3fb",
            padding=7,
            arrowsize=14,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", "#233148")],
            foreground=[("readonly", "#edf3fb"), ("disabled", "#7a899d")],
        )
        style.configure("TCheckbutton", background="#111827", foreground="#e5edf7", padding=4)

        self.source = tk.StringVar(value=next(iter(SOURCES)))
        self.method = tk.StringVar(value=next(iter(METHODS)))
        self.duration = tk.StringVar(value="30")
        self.camera = tk.StringVar(value="0")
        self.port = tk.StringVar(value="5005")
        self.horizon = tk.StringVar(value="100")
        self.recording = tk.StringVar(
            value=str(self.repository / "outputs/cmu_expanded_20260924/13_01.npz")
            if (self.repository / "outputs/cmu_expanded_20260924/13_01.npz").is_file()
            else ""
        )
        self.live_config = tk.StringVar(value=str(self.repository / "configs/live/default.json"))
        self.processor_config = tk.StringVar()
        self.record_raw = tk.BooleanVar(value=False)
        self.unity_editor = tk.StringVar(
            value="F:/Unity/Editor/Unity.exe" if Path("F:/Unity/Editor/Unity.exe").is_file() else ""
        )
        self.status = tk.StringVar(value="Готово к запуску")
        self.detail = tk.StringVar(value="Выберите источник и нажмите «Запустить».")
        self.unity_status = tk.StringVar(
            value="Приём пока не подтверждён.\nВо время сессии проверим телеметрию Unity."
        )
        self.source_hint = tk.StringVar()
        self.method_hint = tk.StringVar()
        self.elapsed = tk.StringVar(value="00:00")
        self.demo_preset = tk.StringVar(value=next(iter(DEMO_PRESETS)))
        self.metrics = {name: tk.StringVar(value="—") for name in ("fps", "sent", "latency")}

        shell = ttk.Frame(root, padding=24)
        shell.pack(fill="both", expand=True)
        title = ttk.Frame(shell)
        title.pack(fill="x")
        ttk.Label(title, text="PoseAhead", style="Title.TLabel").pack(side="left")
        ttk.Label(title, text="ДВИЖЕНИЕ · ОБРАБОТКА · АВАТАР", style="Muted.TLabel").pack(
            side="left", padx=20
        )
        ttk.Label(title, textvariable=self.elapsed, style="Heading.TLabel").pack(side="right")
        ttk.Label(
            shell, text="Управление демонстрацией без командной строки", style="Muted.TLabel"
        ).pack(anchor="w", pady=(2, 16))
        demo_row = ttk.Frame(shell)
        demo_row.pack(fill="x", pady=(0, 12))
        ttk.Label(demo_row, text="Сценарий показа").pack(side="left", padx=(0, 12))
        self.demo_widget = ttk.Combobox(
            demo_row, textvariable=self.demo_preset, values=tuple(DEMO_PRESETS),
            state="readonly", width=48,
        )
        self.demo_widget.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.demo_button = ttk.Button(
            demo_row, text="Применить · без запуска", command=self.apply_demo,
        )
        self.demo_button.pack(side="left")
        self.inputs.extend((self.demo_widget, self.demo_button))
        cards = ttk.Frame(shell)
        cards.pack(fill="x", pady=(0, 16))
        for column, (key, label) in enumerate(
            (
                ("fps", "ПОЛНЫХ ПОЗ / С · КАМЕРА"),
                ("sent", "UDP-ПАКЕТОВ ОТПРАВЛЕНО"),
                ("latency", "CAPTURE → SEND · P95, МС"),
            )
        ):
            cards.columnconfigure(column, weight=1, uniform="metrics")
            card = ttk.Frame(cards, padding=14, style="Card.TFrame")
            card.grid(row=0, column=column, sticky="ew", padx=(0, 10 if column < 2 else 0))
            ttk.Label(card, text=label, style="Card.TLabel", font=("Segoe UI", 8)).pack(anchor="w")
            ttk.Label(card, textvariable=self.metrics[key], style="Value.TLabel").pack(anchor="w")
        body = ttk.Frame(shell)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3, uniform="body")
        body.columnconfigure(1, weight=2, uniform="body")
        body.rowconfigure(0, weight=1)
        controls = ttk.Frame(body)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 24))
        canvas = tk.Canvas(controls, bg="#111827", highlightthickness=0, width=580)
        canvas.pack(side="left", fill="both", expand=True)
        control_scroll = ttk.Scrollbar(controls, command=canvas.yview)
        control_scroll.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=control_scroll.set)
        left = ttk.Frame(canvas)
        control_window = canvas.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind(
            "<Configure>", lambda event: canvas.itemconfigure(control_window, width=event.width)
        )
        left.columnconfigure(0, weight=1)
        left.columnconfigure(1, weight=1)
        ttk.Label(left, text="Источник и обработка", style="Heading.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 10)
        )
        self.source_widget = self.combo(
            left, "Источник движения", self.source, tuple(SOURCES), 1, 0
        )
        self.method_widget = self.combo(left, "Метод обработки", self.method, tuple(METHODS), 1, 1)
        ttk.Label(left, textvariable=self.source_hint, style="Muted.TLabel", wraplength=560).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(6, 8)
        )
        self.recording_widget, self.recording_button = self.file_row(
            left, "Запись NPZ", self.recording, 4, "npz"
        )
        self.processor_widget, self.processor_button = self.file_row(
            left, "JSON процессора", self.processor_config, 6, "json"
        )
        self.duration_widget = self.entry(
            left, "Длительность, с · максимум 3600", self.duration, 8, 0
        )
        self.horizon_widget = self.combo(
            left, "Горизонт экстраполяции, мс", self.horizon, ("33", "66", "100", "150"), 8, 1
        )
        self.camera_widget = self.entry(left, "Индекс камеры", self.camera, 10, 0)
        self.port_widget = self.entry(left, "UDP-порт · 127.0.0.1", self.port, 10, 1)
        self.live_widget, self.live_button = self.file_row(
            left, "JSON камеры / MeTRAbs", self.live_config, 12, "json"
        )
        self.fast_camera_button = ttk.Button(
            self.live_widget.master, text="Быстрый профиль", command=self.apply_fast_camera,
        )
        self.fast_camera_button.pack(side="left", padx=(6, 0))
        self.inputs.append(self.fast_camera_button)
        self.record_checkbox = ttk.Checkbutton(
            left, text="Сохранять исходные наблюдения камеры в NPZ", variable=self.record_raw
        )
        self.record_checkbox.grid(row=14, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.inputs.append(self.record_checkbox)
        ttk.Label(left, textvariable=self.method_hint, style="Muted.TLabel", wraplength=550).grid(
            row=15, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )

        right_container = ttk.Frame(body)
        right_container.grid(row=0, column=1, sticky="nsew")
        right_canvas = tk.Canvas(right_container, bg="#111827", highlightthickness=0, width=390)
        right_canvas.pack(side="left", fill="both", expand=True)
        right_scroll = ttk.Scrollbar(right_container, command=right_canvas.yview)
        right_scroll.pack(side="right", fill="y")
        right_canvas.configure(yscrollcommand=right_scroll.set)
        right = ttk.Frame(right_canvas)
        right_window = right_canvas.create_window((0, 0), window=right, anchor="nw")
        right.bind("<Configure>", lambda _: right_canvas.configure(
            scrollregion=right_canvas.bbox("all"),
        ))
        right_canvas.bind("<Configure>", lambda event: right_canvas.itemconfigure(
            right_window, width=event.width,
        ))
        ttk.Label(right, text="Сессия", style="Heading.TLabel").pack(anchor="w", pady=(0, 10))
        ttk.Label(
            right, textvariable=self.status, foreground="#66d9d0", font=("Segoe UI", 14, "bold")
        ).pack(anchor="w")
        ttk.Label(right, textvariable=self.detail, wraplength=390, style="Muted.TLabel").pack(
            anchor="w", pady=(5, 10)
        )
        buttons = ttk.Frame(right)
        buttons.pack(fill="x", pady=(0, 8))
        self.start_button = ttk.Button(
            buttons, text="▶  Запустить", command=self.start, style="Start.TButton"
        )
        self.start_button.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.stop_button = ttk.Button(
            buttons, text="■  Остановить", command=self.stop, state="disabled"
        )
        self.stop_button.pack(side="left", fill="x", expand=True)
        self.force_button = ttk.Button(
            right, text="Принудительно завершить…", command=self.force_stop, state="disabled"
        )
        self.force_button.pack(fill="x", pady=(0, 14))
        ttk.Separator(right).pack(fill="x", pady=(0, 12))
        ttk.Label(right, text="Аватар в Unity", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(
            right,
            textvariable=self.unity_status,
            wraplength=390,
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(5, 8))
        unity_row = ttk.Frame(right)
        unity_row.pack(fill="x")
        ttk.Entry(unity_row, textvariable=self.unity_editor).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(
            unity_row, text="…", width=3, command=lambda: self.choose_file(self.unity_editor, "exe")
        ).pack(side="left", padx=(6, 0))
        ttk.Button(right, text="Открыть проект Unity", command=self.open_unity).pack(
            fill="x", pady=(8, 4)
        )
        ttk.Label(
            right,
            text="Сцена: PoseAheadDiagnostic 1\nВ Unity нажмите Play; порт должен совпадать.",
            style="Muted.TLabel",
            wraplength=390,
        ).pack(anchor="w", pady=(2, 10))
        self.folder_button = ttk.Button(
            right, text="Открыть результаты сессии", command=self.open_results, state="disabled"
        )
        self.folder_button.pack(fill="x")

        ttk.Label(shell, text="Журнал", style="Heading.TLabel").pack(anchor="w", pady=(14, 5))
        log_frame = ttk.Frame(shell)
        log_frame.pack(fill="x")
        self.log = tk.Text(
            log_frame,
            height=5,
            wrap="word",
            bg="#0c1422",
            fg="#b8c7db",
            relief="flat",
            padx=10,
            pady=8,
            font=("Consolas", 9),
            state="disabled",
        )
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        scrollbar.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scrollbar.set)
        self.restore_settings()
        self.source.trace_add("write", lambda *_: self.refresh_fields())
        self.method.trace_add("write", lambda *_: self.refresh_fields())
        self.refresh_fields()
        self.append_log("Панель не меняет настройки сети и не выбирает победителя экспериментов.")
        root.after(100, self.poll)

    def restore_settings(self):
        if self.preferences is None:
            return
        values, warning = self.preferences.load()
        for field, variable in (
            ("duration_seconds", self.duration), ("camera_index", self.camera),
            ("udp_port", self.port), ("horizon_ms", self.horizon),
            ("recording", self.recording), ("live_config", self.live_config),
            ("processor_config", self.processor_config), ("unity_editor", self.unity_editor),
        ):
            if field in values:
                variable.set(str(values[field]))
        for field, variable, labels in (
            ("source", self.source, SOURCES), ("method", self.method, METHODS),
        ):
            if field in values:
                variable.set(next(label for label, value in labels.items()
                                  if value == values[field]))
        # Recording is an explicit choice for each camera session, not persisted consent.
        self.record_raw.set(False)
        if warning:
            self.append_log(warning)
        elif values:
            self.append_log("Настройки восстановлены. Сессия и запись камеры не запущены.")

    def save_settings(self):
        if self.preferences is None:
            return
        try:
            options = self.options()
            values = {key: getattr(options, key) for key in FIELDS if key != "unity_editor"}
            self.preferences.save({**values, "unity_editor": self.unity_editor.get()})
        except (ValueError, OSError, TypeError) as error:
            self.append_log("Настройки панели не сохранены: " + str(error))

    def combo(self, parent, label, variable, values, row, column):
        title = self.ttk.Label(parent, text=label)
        title.grid(row=row, column=column, sticky="w", pady=(6, 3))
        widget = self.ttk.Combobox(
            parent, textvariable=variable, values=values, state="readonly", width=26
        )
        widget.grid(row=row + 1, column=column, sticky="ew", padx=(0, 10 if column == 0 else 0))
        self.inputs.append(widget)
        self.groups[widget] = (title, widget)
        return widget

    def entry(self, parent, label, variable, row, column):
        title = self.ttk.Label(parent, text=label)
        title.grid(row=row, column=column, sticky="w", pady=(6, 3))
        widget = self.ttk.Entry(parent, textvariable=variable)
        widget.grid(row=row + 1, column=column, sticky="ew", padx=(0, 10 if column == 0 else 0))
        self.inputs.append(widget)
        self.groups[widget] = (title, widget)
        return widget

    def file_row(self, parent, label, variable, row, extension):
        title = self.ttk.Label(parent, text=label)
        title.grid(row=row, column=0, columnspan=2, sticky="w", pady=(6, 3))
        frame = self.ttk.Frame(parent)
        frame.grid(row=row + 1, column=0, columnspan=2, sticky="ew")
        widget = self.ttk.Entry(frame, textvariable=variable)
        widget.pack(side="left", fill="x", expand=True)
        button = self.ttk.Button(
            frame, text="Выбрать", command=lambda: self.choose_file(variable, extension)
        )
        button.pack(side="left", padx=(6, 0))
        self.inputs.extend((widget, button))
        self.groups[widget] = (title, frame)
        return widget, button

    def choose_file(self, variable, extension):
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            parent=self.root,
            initialdir=self.repository,
            filetypes=[(extension.upper(), "*." + extension), ("Все файлы", "*.*")],
        )
        if path:
            variable.set(path)

    def refresh_fields(self):
        busy = self.controller.busy
        for widget in self.inputs:
            state = (
                "disabled"
                if busy
                else "readonly"
                if isinstance(widget, self.ttk.Combobox)
                else "normal"
            )
            widget.configure(state=state)
        source = SOURCES[self.source.get()]
        method = METHODS[self.method.get()]
        enabled = (
            (source == "recording", (self.recording_widget, self.recording_button)),
            (method == "config", (self.processor_widget, self.processor_button)),
            (
                source == "camera",
                (self.camera_widget, self.live_widget, self.live_button,
                 self.fast_camera_button, self.record_checkbox),
            ),
            (method == "kalman_cv", (self.horizon_widget,)),
        )
        for condition, widgets in enabled:
            for widget in widgets:
                for item in self.groups.get(widget, ()):
                    if condition:
                        item.grid()
                    else:
                        item.grid_remove()
            if not condition or busy:
                for widget in widgets:
                    widget.configure(state="disabled")
        if source != "camera":
            self.record_raw.set(False)
            self.record_checkbox.grid_remove()
        else:
            self.record_checkbox.grid()
        port_label, port_widget = self.groups[self.port_widget]
        port_label.grid_configure(row=10 if method == "kalman_cv" else 8)
        port_widget.grid_configure(row=11 if method == "kalman_cv" else 9)
        self.source_hint.set(
            {
                "diagnostic": "Восемь тестовых поз; камера и GPU не нужны. Не датасет для оценки.",
                "recording": "Оригинальная запись не меняется. Воспроизведение в реальном темпе.",
                "camera": "Нужны CUDA, установленный MeTRAbs и человек целиком в кадре.",
            }[source]
        )
        self.method_hint.set(
            "MLP — экспериментальный вариант. Есть шумные validation-результаты; "
            "снижение задержки и преимущество на live пока не доказаны."
            if method == "config"
            else "Метод меняется между сессиями. Горизонт относится только к экстраполяции."
        )
        self.start_button.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(
            state="normal" if busy and not self.controller.stop_sent else "disabled"
        )

    def options(self):
        try:
            duration = float(self.duration.get().replace(",", "."))
            camera, port, horizon = (
                int(self.camera.get()),
                int(self.port.get()),
                int(self.horizon.get()),
            )
        except ValueError as error:
            raise ValueError(
                "Проверьте числовые поля: длительность, камера, порт и горизонт."
            ) from error
        return SessionOptions(
            source=SOURCES[self.source.get()],
            method=METHODS[self.method.get()],
            duration_seconds=duration,
            camera_index=camera,
            udp_port=port,
            horizon_ms=horizon,
            recording=self.recording.get(),
            live_config=self.live_config.get(),
            processor_config=self.processor_config.get(),
            record_raw=self.record_raw.get(),
        )

    def apply_demo(self):
        from tkinter import messagebox

        if self.controller.busy:
            return
        try:
            preset = DEMO_PRESETS[self.demo_preset.get()]
            options = demo_options(self.repository, preset, udp_port=int(self.port.get()))
        except (KeyError, ValueError, OSError, TypeError) as error:
            messagebox.showerror("Сценарий недоступен", str(error), parent=self.root)
            return
        self.source.set(next(label for label, value in SOURCES.items() if value == options.source))
        self.method.set(next(label for label, value in METHODS.items() if value == options.method))
        self.duration.set(str(options.duration_seconds))
        self.horizon.set(str(options.horizon_ms))
        self.recording.set(options.recording)
        self.processor_config.set(options.processor_config)
        self.record_raw.set(False)
        self.detail.set(DEMO_NOTES[preset] + " Нажмите «Запустить», когда Unity готов.")
        self.append_log("Сценарий применён без запуска: " + DEMO_NOTES[preset])

    def apply_fast_camera(self):
        from tkinter import messagebox

        if self.controller.busy:
            return
        try:
            path = fast_camera_profile(self.live_config.get(), self.live_profile_directory)
        except (ValueError, OSError, TypeError) as error:
            messagebox.showerror("Быстрый профиль недоступен", str(error), parent=self.root)
            return
        self.live_config.set(str(path))
        self.append_log(
            "Выбран быстрый профиль MeTRAbs без запуска камеры. Калибровка и остальные "
            "параметры сохранены в отдельном JSON; исходный файл не меняется. "
            "Скорость проверена на записях, не на новой live-сессии."
        )

    def append_log(self, message):
        if not message:
            return
        self.log.configure(state="normal")
        self.log.insert("end", time.strftime("%H:%M:%S") + "  " + str(message)[:4096] + "\n")
        lines = int(self.log.index("end-1c").split(".")[0])
        if lines > 300:
            self.log.delete("1.0", f"{lines - 300}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def start(self):
        from tkinter import messagebox

        try:
            self.controller.start(self.options())
        except (ValueError, OSError, RuntimeError, TypeError) as error:
            messagebox.showerror("Не удалось запустить", str(error), parent=self.root)
            return
        self.error_seen = self.finished_seen = False
        self.unity_deadline = None
        self.stop_time = None
        self.last_metrics = {}
        for variable in self.metrics.values():
            variable.set("—")
        self.status.set("Подготовка сессии")
        self.unity_status.set("Новая сессия: приём пока не подтверждён.")
        self.detail.set("Источник работает в отдельном процессе; интерфейс остаётся доступным.")
        self.folder_button.configure(state="normal")
        self.force_button.configure(state="disabled")
        self.append_log("Запуск. Результаты: " + str(self.controller.output_directory))
        self.save_settings()
        self.refresh_fields()

    def stop(self):
        self.controller.request_stop()
        self.stop_time = time.monotonic()
        self.status.set("Остановка…")
        self.detail.set(
            "Ждём освобождения камеры и сохранения NPZ. Загрузка CUDA может завершаться дольше."
        )
        self.refresh_fields()

    def force_stop(self):
        from tkinter import messagebox

        if messagebox.askyesno(
            "Принудительная остановка",
            "Остановить только процесс этой сессии?\nНесохранённые наблюдения могут быть потеряны.",
            parent=self.root,
        ):
            self.controller.force_stop()
            self.append_log("Запрошена принудительная остановка собственного producer-процесса.")

    def update_metrics(self, event):
        self.last_metrics.update(event)
        if "valid_pose_fps" in event:
            self.metrics["fps"].set(f"{event['valid_pose_fps']:.1f}")
        count = event.get(
            "sent_frames", event.get("capture_available_to_send_complete_ms", {}).get("samples")
        )
        if count is not None:
            self.metrics["sent"].set(str(count))
        latency = event.get("capture_available_to_send_complete_ms", {}).get("p95")
        if latency is not None:
            self.metrics["latency"].set(f"{latency:.1f}")

    def handle_event(self, event):
        """Apply one worker event; independent of Tk timers and queue draining."""
        kind = event.get("kind")
        if kind == "state":
            self._on_state(event)
        elif kind == "progress":
            self._on_progress(event)
        elif kind == "finished":
            self._on_finished(event)
        elif kind == "error":
            self._on_error(event)
        elif kind == "exit":
            self._on_exit(event)
        elif kind == "log":
            self._on_log(event)

    def _on_state(self, event):
        if not self.controller.stop_sent:
            self.status.set(
                {
                    "loading": "Загрузка",
                    "warmup": "Прогрев",
                    "running": "Сессия работает",
                }.get(event.get("state"), "Сессия")
            )
            self.detail.set(event.get("message", ""))
        self.append_log(event.get("message", ""))

    def _on_progress(self, event):
        self.update_metrics(event)
        if "unity" in event:
            self.update_unity_status(event["unity"], live=True)
        stages = event.get("pipeline_latency_ms", {})
        stage_labels = (
            ("frame_age_before_inference_ms", "до 3D"),
            ("backend_return_ms", "MeTRAbs"),
            ("recording_and_processing_ms", "обработка"),
        )
        values = [f"{label}: {stages[key]['median']:.1f} мс" for key, label in stage_labels
                  if stages.get(key, {}).get("median") is not None]
        if values and not self.controller.stop_sent:
            self.detail.set("Медианы этапов · " + " · ".join(values))
        if (
            "valid_pose_frames" in event
            and event["valid_pose_frames"] == 0
            and event.get("processed_frames", 0) > 0
            and not self.controller.stop_sent
        ):
            self.detail.set("Кадры идут, но полной позы нет. Покажите человека целиком в кадре.")

    def _on_finished(self, event):
        self.finished_seen = True
        report = event["report"]
        self.update_metrics(report)
        self.status.set(
            "Остановлено" if report.get("status", "").startswith("stopped") else "Завершено"
        )
        self.update_unity_status(event.get("run_report", {}).get("unity", {}))
        self.detail.set("Сохранены report.json и единый run_report.json.")
        self.append_log("Сессия завершена; отчёты сохранены.")

    def update_unity_status(self, unity, *, live=False):
        self.unity_deadline = None
        if unity.get("status") == "observed":
            fps = unity.get("render_fps")
            rendered = f"{fps:.1f} FPS" if fps is not None else "FPS не измерен"
            text = (
                f"Unity: принято {unity['received_pose_frames']}, "
                f"применено {unity['applied_pose_frames']}.\n{rendered}."
            )
            errors = [str(unity[key])[:300] for key in ("receiver_error", "humanoid_error")
                      if unity.get(key)]
            if errors:
                text += "\nОшибка Unity: " + "; ".join(errors)
            elif not unity.get("receiving", True):
                text += "\nReceiver не принимает новые позы."
            if live:
                # Account for snapshot age, not merely when Tk received the event.
                age_ms = max(0.0, time.time() * 1000 - unity["written_unix_ms"])
                self.unity_deadline = time.monotonic() + max(0.0, (2000 - age_ms) / 1000)
            self.unity_status.set(text)
        else:
            self.unity_status.set(
                "Нет свежей телеметрии этой сессии.\nОткройте диагностическую сцену и нажмите Play."
            )

    def expire_unity_status(self):
        if (self.unity_deadline is not None and not self.finished_seen
                and time.monotonic() >= self.unity_deadline):
            self.unity_deadline = None
            self.unity_status.set(
                "Нет свежей телеметрии этой сессии.\n"
                "Unity перестал подтверждать приём; проверьте Play и порт."
            )

    def _on_error(self, event):
        self.error_seen = True
        self.status.set("Ошибка сессии")
        self.detail.set(str(event.get("message", "Неизвестная ошибка")))
        self.append_log("ОШИБКА: " + self.detail.get())

    def _on_exit(self, event):
        if event["code"] != 0:
            had_error_message = self.error_seen
            self.error_seen = True
            self.status.set("Ошибка / процесс завершён")
            if not self.finished_seen and not had_error_message:
                self.detail.set(
                    "Процесс завершился с кодом " + str(event["code"]) + ". Подробности в журнале."
                )
        elif not self.finished_seen:
            self.status.set("Нет итогового отчёта")
            self.detail.set("Проверьте журнал. Успех сессии не подтверждён.")
        self.append_log("Код завершения: " + str(event["code"]))

    def _on_log(self, event):
        self.append_log(event.get("message", ""))

    def poll(self):
        try:
            while True:
                self.handle_event(self.controller.events.get_nowait())
        except queue.Empty:
            pass
        busy = self.controller.busy
        self.expire_unity_status()
        if busy and self.controller.started_at is not None:
            seconds = int(time.monotonic() - self.controller.started_at)
            self.elapsed.set(f"{seconds // 60:02}:{seconds % 60:02}")
        self.refresh_fields()
        self.force_button.configure(
            state="normal"
            if busy and self.stop_time is not None and time.monotonic() - self.stop_time >= 5
            else "disabled"
        )
        if self.closing and not busy:
            self.save_settings()
            self.root.destroy()
            return
        self.root.after(100, self.poll)

    def open_unity(self):
        from tkinter import messagebox

        editor = Path(self.unity_editor.get())
        project = self.repository / "unity/PoseAheadAvatar"
        if not editor.is_file() or editor.suffix.lower() != ".exe":
            messagebox.showerror(
                "Unity Editor",
                "Выберите Unity.exe закреплённой версии 6000.3.23f1.",
                parent=self.root,
            )
            return
        if (project / "Temp/UnityLockfile").exists():
            messagebox.showinfo(
                "Unity уже открыт",
                "Перейдите в уже открытый Unity и нажмите Play.\n"
                "Панель не закрывает чужие окна и не запускает второй Editor для проекта.",
                parent=self.root,
            )
            return
        try:
            command = unity_launch_command(self.repository, editor)
            subprocess.Popen(command, cwd=self.repository)
            self.append_log("Запущен Unity Editor. Откройте диагностическую сцену и нажмите Play.")
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            messagebox.showerror("Не удалось открыть Unity", str(error), parent=self.root)

    def open_results(self):
        from tkinter import messagebox

        if self.controller.output_directory is not None:
            try:
                if os.name == "nt":
                    os.startfile(self.controller.output_directory)
                else:
                    messagebox.showinfo(
                        "Результаты", str(self.controller.output_directory), parent=self.root
                    )
            except OSError as error:
                messagebox.showerror("Результаты", str(error), parent=self.root)

    def close(self):
        from tkinter import messagebox

        if self.controller.busy:
            if self.closing:
                self.force_stop()
                return
            if not messagebox.askyesno(
                "Завершить работу",
                "Остановить сессию и закрыть панель после сохранения?\nUnity останется открытым.",
                parent=self.root,
            ):
                return
            self.closing = True
            self.stop()
        else:
            self.save_settings()
            self.root.destroy()


def run_gui(repository=None):
    import tkinter as tk

    repository = Path(repository) if repository else Path(__file__).resolve().parents[3]
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError("Не удалось открыть desktop-окно: " + str(error)) from error
    DesktopApp(root, repository)
    root.mainloop()
    return 0
