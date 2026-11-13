# Техническое задание

## Система предиктивной компенсации задержки для видеоуправляемого полнотелого аватара

**Тип работы:** НИР с последующим развитием в ВКР  
**Направление:** 01.03.02 «Прикладная математика и информатика»  
**Целевая среда визуализации:** Unity  
**Основной стек исследовательской части:** Python  
**Статус:** рабочее ТЗ, ориентированное на реалистичный объём бакалаврской ВКР

---

# 1. Назначение проекта

Разработать программно-экспериментальную систему управления полнотелым виртуальным аватаром по видеопотоку с одной RGB-камеры и исследовать методы предиктивной компенсации задержки.

Система должна одновременно выполнять две функции:

1. обеспечивать real-time управление humanoid-аватаром;
2. служить воспроизводимой экспериментальной платформой для сравнения методов temporal processing и prediction.

Главная задача ВКР — не создание нового pose detector или нового 3D-движка, а исследование компромисса:

```text
effective latency <-> pose error <-> jitter
```

---

# 2. Цель НИР

В рамках НИР необходимо построить рабочую инфраструктуру исследования:

- видеозахват;
- оценка полнотелой позы;
- каноническое представление позы;
- передача позы в Unity;
- управление humanoid-аватаром;
- запись и воспроизведение последовательностей;
- искусственное моделирование задержки;
- baseline-методы фильтрации и экстраполяции;
- первый fixed-horizon predictor;
- первые объективные эксперименты.

Желательно начать реализацию adaptive prediction, но полная экспериментальная проработка может быть перенесена в ВКР.

---

# 3. Цель ВКР

Основная цель ВКР:

> разработать и экспериментально оценить адаптивный метод предиктивной компенсации задержки для видеоуправляемого полнотелого виртуального аватара при шумном монокулярном RGB-входе.

Основной исследовательский вопрос:

> позволяет ли прогноз позы вперёд уменьшить effective latency без неприемлемого роста ошибки и jitter?

Второй исследовательский вопрос:

> даёт ли адаптивное изменение горизонта прогноза и/или степени доверия прогнозу преимущество относительно фиксированного прогноза?

---

# 4. Научное позиционирование

Нельзя заявлять как самостоятельную новизну:

- сам факт использования human motion prediction для компенсации задержки;
- прогноз полного тела;
- prediction при фиксированной задержке;
- prediction при произвольной или переменной задержке;
- predictability-aware prediction как абстрактную идею.

Рабочая исследовательская дельта проекта:

```text
noisy monocular RGB input
+
full-body pose
+
real-time avatar control
+
predictability-aware adaptive compensation
+
joint evaluation of latency / jitter / accuracy
```

Научная ценность должна подтверждаться экспериментально, а не только архитектурой системы.

---

# 5. Обязательное ядро ВКР

Следующие части считаются обязательными.

## 5.1. Full-body live pipeline

```text
RGB Camera
    ->
Pose Estimation
    ->
3D Pose
    ->
Temporal Processing
    ->
Unity
    ->
Humanoid Avatar
```

Критерий:

> аватар в Unity стабильно повторяет основные движения пользователя в реальном времени.

Принятый backend для live pipeline: `MeTRAbs EfficientNetV2-L PyTorch` с
зафиксированными source revision и pretrained checkpoints. Он формирует
абсолютные 3D keypoints напрямую из текущего RGB-кадра. В live benchmark запрещён
temporal refinement, использующий будущие кадры. MotionBERT/VideoPose3D можно
добавлять в offline comparison только после проверки temporal context на причинность.

---

## 5.2. Каноническое представление позы

Должна существовать независимая от Unity структура `PoseFrame`.

Минимально:

```text
frame_id
source_timestamp_ns
produced_timestamp_ns

root_position
root_rotation

joint_positions
joint_rotations

joint_confidence
joint_validity
metadata
```

`source_timestamp_ns` задаёт время захвата/события, а
`produced_timestamp_ns` — момент формирования данной версии кадра. Оба значения
должны быть монотонными и принадлежать одному документированному clock domain.
Wall-clock timestamp при необходимости хранится только в metadata. Live- и
offline-логика работают по event time, а не по номеру кадра.

Каноническая система координат — правая camera coordinate system в метрах:
`+X` направлен вправо в изображении, `+Y` вниз, `+Z` от камеры.
`joint_positions` задаются относительно pelvis, `pelvis == (0, 0, 0)`.
Валидный `root_position` задаёт положение pelvis в camera frame. Преобразование
в Unity: `(x, y, z) -> (x, -y, z)`, затем применяются avatar scale и origin
calibration.

Позиции суставов обязательны. `root_position`, `root_rotation` и joint rotations
могут отсутствовать, если монокулярный estimator не восстанавливает их надёжно;
отсутствие отмечается явно, а не маскируется нулями или identity quaternion.
MeTRAbs выдаёт оценку абсолютных H36M-17 координат в миллиметрах; pelvis
передаётся как `root_position`, а скелет приводится к pelvis-relative виду.
Оценка помечается `estimated_monocular`, включая флаг в UDP, и не считается
измеренным global position. Intrinsics могут быть измеренными; иначе допущение о
поле зрения фиксируется в metadata. У модели нет per-joint confidence, поэтому
confidence детектора используется только как явно помеченный proxy. Удалённые
способы по точкам пола/опорной стопе и длине корпуса не являются fallback.
Бег и прыжки не блокируются логикой контакта, но их точность должна быть отдельно
проверена на движении и ground truth. Подробности: `docs/metrabs.md`.
Первый pipeline прогнозирует и передаёт позиции. Unity вычисляет bone rotations
по parent-child directions и откалиброванной T-pose аватара. Ненаблюдаемый twist
вокруг оси кости задаётся rest pose либо документированным pole-vector rule и не
считается измеренным ground truth.

Минимальный канонический скелет:

```text
pelvis
spine
chest
neck
head

left_shoulder
left_elbow
left_wrist

right_shoulder
right_elbow
right_wrist

left_hip
left_knee
left_ankle

right_hip
right_knee
right_ankle
```

Обязательно документировать:

- coordinate system;
- units;
- handedness;
- root convention;
- skeleton parent hierarchy и rest-pose convention;
- joint order;
- local/global transform convention;
- quaternion convention.

---

## 5.3. Recording / Replay

Система должна уметь записывать и воспроизводить последовательности без камеры и Unity.

Для каждого кадра сохранять минимум:

```text
frame_id
timestamps
2D pose
confidence
3D pose
filtered pose
predicted pose
final output pose
active algorithm
prediction horizon
processing timings
```

Также сохраняются семантика часов и, при наличии, ground-truth pose с её
timestamp. Исходные результаты detector/lifter должны оставаться неизменными;
filtered/predicted/final outputs сохраняются как версионированные результаты
конкретного запуска, чтобы запись можно было переобработать другим методом.

Offline replay должен использовать те же алгоритмы, что и live pipeline.

---

## 5.4. Artificial Latency

Обязательные значения:

```text
33 ms
66 ms
100 ms
150 ms
```

`artificial_latency_ms` вводится после формирования pose observation и перед
temporal processor. Это модель доступности наблюдения, а не времени работы
детектора или задержки дисплея.

При задержке `L` алгоритм в момент оценки `t_eval` не должен видеть наблюдения с
`source_timestamp` позже:

```text
t_eval - L
```

Правило реализуется по timestamps и должно работать при переменном FPS и
пропусках кадров. Интерполяция ground truth на точный момент допустима только в
evaluator и не должна попадать в processor.

Необходимо различать `artificial_latency_ms` и `target_offset_ms`: первое
задаёт момент доступности наблюдения, второе — целевой горизонт относительно
последнего доступного наблюдения. Метод может компенсировать всю задержку, её
часть или не компенсировать её вовсе. Необходимо исключить future-frame leakage.

---

## 5.5. Baseline-методы

Обязательные:

1. Raw delayed pose.
2. EMA.
3. One-Euro Filter.
4. Kalman Filter.
5. Kalman constant-velocity extrapolation.
6. Fixed-horizon learned predictor.

Все методы должны получать один и тот же причинно доступный поток наблюдений и
оцениваться относительно ground truth в одни и те же целевые моменты времени.
Гиперпараметры подбираются на train/validation, а не на тестовых
последовательностях, по которым публикуется результат.

---

## 5.6. Основной метод ВКР

Adaptive prediction / adaptive compensation.

Метод должен использовать оценку предсказуемости движения.

Допустимые сигналы:

- recent prediction error;
- joint velocity;
- acceleration;
- jerk;
- direction changes;
- detector confidence;
- model uncertainty;
- ensemble disagreement.

Любой сигнал должен быть причинно доступен в момент принятия решения. В
частности, `recent prediction error` может использовать только прогнозы, для
которых целевое наблюдение уже поступило, но не текущий или будущий ground truth.

Концептуально:

```text
predictable motion
    ->
longer prediction horizon
and/or higher trust in prediction

unpredictable motion
    ->
shorter prediction horizon
and/or lower trust in prediction
```

Допустимая форма blending:

```text
P_out = alpha * P_predicted + (1 - alpha) * P_measured
```

Для обоих слагаемых должны быть заданы pose timestamps. Если они относятся к
разным моментам, смешивание считается намеренной частичной компенсацией
задержки, а не объединением двух оценок одного состояния. Позиции можно
интерполировать линейно; для rotations нужна нормализованная quaternion
interpolation (например, SLERP), а не покомпонентное усреднение.

Формула не считается фиксированной частью ТЗ.

---

# 6. Предсказатель

Первый learned predictor должен быть:

- достаточно лёгким для real-time;
- воспроизводимым;
- конфигурируемым по горизонту;
- пригодным для offline benchmark;
- не обязательно SOTA.

Минимальные горизонты:

```text
33 ms
66 ms
100 ms
150 ms
```

Интерфейс predictor должен поддерживать явное указание целевого временного смещения, например:

```text
target_offset_ms
```

Чтобы архитектура не была привязана только к четырём фиксированным значениям.

Используется одна compact architecture и четыре отдельно обученных checkpoint —
для `33`, `66`, `100` и `150 ms`. Архитектура, длина истории, preprocessing,
training protocol и seeds одинаковы. `0 ms` означает отсутствие predictor.
Неподдерживаемый offset отклоняется явно. Adaptive method выбирает один из
четырёх checkpoint; continuous interpolation является отдельным optional
экспериментом.

---

# 7. Offline benchmark

Offline-режим является обязательной частью ВКР.

Схема:

```text
dataset / recorded sequence
    ->
optional synthetic camera noise
    ->
artificial latency
    ->
selected method
    ->
output sequence
    ->
metrics
    ->
plots / tables
```

Unity не должен требоваться для вычисления результатов.

---

# 8. Данные

Обязательный минимум:

- один публичный motion dataset;
- собственные записи с камеры.

Обязательный основной benchmark:

- Human3.6M.

Разбиение для predictor до нарезки окон:

```text
train: S1, S5, S6, S7
validation: S8
test: S9, S11
```

AMASS остаётся `SHOULD` как второй датасет и не блокирует первый полный
benchmark.

Для ВКР желательно иметь минимум один mocap-источник ground truth и реальные webcam-последовательности для проверки живого pipeline.

Экспериментально следует разделять два трека:

1. **motion-prediction:** mocap/3D ground truth (например, AMASS) и при
   необходимости синтетические искажения pose observations;
2. **camera-pipeline:** синхронизированные RGB/detector outputs и 3D ground truth
   там, где они доступны (например, Human3.6M), плюс собственные webcam-записи
   для live-валидации.

AMASS содержит mocap-derived движения, а не монокулярные RGB-кадры. Поэтому
искажение суставов AMASS следует называть шумом pose observations, а не шумом
изображения камеры; само по себе оно не подтверждает выводы о RGB detector.
Необходимо документировать лицензии и способ получения данных и не коммитить в
репозиторий данные с ограниченной лицензией.

Разбиение данных для learned models выполняется по субъектам и исходным
последовательностям до нарезки окон. Манифесты split сохраняются; пересекающиеся
окна одной последовательности не могут попадать в разные train/validation/test.

---

# 9. Модель шума

Для полноценной ВКР желательно моделировать шум pose observations, характерный
для камерного pipeline. Image-level искажения RGB являются отдельным, более
дорогим экспериментом и не подменяются добавлением шума к суставам.

Минимально допустимые искажения:

```text
Gaussian joint noise
outliers
missing joints
temporary tracking loss
confidence degradation
```

Параметры шума желательно оценивать по реальному детектору.

Начальные stress-test уровни применяются по одному семейству искажений поверх
clean condition:

```text
Gaussian joint sigma: 10 / 25 / 50 mm
outlier probability: 1% / 5%, displacement magnitude: 150 mm
missing-joint probability: 5% / 15% per frame
tracking-gap duration: 100 / 250 ms
```

Для missing joints и gaps confidence равен нулю. Значения можно скорректировать
после измерения residuals MeTRAbs на train/validation, но до просмотра test
results они фиксируются. Полный Cartesian product уровней не обязателен.

---

# 10. Метрики

Все временные метрики используют физическое время из timestamps. Если метрике
нужна равномерная частота, последовательность явно resample-ится с сохранением
частоты и способа интерполяции в конфигурации результата.

## 10.1. Pose Accuracy

Обязательная метрика:

```text
MPJPE
```

Протокол расчёта указывается явно. Как минимум отдельно считаются root-relative
MPJPE и ошибка global/root translation. Нельзя обозначать одним именем raw
MPJPE, scale-aligned MPJPE и Procrustes-aligned MPJPE. Перед расчётом скелет
датасета отображается в canonical skeleton, а прогноз сравнивается с ground
truth в заявленный target timestamp.

Дополнительно:

- per-joint error;
- body-region error;
- orientation error;
- root error.

---

## 10.2. Jitter

Обязательна минимум одна воспроизводимая метрика:

- mean joint acceleration;
- jerk;
- high-frequency energy.

Обычная acceleration/jerk измеряет также намеренное быстрое движение и может
искусственно награждать чрезмерно сглаженный, но неточный метод. Поэтому основной
jitter score должен быть ошибкой производных относительно ground truth или явно
описанным high-frequency residual; простую acceleration/jerk можно показывать
как дополнительную smoothness-метрику. Формула должна быть явно описана в ВКР.

---

## 10.3. Effective Latency

Основной вариант:

```text
cross-correlation lag
```

Результат — в миллисекундах.

Необходимо зафиксировать preprocessing сигнала, допустимые moving windows,
диапазон lag, знак, sub-frame interpolation и агрегацию по суставам и
последовательностям. Статические и выраженно периодические окна могут не давать
однозначной оценки: их следует помечать невалидными и сообщать coverage. Знак и
величина проверяются на синтетических сигналах с известной задержкой.

---

## 10.4. Runtime performance

Измерять:

```text
camera capture
2D pose estimation
3D stage
filtering
prediction
transport
Unity receive/apply
total pipeline time
```

Количественные критерии live pipeline на reference machine:

- **real-time:** минимум 25 валидных выходных `PoseFrame`/с в течение 10 минут,
  целевое значение — 30 FPS;
- **processing budget:** p95 от доступности захваченного кадра до UDP send не
  более 50 ms без учёта настроенной artificial latency;
- **stable:** нет crash, NaN/Inf pose и бесконечно растущей очереди; latest-frame
  queue имеет глубину не более одного кадра; минимум 95% захваченных кадров дают
  валидную pose либо явно учитываются как dropped/invalid.

Начальная reference machine: Windows 11 x64, AMD Ryzen 7 5700X, NVIDIA GeForce
RTX 3070 Ti 8 GB, 16 GB RAM. Для каждого публикуемого запуска всё равно
сохраняются фактические CPU, GPU, RAM, OS, driver и package versions.

---

# 11. Обязательные эксперименты ВКР

## E1. Влияние prediction horizon

Для fixed predictor построить зависимость:

```text
horizon
->
MPJPE
jitter
effective latency
```

Для каждого sweep необходимо указать, какая artificial latency зафиксирована.
Как минимум для каждого обязательного значения задержки сравниваются режим без
прогноза (`target_offset_ms = 0`) и matched-режим
`target_offset_ms = artificial_latency_ms`. При наличии ресурсов желательна
более полная сетка offset/latency.

---

## E2. Плавные и резкие движения

Использовать окна длиной 1 s и motion score, равный 95-му percentile модуля
root-relative acceleration по всем non-root joints. Перед дифференцированием
ground-truth trajectory сглаживается одним фиксированным документированным
low-pass filter. Thresholds вычисляются только по training split:

```text
smooth: score <= training 30th percentile
medium: training 30th < score < training 70th percentile
high-dynamic/abrupt: score >= training 70th percentile
```

Показать, насколько различается поведение fixed prediction.

Thresholds фиксируются до evaluation. Класс называется
`high-dynamic/abrupt`, а не `unpredictable`: резкость наблюдаема, а
предсказуемость зависит от модели. Нельзя классифицировать тестовый сегмент по
ошибке сравниваемого метода.

---

## E3. Baseline comparison

Сравнить:

```text
Raw
EMA
One-Euro
Kalman
Kalman extrapolation
Fixed learned predictor
Adaptive method
```

---

## E4. Adaptive vs Fixed

Главный эксперимент ВКР.

Проверить:

> существует ли область параметров / классов движений, в которой adaptive method даёт лучший компромисс latency / jitter / accuracy.

Fixed comparator и thresholds выбираются только по validation. На одинаковых
held-out последовательностях публикуются per-sequence результаты и агрегированная
неопределённость (например, confidence interval по субъектам/последовательностям),
а не только одно объединённое среднее.

Единственного безусловного победителя по трём метрикам нет: публикуется Pareto
front. Метод можно назвать лучшим в operating point, только если:

1. он не хуже по effective latency, root-relative MPJPE и primary jitter с
   учётом 95% sequence-bootstrap confidence intervals и строго лучше минимум по
   одной метрике; либо
2. он уменьшает effective latency минимум на 20 ms, а MPJPE и primary jitter
   растут не более чем на 10% относительно выбранного по validation fixed
   comparator.

Рост MPJPE или primary jitter более чем на 10% — рабочее определение
`unacceptable`. После pilot на train/validation пороги можно изменить, но до
просмотра test results они должны быть заморожены.

---

## E5. Ablation

Минимум одна абляция.

Например:

```text
adaptive horizon only
vs
adaptive blending only
vs
adaptive horizon + blending
```

или:

```text
without predictability features
vs
with predictability features
```

---

## E6. Real camera validation

Проверить работу метода на реальном webcam pipeline.

Не обязательно иметь ground truth такой же точности, как у mocap benchmark.

Задача — подтвердить перенос метода в live-систему.

Без синхронизированного ground truth этот эксперимент подтверждает runtime
feasibility и качественное поведение, но не является количественным
доказательством большей точности одного метода относительно другого.

---

# 12. Live-demo

Целевая среда: Unity 6.3 LTS `6000.3.23f1`, Windows x64. Версия фиксируется в
`ProjectVersion.txt`; обновление требует отдельного compatibility run и во время
экспериментальной стадии не выполняется неявно.

Unity-приложение должно поддерживать переключение методов без полной перезагрузки.

Желательный интерфейс:

```text
1 - Raw
2 - EMA
3 - One-Euro
4 - Kalman
5 - Fixed Prediction
6 - Adaptive Prediction
```

Показывать:

```text
FPS
active algorithm
configured latency
prediction horizon
predictability score
alpha
processing time
```

Injected observation latency, prediction offset, measured processing latency и
estimated effective latency выводятся отдельными полями; одно общее поле
`latency` недостаточно. Поскольку research processors находятся в Python,
переключение метода в Unity должно отправлять команду Python или выбирать один
из уже вычисленных Python-потоков, а не дублировать алгоритмы в Unity.

Желательно:

```text
LEFT  - baseline
RIGHT - proposed method
```

с одинаковой входной последовательностью.

---

# 13. End-to-end latency

Нужно различать:

1. внутренние timing-метрики pipeline;
2. реальную motion-to-avatar latency.

Sender timestamps передаются в monotonic clock domain отправителя, а Unity
фиксирует receive/apply в своём monotonic clock domain. One-way transport
latency разрешено получать простым вычитанием только после синхронизации или
калибровки часов; иначе отдельно сообщаются sender-side processing,
receiver-side handling, round-trip либо внешняя end-to-end оценка.

Для внешней проверки желательно использовать высокоскоростную запись пользователя и экрана:

```text
120-240 FPS
```

Это усиляет ВКР, но сложный автоматизированный стенд для этого не обязателен.

---

# 14. Граница обязательного и дополнительного

## MUST — обязательно для хорошей ВКР

- [ ] full-body real-time pipeline;
- [ ] 3D pose representation;
- [ ] Unity humanoid avatar;
- [ ] versioned binary UDP transport;
- [ ] recording/replay;
- [ ] canonical pose format;
- [ ] artificial latency;
- [ ] Raw, EMA, One-Euro и Kalman baselines;
- [ ] Kalman extrapolation;
- [ ] fixed learned predictor;
- [ ] adaptive method;
- [ ] MPJPE;
- [ ] jitter metric;
- [ ] effective latency metric;
- [ ] минимум один публичный benchmark dataset с ground truth;
- [ ] offline benchmark;
- [ ] fixed vs adaptive experiments;
- [ ] per-motion-class analysis с заранее заданным правилом классификации;
- [ ] минимум одна ablation;
- [ ] reproducible configs;
- [ ] live demo.

## SHOULD — желательно

- [ ] detector-residual-based pose-observation noise model;
- [ ] AMASS как второй публичный датасет;
- [ ] latency-aware learned baseline;
- [ ] сравнение разных predictability features;
- [ ] external 120-240 FPS latency measurement;
- [ ] two-avatar A/B live demo;

## COULD — только если остаётся время

- [ ] user perception study;
- [ ] uncertainty ensemble;
- [ ] новый neural architecture;
- [ ] sophisticated IK;
- [ ] foot locking;
- [ ] multiple pose detectors;
- [ ] multiple 3D lifting models;
- [ ] hand tracking;
- [ ] facial tracking;
- [ ] multi-camera setup.

Элементы `COULD` не должны блокировать завершение ВКР.

---

# 15. Что не является целью работы

Не требуется:

- собственный 2D pose detector;
- собственная фундаментальная 3D pose architecture;
- SOTA human motion prediction;
- production-ready avatar SDK;
- фотореалистичный Unity renderer;
- perfect IK;
- finger tracking;
- facial animation;
- VR;
- multiplayer;
- cloud deployment.

Готовые pose-estimation решения рассматриваются как инфраструктура исследования.

---

# 16. Архитектурное разделение

```text
Python
|
+-- camera
+-- pose estimation
+-- 3D pose
+-- filtering
+-- prediction
+-- adaptive logic
+-- recording
+-- experiments
+-- metrics
+-- plots
|
+---- versioned binary UDP ----> Unity
                     |
                     +-- receiver
                     +-- retargeting
                     +-- humanoid runtime
                     +-- visualization
```

Исследовательская логика не должна зависеть от Unity.

Начальный wire protocol — custom binary UDP, little-endian, один `PoseFrame` на
datagram размером не более 1200 bytes. В `docs/unity_protocol.md` фиксируются
magic bytes, version, flags, numeric types, joint count/order и compatibility
policy. OSC и TCP в первую реализацию не входят; достаточно Python `struct` и
стандартных binary/span primitives C#.

---

# 17. Этапы работ

## Этап 1 — NIR infrastructure

```text
Camera
-> Pose
-> 3D Pose
-> Canonical PoseFrame
-> Unity
-> Avatar
```

Плюс timestamps и базовое логирование.

---

## Этап 2 — Recording / Offline

Добавить:

- запись;
- replay;
- experiment configs.

---

## Этап 3 — Baselines

Реализовать:

- Raw;
- EMA;
- One-Euro;
- Kalman;
- extrapolation.

---

## Этап 4 — Latency benchmark

Добавить:

- artificial delay;
- MPJPE;
- jitter;
- effective latency.

---

## Этап 5 — Fixed predictor

Обучить/подключить compact learned predictor.

Провести horizon experiment.

---

## Этап 6 — Adaptive method

Реализовать predictability-aware adaptation.

---

## Этап 7 — VKR experiments

Провести:

- baseline comparison;
- smooth vs abrupt analysis;
- adaptive vs fixed;
- ablation;
- real-camera validation.

---

## Этап 8 — Final demo and thesis materials

Подготовить:

- Unity live demo;
- итоговые графики;
- таблицы;
- архитектурные схемы;
- воспроизводимый код;
- инструкции запуска.

---

# 18. Definition of Done для НИР

НИР считается состоявшейся, если:

- [ ] работает webcam full-body pipeline;
- [ ] есть Unity avatar;
- [ ] определён PoseFrame;
- [ ] есть запись/replay;
- [ ] реализованы baseline temporal methods;
- [ ] есть artificial latency;
- [ ] есть первые метрики;
- [ ] есть fixed predictor или его рабочий прототип.

Adaptive method допускается как начатая часть, переходящая в ВКР.

---

# 19. Definition of Done для ВКР

ВКР считается технически завершённой, если:

- [ ] работает стабильный live pipeline;
- [ ] существует offline benchmark;
- [ ] baseline-методы сравниваются на одинаковом входе;
- [ ] реализован learned fixed predictor;
- [ ] реализован adaptive method;
- [ ] рассчитаны MPJPE, jitter и effective latency;
- [ ] проведена оценка минимум на одном публичном benchmark dataset с ground truth;
- [ ] проведён главный adaptive-vs-fixed experiment;
- [ ] проведена минимум одна ablation;
- [ ] есть результаты на реальном webcam pipeline;
- [ ] эксперименты запускаются из конфигураций;
- [ ] результаты воспроизводимы;
- [ ] есть Unity live demo;
- [ ] подготовлены графики и таблицы для текста ВКР.

---

# 20. Приоритет разработки

При выборе между дополнительной функциональностью и качеством исследования использовать порядок:

```text
1. correctness
2. reproducibility
3. experiment quality
4. measurable latency
5. comparable baselines
6. adaptive method
7. live demo
8. visual polish
```

Главная ошибка, которой следует избегать:

> превратить ВКР в большой Unity-проект и не успеть провести нормальные эксперименты.

Главный итог ВКР — не «аватар двигается», а экспериментально обоснованный вывод о применимости предиктивной и адаптивной компенсации задержки.
