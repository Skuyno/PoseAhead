"""Prepare a small verified evidence handout, without copying models or personal video."""

import argparse
import json
from pathlib import Path

from benchmark_metrabs_precision import verify_probe
from check_demo import check_demo

from poseahead.artifacts import sha256_file
from poseahead.evaluation.batch import verify_batch
from poseahead.evaluation.selection import select_from_batches

PANELS = {
    "gaussian": "cmu_adaptive_confirmation_gaussian_20261011",
    "recovery": "cmu_adaptive_recovery_20261011",
    "spec_stress": "cmu_spec_stress_20261011",
}
PRECISION_PROBES = {
    "recorded_motion": "performance_precision_clean_20261018",
    "repeated_still": "performance_precision_still_clean_20261018",
}


def prepare_demo(repository, output, *, include_performance=False):
    repository, output = Path(repository).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing demo evidence is preserved.")
    preflight = check_demo(repository, load_models=True)
    if preflight["status"] != "ready":
        raise ValueError("Presentation preflight failed: "
                         + json.dumps(preflight, ensure_ascii=False))
    sources, selections, verified = {}, {}, {}
    for key, name in PANELS.items():
        panel = repository / "outputs" / name
        for seed in (42, 43, 44):
            result = verify_batch(panel / f"seed{seed}" / "batch")
            verified[f"{key}/seed{seed}"] = result
        path = panel / "selection.json"
        selections[key] = json.loads(path.read_text(encoding="utf-8"))
        reproduced = select_from_batches(panel / "selection_protocol.json",
                                         panel / "selection_manifest.json")
        if reproduced != selections[key]:
            raise ValueError("Saved selection does not reproduce: " + str(path))
        for name in ("selection.json", "selection_protocol.json", "selection_manifest.json"):
            source = panel / name
            sources[source.relative_to(repository).as_posix()] = sha256_file(source)
    live_path = repository / "outputs/ui_20261005_140958_4ed33ebe/run_report.json"
    live = json.loads(live_path.read_text(encoding="utf-8"))
    sources[live_path.relative_to(repository).as_posix()] = sha256_file(live_path)
    gaussian = {x["candidate_id"]: x for x in selections["gaussian"]["candidates"]}
    recovery = {x["candidate_id"]: x for x in selections["recovery"]["candidates"]}
    rows = []
    for profile in ("sigma0", "sigma10", "sigma25", "sigma50"):
        baseline = gaussian["kalman_cv"]["profiles"][profile]["means"]
        adaptive = gaussian["frozen_adaptive"]["profiles"][profile]["means"]
        rows.append({"profile": profile, "kalman_cv": baseline, "adaptive": adaptive})
    dropout = {
        key: recovery[key]["profiles"]["sigma25_dropout002"]["means"]
        for key in ("kalman_cv", "frozen_adaptive", "partial_recovery")
    }
    producer, unity = live["producer"], live["unity"]
    performance = {}
    if include_performance:
        for key, name in PRECISION_PROBES.items():
            directory = repository / "outputs" / name
            performance[key] = verify_probe(directory)
            for filename in ("completion.json", "report.json", "poses.npz"):
                source = directory / filename
                sources[source.relative_to(repository).as_posix()] = sha256_file(source)
    report = {
        "schema_version": 1, "source_dates": {"validation": "2026-10-11", "live": "2026-10-05"},
        "scope": "Verified saved validation/live evidence, "
                 "not a new measurement or final acceptance.",
        "preflight": preflight, "source_sha256": sources, "verified_batches": verified,
        "gaussian": rows, "rare_dropout": dropout,
        "spec_stress_status": selections["spec_stress"]["status"],
        "live": {
            "duration_s": producer["duration_s"], "valid_pose_fps": producer["valid_pose_fps"],
            "p95_capture_to_send_ms": producer["capture_available_to_send_complete_ms"]["p95"],
            "sent": producer["sent_frames"], "received": unity["received_pose_frames"],
            "applied": unity["applied_pose_frames"],
        },
        "limitations": [
            "Repeated validation, three clips/three noise seeds; not held-out results.",
            "Latency superiority is not established. Missing masks restrict comparison.",
            "Both methods fail the severe SPEC coverage panel; no overall winner.",
            "Live target >=25 valid FPS and p95<=50 ms remains unmet.",
            "Monocular root is estimated camera-relative translation, "
            "not ground-truth global motion.",
        ],
    }
    if include_performance:
        report["source_dates"]["recorded_performance"] = "2026-10-18"
        report["recorded_performance"] = performance
        report["limitations"].append(
            "Recorded precision probes exclude capture/decode/UDP/Unity; "
            "neither live acceptance nor ground-truth accuracy is assessed."
        )
    lines = [
        "# PoseAhead — результаты для показа", "",
        "Сохранённые результаты, проверенные по SHA-256. "
        "Это не новый прогон и не финальная приёмка.",
        "", "## Ошибка и дрожание на CMU", "",
        "Три validation-клипа, три seed шума; равный вес клипов/seed. Чем меньше, тем лучше.", "",
        "| Шум | Калман: ошибка, мм | Adaptive: ошибка, мм | "
        "Калман: jitter, м/с² | Adaptive: jitter, м/с² |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        base, adapt = row["kalman_cv"], row["adaptive"]
        lines.append(f"| {row['profile']} | {base['display_mpjpe_mm']:.2f} | "
                     f"{adapt['display_mpjpe_mm']:.2f} | {base['jitter_m_s2']:.2f} | "
                     f"{adapt['jitter_m_s2']:.2f} |")
    lines += ["", "На шумных наблюдениях есть улучшение ошибки/дрожания. "
              "Выигрыш по задержке не доказан; это повторная validation, не test.",
              "", "## Исправление редких пропусков", ""]
    for name, metrics in dropout.items():
        lines.append(f"- {name}: {metrics['display_mpjpe_mm']:.2f} мм; "
                     f"{metrics['jitter_m_s2']:.2f} м/с².")
    lines += ["", "На частых пропусках и потере tracking полный SPEC-порог покрытия не пройден "
              "обоими методами; победителя нет.", "", "## Сохранённый live-прогон 5 октября", "",
              f"600 с: {producer['valid_pose_fps']:.2f} полных поз/с; "
              f"p95 capture → send {report['live']['p95_capture_to_send_ms']:.2f} мс.",
              f"Отправлено / принято / применено: {report['live']['sent']} / "
              f"{report['live']['received']} / {report['live']['applied']}.",
              "Доставка работает. Цели 25 FPS и 50 мс не достигнуты. "
              "Это не задержка от экспозиции камеры до экрана.", ""]
    if include_performance:
        lines += ["## Ускорение MeTRAbs на записанном видео 18 октября", "",
                  "ABBA, один и тот же декодированный префикс в каждом блоке. "
                  "Захват, декодирование, UDP и Unity исключены. Это не live-приёмка.", "",
                  "| Материал | Autocast, поз/с | FP16, поз/с | Прирост | "
                  "Массивы совпадают точно |",
                  "| --- | ---: | ---: | ---: | --- |"]
        for key, result in performance.items():
            speed = result["aggregate_valid_fps"]
            growth = result["throughput_growth_percent"]
            growth_text = f"{growth:.2f}%" if growth is not None else "не определён"
            identity = "да" if result["numeric_identity_all_blocks"] else "нет"
            lines.append(f"| {key} | {speed['autocast']:.2f} | {speed['fp16_conv']:.2f} | "
                         f"{growth_text} | {identity} |")
        lines += ["", "Повторяемый still-файл не обобщается на произвольное движение. "
                  "Совпадение массивов не является проверкой точности относительно ground truth. "
                  "Быстрый профиль остаётся опцией, исходные настройки не заменены.", ""]
    lines += ["## Файлы-источники", ""]
    lines += [f"- `{name}`" for name in sources]
    output.mkdir(parents=True, exist_ok=False)
    with (output / "evidence.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    with (output / "results.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new demonstration directory")
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--include-performance", action="store_true",
                        help="also verify saved ABBA arrays/timings, without CUDA or camera")
    args = parser.parse_args()
    try:
        report = prepare_demo(args.repository, args.output,
                              include_performance=args.include_performance)
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        parser.exit(1, str(error) + "\n")
    print(f"Prepared {args.output}: {len(report['verified_batches'])} verified batches; no camera.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
