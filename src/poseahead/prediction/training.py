"""CPU reference training, train-only updates and validation-only checkpoint selection."""

import json
import math
from pathlib import Path

import numpy as np

from poseahead.artifacts import sha256_file
from poseahead.datasets.normalization import PoseNormalizer
from poseahead.datasets.window_store import WindowStore


def train_predictor(config_path, output_directory):
    # Optional heavy dependency; base CLI/data tooling does not import Torch.
    import torch

    from poseahead.prediction.mlp import ResidualMLP

    config_path, output_directory = Path(config_path), Path(output_directory)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fields = {"version", "train_windows", "validation_windows", "normalizer", "seed",
              "epochs", "batch_size", "hidden_size", "learning_rate"}
    optional = {"input_features", "batch_order", "output_baseline"}
    if (not isinstance(config, dict) or not fields <= config.keys()
            or config.keys() - fields - optional):
        raise ValueError("invalid predictor training config fields")
    if type(config["version"]) is not int or config["version"] != 1:
        raise ValueError("unsupported training config version")
    features = config.get("input_features", "positions")
    batch_order = config.get("batch_order", "within_shards")
    baseline = config.get("output_baseline", "hold")
    if features not in ("positions", "relative_last"):
        raise ValueError("unsupported input_features")
    if batch_order not in ("within_shards", "global_windows"):
        raise ValueError("unsupported batch_order")
    if baseline not in ("hold", "constant_velocity"):
        raise ValueError("unsupported output_baseline")
    for field in ("seed", "epochs", "batch_size", "hidden_size"):
        if type(config[field]) is not int or config[field] < (0 if field == "seed" else 1):
            raise ValueError(f"invalid {field}")
    if config["seed"] >= 2**63 or config["epochs"] > 10000 or config["batch_size"] > 4096:
        raise ValueError("training seed/epochs/batch size outside supported limits")
    lr = config["learning_rate"]
    if type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if output_directory.exists():
        raise FileExistsError("predictor output directory already exists")
    for field in ("train_windows", "validation_windows", "normalizer"):
        if not isinstance(config[field], str) or not config[field]:
            raise ValueError(f"{field} requires a path")
        config[field] = str((config_path.parent / config[field]).resolve())
    train = WindowStore(config["train_windows"])
    validation = WindowStore(config["validation_windows"])
    if train.manifest["split"] != "train" or validation.manifest["split"] != "validation":
        raise ValueError("training requires train and validation only; test is never selected")
    for field, default in (("history_sampling", "exact"),
                           ("max_history_gap_ns", 100_000_000)):
        if train.manifest.get(field, default) != validation.manifest.get(field, default):
            raise ValueError("train and validation history sampling must match")
    norm_bytes = Path(config["normalizer"]).read_bytes()
    normalizer = PoseNormalizer(config["normalizer"])
    if Path(config["normalizer"]).read_bytes() != norm_bytes:
        raise ValueError("normalizer changed while loading")
    normalizer.check_compatible(train)
    normalizer.check_compatible(validation)
    if normalizer.config.get("source_manifest_sha256") != train.manifest_sha256:
        raise ValueError("normalizer was not fitted on these training windows")
    for store in (train, validation):
        if not store.manifest["windows"]:
            raise ValueError("cannot train/select on empty windows")
        store.verify()
    history = train.manifest["history_frames"]
    if baseline == "constant_velocity" and history < 2:
        raise ValueError("constant_velocity requires at least two history frames")
    horizon_steps = (train.manifest["horizon_ms"] * 1_000_000 / train.manifest["step_ns"]
                     if baseline == "constant_velocity" else None)
    cached_train = None
    if batch_order == "global_windows":
        # Explicit pilot-only memory ceiling, including normalization/concatenation copies.
        estimated_bytes = train.manifest["windows"] * (history + 1) * 17 * 3 * 4 * 4
        if estimated_bytes > 256 * 1024 * 1024:
            raise ValueError("global-window shuffle exceeds 256 MiB pilot memory budget")
        cached_train = {key: np.concatenate([shard[key] for shard in train.iter_shards()])
                        for key in ("history_positions_m", "target_positions_m")}
    # Keep caller RNG state unchanged; no CUDA context or device settings are touched.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["seed"])
        model = ResidualMLP(history, config["hidden_size"], input_features=features,
                            output_baseline=baseline, horizon_steps=horizon_steps)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        def batches(store, shuffle):
            source = [cached_train] if shuffle and cached_train is not None else store.iter_shards()
            for arrays in source:
                x = normalizer.transform(arrays["history_positions_m"]).astype(np.float32)
                y = normalizer.transform(arrays["target_positions_m"]).astype(np.float32)
                order = (torch.randperm(len(x)).numpy() if shuffle else np.arange(len(x)))
                for start in range(0, len(order), config["batch_size"]):
                    indices = order[start:start + config["batch_size"]]
                    yield torch.from_numpy(x[indices]), torch.from_numpy(y[indices])

        def validate():
            model.eval()
            total, count = 0., 0
            scale = torch.tensor(normalizer.scale, dtype=torch.float64)
            with torch.no_grad():
                for x, y in batches(validation, False):
                    residual = (model(x).double() - y.double()) * scale
                    errors = torch.linalg.vector_norm(residual, dim=-1)
                    total += errors.sum().item() * 1000
                    count += errors.numel()
            return total / count

        best_score, best_epoch = validate(), 0
        if not math.isfinite(best_score):
            raise ValueError("nonfinite initial validation metric")
        best_state = {key: value.detach().numpy().copy()
                      for key, value in model.state_dict().items()}
        log = [{"epoch": 0, "validation_mpjpe_mm": best_score,
                "training_normalized_mse": None}]
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            total, count = 0., 0
            for x, y in batches(train, True):
                optimizer.zero_grad(set_to_none=True)
                loss = ((model(x)[:, 1:] - y[:, 1:]) ** 2).mean()
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                optimizer.step()
                total += loss.item() * len(x)
                count += len(x)
            score = validate()
            if not math.isfinite(score):
                raise ValueError("nonfinite validation metric")
            log.append({"epoch": epoch, "validation_mpjpe_mm": score,
                        "training_normalized_mse": total / count})
            if score < best_score:
                best_score, best_epoch = score, epoch
                best_state = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}
    if Path(config["normalizer"]).read_bytes() != norm_bytes:
        raise ValueError("normalizer changed during training")
    output_directory.mkdir(parents=True, exist_ok=False)
    with (output_directory / "model.npz").open("xb") as stream:
        np.savez_compressed(stream, **best_state)
    with (output_directory / "normalizer.json").open("xb") as stream:
        stream.write(norm_bytes)
    report = {
        "format": "poseahead-residual-mlp", "version": 1, "device": "cpu",
        "config": config, "history_frames": history, "hidden_size": config["hidden_size"],
        "horizon_ms": train.manifest["horizon_ms"], "step_ns": train.manifest["step_ns"],
        "train_manifest_sha256": train.manifest_sha256,
        "validation_manifest_sha256": validation.manifest_sha256,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "best_epoch": best_epoch, "validation_mpjpe_mm": best_score, "epochs": log,
        "selection": "minimum_validation_canonical17_mpjpe_including_pelvis_epoch0_hold_eligible",
        "optimizer": "Adam", "gradient_clip_norm": 1.,
        "shuffle": "seeded_within_shards_fixed_shard_order", "test_data_used": False,
        "model_sha256": sha256_file(output_directory / "model.npz"),
        "normalizer_sha256": sha256_file(output_directory / "normalizer.json"),
        "source_sha256": {name: sha256_file(Path(__file__).resolve().parents[1] / name)
                          for name in ("prediction/training.py", "prediction/mlp.py",
                                       "datasets/normalization.py", "datasets/window_store.py")},
    }
    if features != "positions":
        report.update(version=2, input_features=features)
    if baseline == "constant_velocity":
        report.update(version=3, input_features=features, output_baseline=baseline,
                      selection="minimum_validation_canonical17_mpjpe_including_pelvis_"
                                "epoch0_constant_velocity_eligible")
    if batch_order == "global_windows":
        report["shuffle"] = "seeded_global_windows_each_epoch_256mib_pilot_limit"
    with (output_directory / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return {"best_epoch": best_epoch, "validation_mpjpe_mm": best_score,
            "horizon_ms": train.manifest["horizon_ms"], "test_data_used": False}
