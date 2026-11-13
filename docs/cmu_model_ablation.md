# CMU input-representation and batch-order experiment

This is a fixed 2x2 validation-only experiment on the expanded CMU pilot's 100-ms
windows. It uses the exact existing window directories and train-only normalizer;
no new download, test evaluation or change to the live configuration is involved.

| Variant | Network inputs | Training order |
| --- | --- | --- |
| positions_within_shards | Normalized pelvis-relative coordinates | Existing fixed shard order |
| positions_global_windows | Same coordinates | All train windows shuffled each epoch |
| relative_last_within_shards | Each history pose minus the last pose | Existing fixed shard order |
| relative_last_global_windows | Each history pose minus the last pose | All train windows shuffled each epoch |

Every variant uses ten history points at 20-ms spacing, two 64-unit hidden layers,
Adam at 0.001, eight epochs, batch size 64 and seed 42. The output remains a
correction to the last pose. Epoch zero is hold-pose in all four variants; no
constant-velocity forecast is inserted into the network. Relative-last inputs are
displacements in normalized coordinates, not velocities in metres/second. They
discard static joint configuration, which may be a limitation for anatomical
constraints. A future velocity-residual architecture would be a separate change.

The global shuffle caches training windows only, with a conservative 256-MiB data
working-set guard. This is a pilot facility, not an unbounded large-dataset loader.
The random state is seeded and restored; no CUDA context is requested. Validation
remains in fixed order. Global batches can cross recording boundaries, but each
window still comes from a single clip: no temporal sequence is concatenated.

Legacy models keep version-1 manifests and their original input semantics.
Relative-last checkpoints require version 2 with an explicit input_features field;
the loader reconstructs the matching model and rejects an ambiguous downgrade.
Numeric NPZ checkpoint loading and checksum verification remain in place.

```powershell
.venv/Scripts/python.exe scripts/cmu_model_ablation.py outputs/cmu_expanded_20260924/h100 outputs/cmu_ablation_NEW
```

Retained run: `outputs/cmu_ablation_20260927/`. `protocol.json` is written before
training; each variant saves a complete model and epoch log. `report.json` is the
completion marker and includes hold, constant-velocity and reloaded-model errors.
Training and validation window manifest hashes and normalizer hashes are frozen.

## Completed results (2026-09-27)

Validation canonical17 MPJPE, including zero pelvis, after checkpoint reload:

| Variant / baseline | Best epoch | Error, mm |
| --- | ---: | ---: |
| Hold pose | — | 30.5192 |
| Constant velocity, last two samples | — | 12.6798 |
| positions_within_shards | 0 | 30.5192 |
| positions_global_windows | 0 | 30.5192 |
| relative_last_within_shards | 4 | 27.0484 |
| relative_last_global_windows | 6 | 20.9990 |

The original-feature control reproduced the prior expanded-data result. Relative
inputs enabled a learned improvement over hold, and global mixing improved that
variant further. The best neural result is about 31.2% below hold error but remains
about 65.6% above constant-velocity error. This isolates two candidate factors at
one horizon/seed, not a general proof about architectures. No checkpoint was promoted
to the live configuration. Next: check repeat seeds and other horizons, and compare
against a velocity-based residual architecture under the shared delay/noise protocol.

Interpret results as exploratory validation evidence: one seed, three validation
clips from one source ID, repeatedly examined during development. No confidence
interval, held-out generalization, latency/jitter reduction or webcam-noise
robustness is established by this experiment. Do not promote its winner directly
to the application or describe epoch-zero selection as learned improvement.
