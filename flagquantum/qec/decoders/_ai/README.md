# Private AI predecoding

Core owns this offline CPU research implementation. It composes the canonical
QEC DEM, memory circuits and existing global decoder protocol. It does not own
code construction, circuit execution, SDK adapters or a second decoder registry.
There are no public exports from this package.

| Module | Responsibility |
| --- | --- |
| `predecoder.py` | Shared local MLP, DEM-bound buffers and latent training batches; local correction, residual syndrome and logical contribution composition. |
| `spacetime.py` | Square-grid detector layout, masks and local 3D CNN; reuses the same sampling and hybrid decode implementation. |
| `training.py` | CPU Adam training and exact optimizer, batch-generator and step recovery. |
| `assessment.py` | Held-out evaluation, paired logical counts, confidence intervals and CPU whole-block timing. |

Dependencies flow from training and assessment to the predecoder; the CNN
extends it. Allowed dependencies are PyTorch, the standard library and existing
QEC records/protocols. Importing `decoders` does not eagerly import this package.
Keep future AI diagnostics and calibration here, and add modules only when they
contain implemented behavior. Accelerator and streaming support remain gated by
the [delivery plan](../../../../docs/roadmap/QEC_AI_DECODING_PR_PLAN.md).

## Ten-minute workflow

From the repository root, with PyTorch installed:

```bash
PYTHONPATH=. python examples/qec/ai_predecoder.py \
  --model cnn --steps 2 --shots 16 --checkpoint /tmp/qec-ai-cnn.pt
PYTHONPATH=. python examples/qec/ai_predecoder.py \
  --model cnn --evaluate-only --shots 16 --checkpoint /tmp/qec-ai-cnn.pt
PYTHONPATH=. python examples/qec/assess_predecoder.py \
  --smoke --output /tmp/qec-ai-assessment.json
```

The first two commands report training step 2 and identical held-out baseline
and hybrid failure counts. The reload has null training-loss fields. The third
command exclusively creates a JSON file with configuration and source hashes,
paired counts, validation-selected thresholds and scoped timings. Use a fresh
output path; smoke results do not establish scientific benefit.

For a model change, edit its owning module and run:

```bash
python -m pytest tests/qec/test_neural_predecoder.py \
  tests/qec/test_spacetime_predecoder.py \
  tests/qec/test_neural_training_data.py \
  tests/qec/test_predecoder_assessment.py -q
```

Retain the small CPU, independent graphlike DEM support boundary. The CNN is
offline and reads both past and future. No accuracy, latency, hardware or
production capability is promoted by reorganizing these modules.

## Retired private paths

The former `decoders._neural`, `decoders._spacetime` and
`decoders._assessment` modules are removed. Repository examples, tests and
assessment source hashing use the files above. There are no forwarding modules,
import aliases or pickle bridges. Existing public QEC imports and decoder
registration are unchanged.

Training checkpoints contain state dictionaries, not pickled model instances.
Their version, DEM binding, tensor names and optimizer/generator state are
unchanged. Existing MLP and CNN training checkpoints remain loadable. Historical
assessment JSON and hashes continue to describe their original source revision;
new runs fingerprint all four current implementation modules. Do not rewrite
old evidence or present the reorganization as a new experiment.
