# Private AI predecoding

Core owns this offline research implementation. Training remains CPU-based; an
explicit experimental comparison lane transfers inference to CUDA. It composes the canonical
QEC DEM, memory circuits and existing global decoder protocol. It does not own
code construction, circuit execution, SDK adapters or a second decoder registry.
There are no public exports from this package.

The correction-label study uses `diagnose_equivalence(h, logical, target,
candidate)` to distinguish latent mismatch from equal complete H/L output.
`FeatureTable.update(model._features(detectors), logits, labels)` streams exact
three-feature MLP groups; `summarize()` returns counts, score aggregates and
empirical conditional BCE. These are CPU research diagnostics, not final logical
accuracy or population Bayes bounds. The three-feature table does not claim CNN
support. Follow the frozen
[diagnostic protocol](../../../../docs/development/QEC_AI_LABEL_IDENTIFIABILITY.md)
before collecting data. The archived-checkpoint runner is
`examples/qec/diagnose_predecoder.py`; the fixed
[outcome](../../../../docs/development/QEC_AI_LABEL_IDENTIFIABILITY_RESULTS.md)
retains all three seeds and inactive controls without promoting a benefit gate.
No retraining is licensed by the diagnostic implementation.

```bash
PYTHONPATH=. python examples/qec/diagnose_predecoder.py --fixture-smoke
```

This exact probe reports `latent_equal=0`, `detector_equal=1` and
`detector_and_logical_equal=1`. It draws no noise samples and trains no model.
For the fixed study, pass `--input-directory` pointing to the preserved A800-172
study and a fresh `--output-directory`, inside the same producing environment.
Every input and all three checkpoints are checked before any new sample. The
runner writes a manifest and per-seed reports; interrupted partial output must
be retained, not retried as a newly selected completion.

| Module | Responsibility |
| --- | --- |
| `predecoder.py` | Shared local MLP, DEM-bound buffers and latent training batches; local correction, residual syndrome and logical contribution composition. |
| `spacetime.py` | Square-grid detector layout, masks and local 3D CNN; reuses the same sampling and hybrid decode implementation. |
| `training.py` | CPU Adam training and exact optimizer, batch-generator and step recovery. |
| `assessment.py` | Held-out evaluation, paired logical counts, confidence intervals and CPU whole-block timing. |
| `exact_statistics.py` | Separately named, outward-bracketed exact-binomial intervals and conservative paired-count ratio summaries; old assessment defaults stay unchanged. |
| `diagnostics.py` | Label sparsity, positive/negative BCE, complete H/L correction equivalence and streaming exact MLP feature/label groups. |
| `calibration.py` | Calibration-only temperature fitting and separate validation threshold selection. |
| `comparison.py` | Shared CPU global decoding boundary, explicit device transfers, exact correction-bit parity and whole-batch timing with caller-owned synchronization. |

Dependencies flow from training and assessment to the predecoder; the CNN
extends it. Allowed dependencies are PyTorch, the standard library and existing
QEC records/protocols. Importing `decoders` does not eagerly import this package.
Keep AI implementation here, and add modules only when they contain implemented
behavior. Accelerator and streaming support remain gated by
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

The [calibration protocol](../../../../docs/development/QEC_AI_CALIBRATION.md)
declares a separate fixed budget. To check its workflow:

```bash
PYTHONPATH=. python examples/qec/calibrate_predecoder.py \
  --smoke --output-directory /tmp/qec-calibration-smoke
python -m pytest tests/qec/test_predecoder_calibration.py -q
```

Use a new directory. The command writes a manifest, two smoke training
checkpoints and results for raw/calibrated controls. All training checkpoints and
then all validation decisions are fixed before any primary test is sampled.
The default decoder temperature is 1.0, so existing callers retain their
behavior. Calibration settings are explicit in the report; training checkpoint
contents are unchanged. Neither probability diagnostics nor smoke results are
evidence of a logical-error-rate benefit.

Retain the small CPU, independent DEM support boundary. The MLP accepts visible
hyperedges only with `allow_hyperedges=True`; each complete signature remains one
latent mechanism. Default construction still rejects them. Exclusive-location
groups and invisible mechanisms remain unsupported. The new `correct()` method
returns residual/frame bits; the existing `decode()` composes them as before.
The CNN is
offline and reads both past and future. No accuracy, latency, hardware or
production capability is promoted by reorganizing these modules.

The optional [Ising comparison protocol](../../../../docs/development/QEC_AI_ISING_COMPARISON.md)
uses a pinned external checkout from the example entry point, without importing
vendor code in this package. It keeps circuit-shot sampling distinct from the
explicit independent-DEM approximation used for training and matching. Its smoke
executes real upstream code with untrained weights and is never a ranking.

The separate [A800 protocol](../../../../docs/development/QEC_AI_ISING_A800.md)
fixes d=9/R=9, actual pinned Fast weights, FP32 and complete CPU-input-to-CPU-output
timing, including transfers and matching. On an isolated A800 container exposing
one device, add `--a800 --image-id "$IMAGE_ID" --ising-checkpoint /task/fast.safetensors`
to the existing comparison command and use a fresh directory. `--a800 --smoke`
reports 32 primary test shots and `scientific_ranking_allowed=false`; it never
substitutes random weights for missing authorized weights. CUDA availability and
source/weight/image metadata are checked before training. Exact residual/frame
parity on declared probes and final predictions on physical probe shots must pass
before primary test sampling. This is not a streaming decoder or optimized
TensorRT deployment. Public exports and training checkpoints are unchanged.

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


For the prospectively registered continued-checkpoint study, `exact_statistics.py`
uses standard-library Decimal directed rounding and positive binomial recurrences.
Its certified scope is 1 to 4194304 shots and marginal alpha in [1e-12, 1).
Callers allocate family error explicitly; `summarize_exact` splits comparison alpha
across its two marginal intervals. It preserves `PairedCounts` and the existing
exact paired p-value; zero reference lower bounds remain inconclusive. It does
not change `assessment.rate_interval` or `assessment.summarize`. Tail summation
cost grows with the smaller failure/success count; it is not a latency primitive.

Run `python -m pytest tests/qec/test_exact_statistics.py -q` for independent
rational-binomial tail/coverage tests and pinned SciPy1.17.1 beta-quantile numeric
references (no SciPy runtime dependency). These are synthetic statistical
fixtures, not measured logical-quality benefit. The physical study runner and
new data collection remain separate prerequisites in
[the frozen protocol](../../../../docs/development/QEC_AI_CONTINUED_PHYSICAL_ASSESSMENT.md).

The optional `examples/qec/continued_selection.py` module has internal `calibrate` and `freeze_choices`
stages. Calibration saves 128 batches of 32 labels and both horizons' logits per
seed, then calls the existing temperature fitter. Physical validation replays
saved circuit shards through the complete residual/global/logical-XOR path,
retains every threshold trial and uses the same conservative tie rule as
`select_threshold`. Choices bind checkpoint, protocol, calibration, validation
and source hashes; subsequent stages must verify the externally frozen choices
hash. Existing default calibration behavior and checkpoint contents are unchanged.
Run `python -m pytest tests/qec/test_continued_assessment.py
 tests/qec/test_predecoder_calibration.py -q` for synthetic software scenarios.
The CLI remains verification-only. Actual two-host execution, primary prediction
shards, complete orchestration and matching-environment preflight are pending;
these stages do not authorize collecting the registered physical study yet.

The optional `examples/qec/continued_parity.py` module captures one post-choice
64-shot physical probe and replays the same file on both hosts. Its exact bit
oracles cover all fifteen paths, all zero/one/one-hot detector rows, and final
predictions on physical rows. The hardware receipt writer requires explicit CUDA
and the pinned Fast reference. Two receipt hashes must bind the same choices,
probes, source and per-path evidence on distinct recorded GPU UUIDs. This checks
stored bindings, not independent authentication of the remote host: the authorized
launcher and complete environment preflight remain required. Tests use synthetic
CPU paths and fabricated parser documents, never A800 evidence. Run
`python -m pytest tests/qec/test_continued_parity.py -q`. Primary prediction shards,
complete orchestration and actual two-host execution remain pending; no new real
research data are authorized by these primitives alone.
