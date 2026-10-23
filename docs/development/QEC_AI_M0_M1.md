# AI predecoder M0 / M1 reference

## Scope and baseline

The implementation was first inspected at `3490aabd4` and is validated against
main snapshot `63070070d91a0c892920ce7fa7aa796341b0fa45` after the QEC restructure
and historical-import retirement. The reference is a private CPU research path;
no existing export, registry entry, compatibility alias, dependency or frozen
schema changes. `flagquantum.qec.decoders._neural` is intentionally private.

Reuse the canonical DEM (`dem.model`), parity matrices, memory builder,
phenomenological noise, decoder protocol and the existing decoder factory.
The new implementation owns only latent-label sampling, a shared local neural
mechanism scorer, residual composition, Adam training/checkpointing and paired
evaluation. The example composes these with a distance-three surface memory.

M0 classification: `single_device_fast_path`. Scope: Core-owned QEC implementation,
shared tests, example and this document. Ownership checker passed. No distributed,
GPU, hardware feedback, CUDA-Q parity or production capability is promoted.

## Minimal complete workflow

```bash
PYTHONPATH=. python examples/qec/ai_predecoder.py \
  --steps 100 --shots 512 --backend pymatching --checkpoint /tmp/predecoder.pt
PYTHONPATH=. python examples/qec/ai_predecoder.py \
  --resume --steps 20 --shots 512 --backend pymatching --checkpoint /tmp/predecoder.pt
PYTHONPATH=. python examples/qec/ai_predecoder.py \
  --evaluate-only --shots 512 --backend pymatching --checkpoint /tmp/predecoder.pt
```

PyMatching is optional (`pip install 'flagquantum[pymatching]'`). Omit `--backend`
for the bounded in-tree exact matcher. It retains its existing defect ceiling.
The checkpoint binds the DEM, model buffers, weights, optimizer, batch generator
state and step. A different DEM is refused. Checkpoints are experimental Torch
weights-only files, not existing FlagQuantum deployment packages.

The scorer uses three local features per mechanism: active incident detectors,
detector degree and prior log odds. Its shared MLP intentionally establishes an
end-to-end reference before geometric 3D CNN work. It does not claim cross-distance
generalization. Inference consumes only detector bits; sampled mechanisms and
logical truth remain supervision/evaluation data. Batches are generated on demand.
Dense parity matrices bound this reference to small CPU workloads.

For correction c, residual = s XOR Hc; final prediction = Lc XOR D(residual).
The predecoder need not explain the complete syndrome. Both paths use the same
original DEM decoder weights; these are a heuristic for the residual distribution,
not a learned or exact residual posterior. Threshold defaults to 0.9 and can be
specified explicitly. It must be chosen on validation data, not held-out tests.
Unsupported hyperedges, invisible mechanisms, exclusive groups and probabilities
outside (0, 0.5) are refused. No arbitrary-channel or circuit-noise support follows.

## Verification and interpretation

```bash
python -m pytest tests/qec/test_neural_predecoder.py -q
python -m pytest tests/qec tests/api_contract/test_experimental_namespace_contract.py \
  tests/api_contract/test_qec_retirement_contract.py -q
python tools/ci_tier.py pr-default
```

Seven behavioral tests exercise local logical contribution plus residual decoding,
overlapping mechanism XOR, disabled-predecoder equivalence, latent parity and
marginals, trainable parameter changes, exact resumed optimizer/batch trajectory,
held-out evaluation, mismatch and invalid-input failures. Existing QEC/API tests
remain unchanged.

On Python 3.11.16 / Torch 2.10.0 CPU, the example with 100 steps recorded loss
0.7813841 -> 0.2834654. At evaluation seed 1001, 512 held-out samples gave 20
baseline and 20 hybrid logical failures. Reloaded evaluation reproduced these
counts. This is a smoke experiment, not evidence of accuracy improvement or
latency benefit. Conservative thresholds can select no corrections. The tests
force nonzero corrections separately to verify the otherwise hidden Lc path.
The example reports counts and rates only; a release-quality study must add
confidence intervals, multiple training seeds, predeclared sample budgets and
paired statistical analysis. Never interpret zero observed failures as zero risk.

The first full-QEC run at the earlier baseline reached 95% without recorded
failures before its 180-second bound; it is incomplete evidence. The PR records
final current-snapshot validation separately. This document does not turn a
historical or incomplete run into a passing baseline.

## M1 acceptance and next work

M1 delivers executable sampling -> training -> checkpoint/resume -> residual
predecode -> global decode -> held-out logical evaluation. The shared local MLP
is narrower than the planned geometric CNN. X/Z sweeps, spatial input layout,
independent circuit-sampler cross-checks for training labels, quantization,
accelerators, latency, confidence intervals and scientific improvement belong
to follow-up work. Existing DEM/sampling tests provide the independent reference
foundation; sampling generated by the new learner is not its own scientific proof.

Before M2, freeze a validation/test grid, add geometry-aware inputs and stronger
baselines, and predeclare a quality/performance objective. Do not migrate kernels
or claim Ising-Decoding parity based on decreasing training loss.
