# QEC workflows with Stim and PyMatching

Use FlagQuantum code/memory and DEM objects with optional Stim circuit sampling
and a choice of decoders. This guide covers the supported CPU integration; the
[compatibility table](../reference/QEC_STIM_PYMATCHING_COMPATIBILITY.md) states
which paths are native, SDK-backed or refused.

## Install the candidate you are reviewing

For development from this checkout, use a private environment:

```sh
python -m pip install -e '.[stim,pymatching]'
```

For release acceptance, install the exact candidate wheel with the same extras
in a fresh environment outside the checkout, unset source Python paths, run
`python -m pip check`, and confirm `flagquantum.__file__` belongs to that
environment. Replace the candidate path in this template with the reviewed file:

```sh
python -m pip install '/absolute/path/to/candidate.whl[stim,pymatching]'
```

The new report entry is available in this PR's candidate. An older published
package does not gain it merely by adding extras. The core package works without
Stim/PyMatching; Stim-only supports native decoder choices, PyMatching-only
supports the declared-model decoder path, and both extras enable the full route.
No service credentials, GPU or hardware submission is required.

## Start with an existing Stim circuit

```python
from flagquantum.qec.execution.experiments import run_stim_experiment

source = (
    "R 0 1\nX_ERROR(0.125) 0\nX_ERROR(0.25) 1\nM 0 1\n"
    "DETECTOR rec[-2]\nDETECTOR rec[-1]\n"
    "OBSERVABLE_INCLUDE(0) rec[-1]\nOBSERVABLE_INCLUDE(1) rec[-2]\n"
)
result = run_stim_experiment(
    source, decoder="pymatching", shots=35, batch_size=16, seed=53
)
print(result.shots, result.logical_failures, result.observable_failures)
print(result.sampling_source)
```

Expected:

```text
35 0 (0, 0)
stim_circuit
```

D0 maps to L1 and D1 to L0. Each of the four independent-fault outcomes is
correctable in this example, explaining the zero count without asserting any
physical suppression or general decoder advantage. The report also carries the
logical error rate, existing 95% Wilson interval and seed. Shots are drawn as
16, 16, 3. Use
[`stim_memory_experiment.py`](../../examples/qec/stim_memory_experiment.py)
for the checked-in equivalent.

## Build a memory experiment here

```python
from flagquantum.qec import (
    PhenomenologicalNoise, RepetitionCode, build_memory_circuit,
)
from flagquantum.qec.execution.experiments import run_memory_experiment

memory = build_memory_circuit(RepetitionCode(distance=3), rounds=1)
result = run_memory_experiment(
    memory,
    noise=PhenomenologicalNoise(data_flip=0.01, measurement_flip=0.01),
    decoder="pymatching", shots=35, batch_size=16, seed=53,
)
assert result.shots == 35
assert result.observable_failures == (result.logical_failures,)
print(result.sampling_source)
```

Expected source: `memory_circuit`. This entry remains unchanged and distinct
from the external-text report. Both require Stim for original-circuit execution;
choosing a native decoder does not remove that sampling dependency.

The existing bounded example exercises all five default decoder names. Run one
of these development commands from the checkout after installation:

```sh
python examples/qec/memory_experiment.py --code repetition
python examples/qec/memory_experiment.py --code surface --basis x
python examples/qec/memory_experiment.py --code surface --basis z
```

Each reports 35 shots and checks the interval. Noisy Surface failure counts come
from actual draws and need not match between decoder algorithms. The X-memory
example uses its declared X product; changing a print label cannot change the
prepared logical basis.

## Inspect the model and decode the same batch

```python
from flagquantum.qec import DetectorErrorModel, get_decoder
from flagquantum.qec.adapters.stim import (
    compile_stim_sampler, stim_circuit_to_dem_text,
)
from flagquantum.qec.decoders.batch import decode_batch

model = DetectorErrorModel.from_stim_text(
    stim_circuit_to_dem_text(source, decompose_errors=False)
)
decoder = get_decoder("pymatching", model)
with compile_stim_sampler(source, seed=53) as sampler:
    samples = sampler.sample(shots=35)
    rows = tuple(tuple(row.nonzero().flatten().tolist()) for row in samples.detectors)
    records = decode_batch(decoder, rows)
assert len(records) == 35
assert samples.observables.shape == (35, 2)
```

Only detector events enter the decoder. Actual observable flips are retained
for evaluating predictions. Returned complete records preserve the selected
algorithm's correction/weight/convergence semantics. The dense Tensor method
`PyMatchingDecoder.decode_batch` is a separate logical-only operation; it does
not return correction edges. The sampler context owns cleanup and cannot be
shared arbitrarily across processes or threads.

Models with unsupported hyperedges, parallel-edge translations, isolated
detectors or unsupported circuit instructions fail through their existing owner.
Do not change the graph or silently switch decoders to force acceptance. A
supported alternative can be selected explicitly by name after checking its own
model contract. SDK failures do not produce partial successful statistics.

## Continue with focused examples

| Task | Example | Prerequisites |
| --- | --- | --- |
| Complete records on fixed independent syndromes | [decoder comparison](../../examples/qec/decoder_comparison.py) | Core plus PyMatching; no sampling |
| Batch versus per-shot semantics | [independent batches](../../examples/qec/independent_batch_decoding.py) | Existing declared profile |
| Circuit export and owned sampler | [Stim bridge](../../examples/qec/stim_experiment_bridge.py) | Stim, plus PyMatching for matching sections |
| Detector/logical joint probabilities | [joint parity](../../examples/qec/dem_joint_parity.py) | Core model mathematics |
| Layer ownership, roundtrip and sparse projections | [DEM chunks](../../examples/qec/dem_model_and_chunks.py) | Existing declared profile |
| Finite distance/noise example | [Surface sweep](../../examples/qec/surface_memory_sweep.py) | Stim and PyMatching; no threshold inference |

The historical `stim_user_migration.py` demonstration contains a 200,000-shot
reference marginal. It is not required by this short tutorial and is not run by
the new report acceptance. Existing scientific protocols and execution budgets
remain separate from the SDK quickstart.

For exact source/refusal and lifecycle semantics, see the
[report contract](../development/QEC_STIM_TEXT_EXPERIMENT_CONTRACT.md). For
current delivery and pending acceptance, see the
[central ledger](../roadmap/QEC_STIM_PYMATCHING_DELIVERY_PLAN.md).
