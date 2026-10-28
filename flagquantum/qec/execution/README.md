# QEC execution composition

Core owns the composition of existing QEC declarations, noise, DEMs and decoders
with compiler/runtime/stabilizer execution. This package must not define another
code or decoder registry, numerical kernel, code family or serialized schema.
Its initializer imports no experiment machinery. Allowed dependencies are the
existing QEC records and algorithms, compiler lowering, runtime feedback/session
execution, core IR, noise channels and the optional stabilizer sampler.

- `repetition.py`: the original repetition-memory source and feedback controller.
- `sampling.py`: lowering plans, noise placement, measurement records and sampled
  detection/observable parity. It shares lowering with `signatures.py` and keeps
  existing one-shot draws, location order and seed handling unchanged.
  `compile_memory_sampler` prepares a separate reusable stream: lowering,
  placement and Simulation compilation happen once; immutable parity-column
  tuples are retained instead of the caller's memory/noise records. Each draw
  returns the existing `DemSample` and retains no sample buffers.
- `context.py`: declaration-to-DEM and per-basis measurement maps, without shot
  execution. It consumes declarations and DEMs; declarations never import it.
- `signatures.py`: source injection, exact lowering, forced reference execution
  and the existing shared fault vocabulary. Gate-bound signatures now compose
  a private Simulation-owned Clifford parity primitive with the detector and
  observable layouts, without executing an amplitude store. The original forced
  execution remains the independent small-system test reference. DEM construction consumes these
  functions/constants without a reverse import, avoiding a module cycle.
- `experiments.py`: the approved fixed-shot composition of original-source
  export, strict actual-law DEM, complete-record batch decoding and a single
  owned sampler. Canonical imports only; no root/parent re-export or persistence
  schema. Run `examples/qec/memory_experiment.py` with the Stim/PyMatching extras
  for all five default decoders; unsupported model/decoder intersections refuse.
- `noise_sweep.py`: the existing finite-shot loop over repetition execution.

The approved 0.4.0 retirement removes flat execution imports and their pickle
lookups. Use the root exports or the owning modules above. Definitions and own
methods now use canonical metadata; `run_repetition_memory_noise_sweep` keeps
its surviving `qec.noise` identity. Caller-owned `weights_only=True` allowlists
must use current objects when loading new canonical checkpoints. No library
allowlist or custom reducer restores retired lookups. The eager root is unchanged.

The shortest complete user workflow is:

```bash
pip install -e '.[stim]'
PYTHONPATH=. python examples/qec/memory_layout_and_sampling.py
```

It constructs an existing repetition memory, reads decoder maps, samples zero-noise
measurements and detection events, and executes all three feedback modes. It prints
`9 8 1`, map shape `(8, 9)`, measurement shape `(4, 9)`, no detection/observable
flips, and `0.0 (0, 0, 0)` for every feedback mode. With only core dependencies,
`PYTHONPATH=. python examples/qec/restructure_compatibility.py` remains runnable.

A typical sampling/noise-placement change starts in `sampling.py` or
`signatures.py`. Preserve instruction/measurement order, source anchors and RNG
consumption; then run:

```bash
python -m pytest tests/qec/test_restructure_compatibility.py tests/qec/test_restructure_boundaries.py tests/qec/test_restructure_execution.py tests/qec/test_sampling_memory_circuit.py tests/qec/test_decoder_context.py tests/qec/test_repetition_memory.py -q
```

The trusted pre-004 fixture protects exact source/layout/map results, matrix-vs-
circuit DEM semantics, forced signatures, raw samples, detected parities, feedback
results and all surviving Python identifiers; retired identifiers are refused. Raw seeded sample equality is scoped
to its recorded Stim version and machine architecture; the unchanged independent
statistical/reference tests cover other supported sampling environments.

## Repeated batches

```python
from flagquantum.qec import (
    PhenomenologicalNoise, RepetitionCode, build_memory_circuit,
    compile_memory_sampler,
)

memory = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
with compile_memory_sampler(memory, noise=PhenomenologicalNoise(), seed=53) as sampler:
    for _ in range(2):
        batch = sampler.sample(shots=32)
        assert batch.detectors.shape == (32, 8)
        assert not batch.detectors.any() and not batch.observables.any()
```

The handle belongs to its creator process and thread. Context exit releases the
prepared numerical stream even on an exception; `close()` is owner-idempotent.
Sampling after close or from another owner raises `RuntimeError`. Invalid shot
counts do not advance the stream. Execution/parity failure closes it before
propagating the original error. Live handles cannot be copied or pickled.
Create separate handles and assign seeds explicitly in separate workers.

A fixed seed reproduces an identical sequence of sample calls within one engine
version and machine instruction set. Different batching is not a promise of
identical bits. `PYTHONPATH=. python examples/qec/stim_pymatching_baseline.py`
uses this stream with the optional PyMatching batch adapter, including a final
partial chunk. Configuration changes require constructing a new handle.

Run `python -m pytest tests/qec/test_memory_sampler.py tests/team/simulation/test_prepared_stabilizer_sampling.py -q`
for lifecycle, numerical seam and replacement checks. The approved additive
surface is recorded in `contracts/qec-memory-sampler-v1.json`; historical
restructuring fixtures remain unchanged.

Gate-bound one-/two-qubit Pauli mixtures are placed as complete channels using
the locations shared with DEM assembly. Branches at one location are exclusive;
different locations remain independent. The prior X/Z channel reconstruction
and seeded instruction stream are preserved. See
[QEC-FUNCTIONAL-004](../../../docs/development/QEC_FUNCTIONAL_004.md) and run
`PYTHONPATH=. python examples/qec/correlated_circuit_noise.py` for the core-only
declared-model example; physical circuit sampling uses the existing Stim extra.
