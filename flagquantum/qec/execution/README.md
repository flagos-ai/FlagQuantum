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
  sampler draws, location order and seed handling unchanged.
- `context.py`: declaration-to-DEM and per-basis measurement maps, without shot
  execution. It consumes declarations and DEMs; declarations never import it.
- `signatures.py`: source injection, exact lowering, forced reference execution
  and the existing shared fault vocabulary. DEM construction consumes these
  functions/constants without a reverse import, avoiding a module cycle.
- `noise_sweep.py`: the existing finite-shot loop over repetition execution.

The historical flat `repetition`, `sampling`, `context`, `circuit` and
`dem_construction` imports resolve to the same moved objects. Their defining
metadata and own methods/accessors retain the old modules for pickle and
caller-owned `torch.load(weights_only=True)` allowlists. Existing private helper
aliases remain reachable too. No library-managed allowlist or custom reducer is
introduced. Source tools should follow physical owners rather than use a retained
class `__module__` as its physical filename. The eager QEC root facade is unchanged.

Core owns these aliases. No removal release is approved; removing an old import
or pickle path requires a separate approved deprecation and migration proposal.

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
results and historical Python identifiers. Raw seeded sample equality is scoped
to its recorded Stim version and machine architecture; the unchanged independent
statistical/reference tests cover other supported sampling environments.
