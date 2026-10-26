# Frozen hybrid attribute encoding

This Compiler-owned change removes a redundant sorted-item list when private
hybrid IR attributes are converted to JSON-compatible data. `_freeze` already
copies and recursively sorts attribute mappings into immutable mapping proxies.
`_encode` can walk those proxies in their existing order. Other mappings retain
the original sorting path. Every conversion still creates fresh dictionaries and
lists; no model, attribute tensor, serialized result or identity is cached.

The canonical JSON writer is unchanged: ASCII escaping, compact separators,
sorted keys and UTF-8 encoding still feed SHA256. Pass verification and every
stage's semantic identity remain in place. Source locations remain diagnostic
and excluded from identity. No public API, schema, numerical tolerance, default,
dependency or SDK ownership changes.

## Evidence and scope

The baseline is `21b8e7f6e34a638e5120a0f4d7fff64fe196adb2`. The original
encoder failed a regression that refuses another attribute sort after model
construction; its other eight new semantic checks passed. Candidate tests compare
canonical bytes and SHA256 against independently stated wire data, including
nested mappings and sequences, Unicode and escapes, signed zero, large integers,
small floats and the existing non-finite-float spelling. They also verify sorted
attribute iteration, fresh returned containers, input isolation and the retained
ordinary-mapping sorting path.

The final combined hybrid Compiler and real-Stim memory/bridge checks passed
560 tests, with two existing PyMatching-dependent skips. The exact Python example
below printed its expected output. Local PyMatching checks were unavailable
because SciPy was absent; shared environments were not modified. Strict typing
of the modified module, formatting, lint, architecture, read-only public API,
dependency, scope, language and documentation checks passed.

The initial serial default run stopped at its 300-second budget and remains
incomplete. A bounded two-worker run completed in 97.16 seconds with 2,198 passed,
173 skipped and eight failures at the configured cutoff. All eight failures were
reproduced on the exact unmodified baseline with the same worker/thread settings:
two native behavior expectations and six local socket-permission failures. The
complete focused baseline groups had 106 passed and 127 skipped. Local default
is not called green, and neither unrelated assertions nor skips were changed.

The change eliminates a verified repeated sort and its intermediate list. It
does not eliminate the JSON-compatible data tree or the final JSON string and
UTF-8 bytes. It makes no wall-time, memory-capacity or general performance claim.
Earlier QEC instrumented profiles and paired timing studies are unchanged and
are not measurements of this candidate. The two native-workflow lowerings belong
to separate public export and sampler calls; this change does not combine them.

## Existing user workflow

Install `python -m pip install -e '.[stim]'`. This CPU workflow exercises the
existing memory lowering and sampler without exposing private serialization:

```python
from flagquantum.qec import (
    PhenomenologicalNoise, RepetitionCode, build_memory_circuit,
    compile_memory_sampler,
)

memory = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
with compile_memory_sampler(memory, noise=PhenomenologicalNoise(), seed=53) as sampler:
    for shots in (8, 8, 1):
        batch = sampler.sample(shots=shots)
        assert batch.detectors.shape == (shots, 8)
        assert batch.observables.shape == (shots, 1)
        assert not batch.detectors.any() and not batch.observables.any()
print("17 shots, no noiseless flips")
```

Expected output: `17 shots, no noiseless flips`. The workflow's sampling law,
seed scope, detector layout and handle lifecycle are unchanged. Exact-head
complete CI, GPU checks, reviews and branch rules remain delivery requirements.
