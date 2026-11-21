# Preserve IR bytes while shortening container dispatch

This Integration-owned implementation supports the local compiler's
`single_device_fast_path`. Core remains the sole owner of IR encoding. The
change adds four executable lines to the existing private encoder: exact built-in
dicts, lists and tuples use their existing recursive encodings before symbolic
and abstract-protocol checks. Subclasses retain the entire previous dispatch
order. There is no cache, dependency, public option, supported-type extension,
IR version change or serialization contract change.

## Requirement and existing user workflow

A complete Rainbow integration reference profile identified recursive IR value
encoding as a general contributor to compilation overhead. This implementation
removes repeated dispatch for ordinary containers; it does not recognize circuit
families, fixtures, SDK identities or target workloads. Mature codec behavior is
reused rather than replaced with a separate serializer.

```python
import flagquantum as fq

source = fq.Circuit(2).h(0).cx(0, 1).to_ir()
encoded = source.to_json()
restored = fq.CircuitIR.from_json(encoded)
assert restored.to_json() == encoded
assert restored.content_hash == source.content_hash
```

Expected: the existing program and content identities round-trip without a new
option or user action. Stringified keys, recursive copies, symbolic tags, tensor
reads and existing errors remain unchanged. Output dictionaries and lists remain
fresh and owned by the caller; there is no shared mutable encoded result.

## Validation

Three new behavior tests cover nested exact containers, canonical bytes and
hashes, mutation of returned nested lists/dicts, and list/dict subclasses that
also carry Parameter semantics. Existing tests cover tensor and scalar-subclass
protocol precedence, current mutable tensor values, complex values, symbolic
expressions, custom mappings and unsupported types. No golden bytes, identity
fixtures, contracts or test expectations were updated.

Across the focused IR, Core, API-contract, CPU vertical, artifact compilation and
v4 evidence selection, 1,090 unique tests passed. Initial sparse-checkout
collection lacked tracked helper/example files. The subsequent complete selection
passed 1,089 tests and failed the CPU example because its tracked module was
still absent; after restoring that module, the exact failed test passed. This is
aggregate focused coverage, not a claim of one uninterrupted successful suite.
The full smoke/unit and runtime tiers remain required CI.

## Development measurements and limits

The unchanged [existing measurement runner](../../benchmarks/ir_scalar_encoding.py)
loads the exact parent encoder AST from main
`a7d84a605080c511d4a07037ef2b4f3b90a645d1`. All other code stays at the
candidate source. The [raw report](evidence/ir-container-encoding-development.json)
records the measured source SHA-256. Each variant gets two warmups and 41
serialization or 21 workflow samples, with alternating order and all outliers
retained. Optional adapters use the unchanged SDKBench checkout
`9aa9260920984e2ca05bcb293abb885fccd9c144`.

| Inspected workload | Parent median | Candidate median |
| --- | ---: | ---: |
| Serialize 768 native RZ instructions | 0.869 ms | 0.725 ms |
| Serialize 768 annotated CX instructions | 0.969 ms | 0.820 ms |
| Ordinary restored compilation, 12-wire line QFT | 34.226 ms | 33.220 ms |
| Target artifact compilation, same QFT | 741.409 ms | 692.979 ms |

Complete serialized IR strings match exactly. Both workflow variants pass the
unchanged independent three-probe semantic acceptance and restored-layout check
on their first observation; all subsequent operations/final-layout outputs
match that accepted output. The runner excludes elapsed fields from equality,
retains them for validation, and does not compare clock-dependent full artifact
identities across separate target calls. Existing golden artifact tests separately
protect those identities.

These observations are inspected development inputs on shared macOS CPU, using
one numerical thread, with no overlapping local test job during the measurement.
They do not establish isolated-host performance, statistical significance, cold
start, process memory, unseen-input performance, cross-SDK ranking, hardware
performance or scalability. Formal SDKBench qualification remains user-deferred.
The earlier scalar report and its original source pins are unchanged; these
samples must not be divided across campaigns to attribute a combined speedup.

Reproduce with the baseline Git object and pinned SDKBench checkout available:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.ir_scalar_encoding \
  --baseline a7d84a605080c511d4a07037ef2b4f3b90a645d1 --sdkbench
python -m pytest tests/unit/test_ir_container_encoding.py \
  tests/unit/test_ir_scalar_encoding.py tests/unit/test_ir_public_api.py \
  tests/unit/test_public_api_candidate.py tests/team/core tests/api_contract \
  tests/integration/test_cpu_vertical_slice.py \
  tests/integration/test_artifact_compilation_pipeline.py \
  tests/unit/test_compilation_evidence_v4.py -q
```

The optional workflow probe needs the pinned checkout's `src` on `PYTHONPATH`
and its `docs/implementation` bundle. The retained measurement used a 240-second
outer process-group timeout and submitted no remote jobs.
