# Preserve IR bytes while shortening scalar dispatch

This Integration-owned implementation change serves the local compiler's
`single_device_fast_path`. It changes no IR version, field, public API, supported
type, exception policy, canonical JSON, content hash, or capability level. Core
continues to own encoding; consumers continue calling the same methods.

## Work removed and boundaries preserved

Ordinary scalar leaves previously passed symbolic, complex, mapping, sequence,
and tensor-protocol checks before reaching their existing scalar return. The
encoder now returns `None` and exact built-in strings, integers, floats, and
booleans first. All other objects, including scalar subclasses, retain the
original dispatch. No serialized value is normalized or cached.

The guard uses type identity. Tuple membership was rejected because a custom
metaclass can claim equality with a built-in type. A tensor-like integer subclass
with such a metaclass must retain its tensor tag; the regression exercises this
precedence. String enums retain their existing path. The unchanged tensor path
reads current values on every call, so in-place edits still change JSON and its
hash. Symbolic expressions, complex tags, custom mappings, nested sequences,
unsupported-object errors, signed zero, large integers, Unicode escaping, and
the existing nonfinite metadata spelling remain unchanged.

## Evidence and limits

The final implementation passed 1,075 tests covering the new codec regressions,
existing IR API, Core, full API-contract directory, CPU vertical slice, target
artifact compilation, and v4 evidence. No golden artifact bytes or identities
were updated. Initial sparse-checkout collection and fixture failures were
corrected by restoring tracked examples, benchmark evidence, and QEC fixtures;
those incomplete runs are not passing evidence. Full default/runtime tiers
still require CI.

The [raw samples and fingerprints](evidence/ir-scalar-encoding-development.json)
compare the exact encoder AST from main
`2d78133a1f5a7d234acbb128724ccf2674e2eeb8` with the candidate. Other loaded
implementations remain unchanged. Two warmups precede 41 serialization samples
or 21 workflow samples per variant; order alternates. All outliers remain.

| Inspected workload | Parent median | Candidate median |
| --- | ---: | ---: |
| Serialize 768 native RZ instructions | 1.070 ms | 0.851 ms |
| Serialize 768 annotated CX instructions | 1.393 ms | 0.948 ms |
| Ordinary restored compilation, 12-wire line QFT | 38.463 ms | 36.013 ms |
| Target artifact compilation, same QFT | 871.360 ms | 739.812 ms |

Serialized IR strings match exactly. Optional SDKBench probes retain identical
output operations and final layouts; unchanged independent three-probe acceptance
passes for each variant. Elapsed fields remain present for independent validation
and are excluded from output equality. Clock-dependent target snapshots prevent
comparing full artifact identities between separate adapter calls; golden artifact
and codec tests cover canonical identity behavior separately. An initial diagnostic
incorrectly compared elapsed fields, then omitted that field from the validator;
neither failed diagnostic produced an accepted measurement artifact.

These are shared macOS CPU observations on inspected development inputs. The
approximately 15% target-workflow median reduction does not establish statistical
significance, isolated-host performance, cold start, process memory, held-out
performance, a cross-SDK ranking, or scalability. PR #1266's native construction
hash reuse is not applied in this baseline or candidate; combining both changes
needs fresh integration evidence. Formal SDKBench qualification remains deferred,
with its protocol unchanged.

## Existing user workflow

```python
import flagquantum as fq

source = fq.Circuit(2).h(0).cx(0, 1).to_ir()
encoded = source.to_json()
restored = fq.CircuitIR.from_json(encoded)
assert restored.to_json() == encoded
assert restored.content_hash == source.content_hash
```

Expected: byte-for-byte program and content identity round trips, with no new
option or user action. Reproduce with the parent Git object available locally:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.ir_scalar_encoding \
  --baseline 2d78133a1f5a7d234acbb128724ccf2674e2eeb8
python -m pytest tests/unit/test_ir_scalar_encoding.py \
  tests/unit/test_ir_public_api.py tests/team/core tests/api_contract \
  tests/integration/test_cpu_vertical_slice.py \
  tests/integration/test_artifact_compilation_pipeline.py \
  tests/unit/test_compilation_evidence_v4.py -q
```

The optional `--sdkbench` probe requires an installed or importable checkout of
`sdkbench-org/sdkbench` at source
`9aa9260920984e2ca05bcb293abb885fccd9c144`, including its `docs/implementation`
bundle. Add that checkout's `src` directory to `PYTHONPATH`. The retained run used
a 180-second outer process-group timeout. It submits no remote jobs and changes
no SDKBench adapter, case, acceptance threshold, or resource ceiling.
