# Native matrix reader reuse: rejected development experiment

Reusing numeric matrices within native block analysis reduced matrix-reader calls
but increased traced peak allocation on all three inspected workflows and did not
consistently reduce compilation time. The production implementation is unchanged.
The rejected patch and exact diagnostic driver are retained in the
[raw evidence](evidence/native-matrix-reads.json), together with every paired
observation. This is a `single_device_fast_path` development result, not a
qualified performance, SDK ranking or scalability claim.

## Proposed mechanism and existing boundaries

The native CZ block workflow reuses the existing Qiskit-derived KAK/Weyl synthesis,
bounded collection, residual checks and separate physical-evidence verification.
The experiment changed only `_native_matrix_reader` inside
`compiler/lowering/native_block_optimization.py`: each reader memoized the numeric
matrix for `(instruction identity, ordered physical pair)`. Trusted instruction
membership, baseline replay, topology replay and all acceptance thresholds stayed
unchanged. A separate verifier constructed another reader; no data crossed calls.

The intended opportunity was repeated numerical reads during spectator admission,
block collection, product construction and operator verification. The operand pair
must be part of identity: a one-wire rotation occupies different matrix tensor
factors depending on its position in that pair. Keeping a reader alive through
later calls would also be invalid because native IR can contain mutable tensor
angles or parameter mappings.

New scenario tests preserve a shared wire's operand position across two interacting
pairs and require later verification to reject modified tensor and mapping angles
after an accepted rewrite. Fresh analysis must read the changed value and produce
an independently equivalent circuit. These tests remain useful without caching.

## Frozen inspected controls

Production control: `6db1fbda8a168599e5c0f155b9fb14052cf01342`.
The candidate differs only by the archived reader patch. SDKBench remains clean at
`9aa9260920984e2ca05bcb293abb885fccd9c144`, with already inspected nonce `01`
repeated 32 times. The 60 application/topology pairs span widths 6/8/12, QAOA,
Ising, hopping, QFT and repetition circuits, and line/ring/grid/sparse connectivity.
The native operation budget and independent acceptance criteria are unchanged.

All 60 pairs retain identical normalized complete-output hashes, evaluator
metrics, endpoint layouts, restored SWAP counts and native rewrite statistics;
120 independent acceptances completed. Six-wire checks cover complete operators
up to global phase; larger workflows retain the existing probe scope. This set
is development evidence and must not be called unseen qualification.

Two warmups and seven alternating retained samples per variant, with separate
Python allocation observations, give:

| Inspected workflow | Existing reader median | Memoized reader median | Python traced peak, existing / memoized |
| --- | ---: | ---: | ---: |
| 6-wire QFT, grid | 100.756 ms | 112.956 ms | 1,894,644 / 2,129,785 bytes |
| 6-wire repetition, ring | 128.397 ms | 127.780 ms | 2,406,525 / 2,582,515 bytes |
| 12-wire QFT, grid | 630.116 ms | 647.097 ms | 10,109,183 / 10,591,173 bytes |

The first and third observations are slower; the second is nearly unchanged.
Separate profiler runs reduce `_numeric_matrix` calls from 2,706 to 1,709,
2,487 to 1,700, and 14,709 to 9,377 respectively. Fewer calls are not sufficient
admission evidence. Dictionary storage and lookup introduce costs that the
end-to-end observations must retain. Profiling includes instrumentation overhead
and is not substituted for the unprofiled samples. Python traced allocation is
not process RSS. Shared-host samples do not establish statistical significance.

## Decision and verification

Reject this patch for production admission. Do not add a persistent cache, weaken
independent replay or change tolerances to rescue the experiment. Further work
should follow measured end-to-end costs rather than a reader-call proxy.

The raw record includes the exact patch, production-source hash, driver hash,
source revision, all 60 paired outputs, timing samples and separate profiles.
The original diagnostic called the candidate source hash `runner_sha256`; the
archive labels that value `implementation_sha256` and records the actual driver
hash as `runner_sha256`. Its original document hash is retained. Numeric
observations and source revisions are unchanged.

The existing user-facing workflow remains the target-artifact compiler path used
by SDKBench's `flagquantum-target-artifact` adapter. The retained tests exercise
that native workflow and its separate verifier:

```bash
python -m pytest tests/unit/test_native_block_optimization.py \
  tests/integration/test_native_block_physical_evidence.py -q
```

Expected: 131 passing focused unit/integration tests, including parameter mutation
rejection and complete-operator equivalence. The same test selection passed with
the experimental reader; the final unchanged production reader is checked again
before publication. Full-checkout CI remains required before integration. The
reproduction driver has measured local `/private/tmp` paths; adjust locations
without changing the pinned harness, cases, limits or acceptance criteria.

The local default-tier command also passes its sparse selection: 89 unit tests
passed and 42 integration tests were deselected. This is not a complete-checkout
default-tier certification; the explicit 131-test run above includes those
integration tests. The production reader file has no diff from the control.
