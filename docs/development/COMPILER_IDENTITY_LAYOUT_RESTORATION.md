# Identity-layout restoration fast path

A layout that already maps each logical qubit to its own physical qubit needs
no restoration SWAPs. The existing SABRE restoration helper now returns after
its permutation validation and edge-accounting loop in that case, before
building and walking a spanning forest. Nonidentity layouts retain the existing
algorithm. This Compiler-owned `single_device_fast_path` change adds no public
API, dependency, routing policy, serialized field, numerical tolerance or cache.

The existing [upstream reuse record](COMPILER_UPSTREAM_REUSE.md) documents the
Qiskit/Cirq adaptations. This change removes native restoration work; it is not
a new upstream algorithm port. It preserves the restoration contract instead of
allowing a free final permutation or weakening routing replay.

## Correctness and independent acceptance

Eight new test instances cover identity layouts on empty, disconnected and line
edge sets, invalid permutations, and replay of every five-wire permutation on
line and ring topologies. The replay checks physical token ownership and edge
legality independently of the restoration algorithm. Existing tests cover
nonidentity initial placements, restored output layouts, symbolic parameters,
gradients, channels, measurements, barriers, classical dependencies, deterministic
trial selection and failure behavior.

The sparse default tier passes all 367 selected unit tests; the runtime tier
passes 42 integration tests. An initial 240-test focused attempt passed 239 tests
and failed the default-authorization check because its tracked contract fixture
was absent from sparse checkout. Restoring that unchanged fixture resolves the
setup failure; no authorization record, snapshot or assertion was changed.
Strict Linux-target mypy passes the complete SABRE module and the new test module;
Black and Ruff pass for both Python files. Full-source CI remains required.

The [exact paired record](evidence/identity-layout-restoration.json) compares the
baseline restoration function loaded from integrated main
`5bc56a03cc584030e68e00b55ea75e45f9da36bd` with the changed function in the same
source tree. The implementation source hash records the uncommitted SABRE bytes
measured there. No other function differs between variants. The clean SDKBench
checkout is pinned to `9aa9260920984e2ca05bcb293abb885fccd9c144`; its cases,
adapters, budgets and independent evaluator remain unchanged.

Widths 6/8/12, five application families and four topologies are checked in both
ordinary restored and native target-artifact workflows: 120 pairs and 240
independent acceptances. Complete normalized operation/layout digests, evaluator
metrics and adapter audit counters match in every pair. All endpoint layouts
remain identity. Eighty six-wire executions check complete operators up to
phase; the remaining 160 executions check three probes only. These use the
previously inspected nonce, not new prospective inputs.

## Development measurements and limits

A separate direct restoration diagnostic uses two warmups and nine alternating
retained timing samples per variant, followed by separate Python allocation
observations:

| Identity layout width | Baseline / changed median | Python traced peak bytes, baseline / changed |
| --- | ---: | ---: |
| 128 | 108.000 / 6.083 microseconds | 26,832 / 9,096 |
| 512 | 1.080 / 0.029 milliseconds | 120,272 / 57,316 |
| 2,048 | 17.335 / 0.145 milliseconds | 510,800 / 267,940 |

This isolates the eliminated forest traversal. It does not measure complete
compilation, routing search, numerical execution or cross-SDK performance.

A separate complete `route_to_topology(..., strategy="sabre")` workflow routes
two H gates at the ends of a line, leaving the remaining wires idle. Fresh
coupling maps are constructed outside each timed call. After two warmups, seven
alternating retained samples give:

| Width | Baseline / changed median | Complete output |
| --- | ---: | --- |
| 64 | 0.427 / 0.386 milliseconds | Identical serialized digest; zero SWAPs |
| 128 | 1.486 / 1.373 milliseconds | Identical serialized digest; zero SWAPs |

These sparse local-gate cases expose identity-restoration work and are not
representative of dense interacting programs. The shared macOS host is not
qualified or reserved; no local tests overlapped these measurements. Timings,
Python traced allocations and one-shot application elapsed fields do not prove
statistical significance, process RSS, a general speedup, an SDK ranking or
scalability. Formal host-qualified SDKBench remains user-deferred. This change
does not resolve the separate native-CZ selector's resource-admission gap.

## Existing workflow and reproduction

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, route_to_topology

source = fq.Circuit(128).h(0).h(127)
routed = route_to_topology(source, CouplingMap.line(128), strategy="sabre")
assert routed.metadata["routing"]["inserted_swap_count"] == 0
assert routed.metadata["routing"]["final_logical_to_physical"] == tuple(range(128))
```

This compiles only; it allocates no exponential statevector and needs no hardware
or credentials. Nonidentity layouts still receive explicit restoration SWAPs.

```sh
python -m pytest tests/unit/test_compilation_identity_restoration.py \
  tests/unit/test_compilation_sabre_layout.py \
  tests/unit/test_compilation_sabre_routing.py -q
python tools/ci/ci_tier.py pr-default
python tools/ci/ci_tier.py pr-runtime
```

The archive retains the exact driver, every paired acceptance, normalized output
hash, all timing samples and allocation observations. Its temporary paths describe
the measured setup and must be adjusted elsewhere. Use one numerical thread and
a separate 600-second process-group supervisor with TERM/KILL cleanup; do not
install an outer signal alarm over SDKBench's own stage timer.
