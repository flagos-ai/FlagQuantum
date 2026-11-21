# Bounded native CZ cancellation prefixes

This development increment changes the native target compiler's private block
collector. It preserves the existing public workflow, numerical acceptance,
64-input rewrite bound, v4 evidence schema and identity endpoint layouts. Its
path classification is `single_device_fast_path`; it makes no scalability claim.

## Problem and implementation

A numeric pair block can contain adjacent inverse routing or logical SWAPs.
Their native CZ expansion can exceed 64 leaves. Splitting that stream at the
64th leaf can cut across an independently cancelling prefix and leave avoidable
CZs in the delivered program. Raising the protected evidence bound would not
solve the underlying choice of split point.

When a native block reaches the existing 64-member capacity, the collector now
looks for its earliest prefix containing at least two two-wire members whose
4-by-4 product passes the existing identity-up-to-phase check. If found, that
prefix closes and the remaining members stay in the open pair block. Otherwise,
the original capacity split applies. Short blocks retain their collection path
without this matrix scan. Disjoint operations, barriers, symbolic values,
metadata admission and exclusive ownership follow the existing collector rules.

Prefix detection only proposes a boundary. Leading local operations still join
the closed block, and the existing native synthesizer must verify the complete
block and accept a strict CZ reduction. Native lineage, artifact verification,
serialization, topology and budget checks remain mandatory. No operation is
removed solely because the prefix detector reports identity.

The scan uses at most 64 matrix products per capacity event, one 4-by-4 product
and a bounded suffix list. It adds no dependency, global cache, external SDK IR,
public option or second numerical synthesis implementation. Generic private
`resynthesize_cz_blocks` keeps the original collector behavior. This is an
FQ-owned collector refinement, not a new Qiskit/Cirq code port; existing reused
mechanisms and attribution remain in [the upstream audit](COMPILER_UPSTREAM_REUSE.md).

## Independent development controls

The [runner](../../benchmarks/compiler_cz_identity_prefix.py) records frozen
candidate source `283ae81430c35fbcf030360f052d94f20ec5fb18` and unchanged SDKBench
`9aa9260920984e2ca05bcb293abb885fccd9c144`. The control dynamically loads the exact
historical collector from main `2d78133a1f5a7d234acbb128724ccf2674e2eeb8` and
replaces only the native collection boundary. This isolates collection inside
the same complete target-artifact workflow; it is not a pristine historical
compiler process or a registered formal participant profile. Pending pipeline
integration and target-plan reuse changes are excluded.

The [complete raw report](evidence/compiler-cz-identity-prefix.json) retains all
60 paired cases: QAOA, Ising, hopping, QFT and repetition; widths 6/8/12;
line/ring/grid/sparse coupling; inspected nonce `01` repeated 32 times. Both
variants pass all 120 independent acceptances with unchanged targets, caps and
identity endpoint layouts. Width six uses complete operators; widths eight and
twelve use three probes only. These cases were inspected during development and
are not held-out qualification.

| Delivered quality | Historical collector | Capacity-only prefix collector |
| --- | ---: | ---: |
| Repetition, 12 wires, grid: full CZ count | 104 | 100 |
| Same case: two-body dependency depth | 68 | 68 |
| Other 59 cases: CZ count and dependency depth | Unchanged | Unchanged |

Four artifact regressions exercise four inverse SWAPs, reversed operands,
leading RX and an interleaved disjoint RZ, both with and without topology
routing. Old production code leaves four CZs and fails all four new assertions;
the candidate emits zero CZs. Complete operators, phase, retained input lineage,
each rewrite's 64-member bound, v4 round trips and independent evidence
verification pass. Existing rejection and budget scenarios remain in the
375-test focused compiler/artifact set.

## Time, memory and rejected implementation

An initial eager detector at source
`94c838b0fb51088b4c6d39a32923c78e5ec734d2` checked every admitted member. Its
[retained report](evidence/compiler-cz-identity-prefix-eager.json) has 120 accepted
application outputs but no CZ/depth improvement in those cases. Local median
compilation intervals increased from 205.491 to 212.293 ms for Ising-8/line and
744.284 to 761.648 ms for QFT-12/grid. That design was replaced with capacity-only
inspection, rather than dismissing its overhead.

The final report uses alternating variant order, two warmups and 21 retained
samples per variant and case. Timings include native artifact verification and
serialization, but exclude independent acceptance, startup and fixture setup.
Separate traced-memory runs do not contribute timing samples.

| Local observation | Historical collector | Capacity-only prefix collector |
| --- | ---: | ---: |
| Ising-8/line median, ms | 204.594 | 203.774 |
| QFT-12/grid median, ms | 737.030 | 742.578 |
| Ising-8/line Python allocation peak, bytes | 3,315,700 | 3,315,554 |
| QFT-12/grid Python allocation peak, bytes | 10,119,125 | 10,116,631 |

This shared macOS process is not an exclusive qualified host. Other development
checks may overlap the campaign. The small final timing differences do not
establish a speedup or absence of overhead. Tracemalloc observes Python
allocations, not process RSS, native allocation peaks or the formal executor's
memory metric. The two campaigns used distinct intervals; their medians must
not be divided to claim a measured optimization speedup. There is no SDK ranking,
physical-duration, hardware, production, release or superiority claim.

## Reproduction and user impact

With the pinned, unchanged SDKBench checkout and FlagQuantum dependencies:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python benchmarks/compiler_cz_identity_prefix.py \
  --sdkbench-root /path/to/sdkbench --output /tmp/cz-prefix.json
```

The runner writes checkpoints, retains exceptions, attempts the full case set
and exits nonzero if any observation fails. Use an external bounded watchdog;
the recorded runs had a 360-second total budget and 60-second no-output budget.
This local driver does not implement the deferred formal resource protocol.

Existing user workflow:

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

circuit = fq.Circuit(4).h(0).cx(0, 3)
program = compile(circuit, coupling_map=CouplingMap.line(4))
assert program.metadata["routing"]["mapping_restored"] is True
print("Restored logical output layout")
```

Expected: the message prints. This ordinary compiler entry remains unchanged;
the improvement applies in its existing native target-artifact counterpart.
It does not enable a new route-selection policy. The target artifact tests use
the existing workflow in
[the integration scenarios](../../tests/integration/test_native_block_physical_evidence.py).

Local verification: 375 focused tests pass; the sparse checkout's default tier
selects 317 passing tests and runtime tier selects 58 passing tests. These are
limited local selections, not the full repository tiers. Black, Ruff, public API
snapshot, architecture, English, document links and whitespace checks must pass
before publication; full repository validation remains required CI.
