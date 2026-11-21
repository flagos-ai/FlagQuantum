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


## Verification with the merged compiler pipeline

PR #1268 merged at main `a7d84a605080c511d4a07037ef2b4f3b90a645d1` after
243 successful checks and three optional skips. This branch merged that exact
revision without conflicts. The combined production source is
`7619bbe25f034570ad01c495cab6a4bbfb0744b9`; target-plan reuse remains excluded.
Earlier reports retain their original source pins and numerical observations.

The unchanged collector-control runner produced a
[new complete report](evidence/compiler-cz-identity-prefix-integrated.json):
all 60 pairs / 120 independent acceptances pass, with the same one-case CZ
improvement and 59 unchanged CZ/depth pairs. Two warmups and 21 alternating
samples per variant give local historical-collector/prefix medians of
165.695/165.690 ms for Ising-8/line and 619.430/619.457 ms for QFT-12/grid.
These almost equal shared-host observations do not prove zero overhead or a
stable speedup. The previous campaigns ran different source trees and intervals;
their medians must not be divided to attribute a pipeline speedup. Separate
traced Python peaks are 2,896,522/2,984,180 bytes for Ising and
10,116,749/10,116,631 bytes for QFT (control/prefix). The Ising observation
increases by 87,658 bytes; these are single traced runs, not process-RSS or
qualified memory comparisons.

The combined focused suite passes 962 tests with four optional skips. It covers
classical feedback, scalar encoding, target lowering, generic and native block
optimization, routing, complete artifact evidence, directed topology and the
CPU workflow. An earlier collection attempt lacked tracked synthesis examples;
restoring those sparse-checkout fixtures enabled the full selection. These
focused tests are not the full repository tiers; synchronized-head CI is required.

## Rechecking the previously rejected depth tie policy

The [earlier depth study](COMPILER_SABRE_DEPTH.md) rejected an equal-SWAP
candidate because the six-qubit grid QFT target output increased from 65 to
67 CZs. That historical decision and raw report remain intact. Prefix splitting
changes the downstream interaction, so the complete application study was
rerun at the combined source above. The diagnostic runner now records its
existing-vocabulary `implementation_commit` as well as source/runner digests;
this adds measurement provenance, not a production compiler setting.

The [new application report](evidence/sabre-depth-after-identity-prefix.json)
retains 120 workflow/input pairs / 240 independent acceptances across both
ordinary-restored and target-artifact workflows. It has six lexicographic
CZ-count/depth improvements, 114 unchanged pairs and no observed regression.
The QFT target witness now changes from 65 CZs / depth 48 to 63 CZs / depth 38;
its ordinary counterpart changes from 75 / 52 to 75 / 42. All existing endpoint
layouts and acceptance thresholds are unchanged.

This is a diagnostic override of the existing trial selector, not a registered
formal profile or an enabled default. Inputs use the already-inspected nonce;
these are not unseen draws. The selector also extends the trial's pruning bound
by one to retain equal-cost candidates, so search overhead needs measurement.
Quality on these inspected applications does not qualify latency, memory,
held-out behavior or SDK rankings. The policy remains disabled pending those
checks; there is no special-case treatment of the QFT witness.


The [rerun generic control](evidence/sabre-generic-depth-after-identity-prefix.json)
also completes all 80 seeded CX cases at widths 6/8/12/16 on line/grid devices,
using the existing exact GF(2) operator checks. Its reported logical two-qubit
depth counts CX and SWAP equally, whereas the diagnostic selector expands each
SWAP to weight three. That generic depth is not the delivered native CZ metric,
and must not be substituted for the application measurements above.


## Cold-process import validation

The synchronized head's coverage shard rejected the collector runner's
package-level submodule import. The import worked when executing the runner,
but the existing benchmark import guard checks attributes of a freshly imported
package without loading its children. The runner now imports both owning modules
by their complete paths, so it no longer relies on package attributes populated
by an earlier import. The guard and production compiler are unchanged. A fresh
process passes all four import-policy tests and all 42 native physical-evidence
tests. Historical reports retain their recorded runner hashes; they describe
the runner revisions that actually produced them. Full synchronized CI remains
required before merge.

The [complete rerun after the import correction](evidence/compiler-cz-identity-prefix-import-validation.json)
records implementation `90bd5c75f5b98d3109b10f5a71a877954f01db99` and the
updated runner digest. All 60 pairs / 120 acceptances pass. Every normalized
output hash, layout and non-timing/non-memory observation matches the prior
integrated report for both variants. The report retains 84 post-warmup latency
observations and four separate Python-allocation measurements. Another local
workflow ran concurrently during part of this interval; these observations are
shared-host diagnostics, not performance qualification or a speedup claim.
