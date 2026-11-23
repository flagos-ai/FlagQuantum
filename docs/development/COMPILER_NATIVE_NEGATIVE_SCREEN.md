# Reusing negative native synthesis classifications

## Requirement and ownership

Compiler owns this `single_device_fast_path` experiment. Complete target-CZ
route previews currently synthesize blocks whose exact matrices have already
been classified during shortlisting as requiring at least their existing CZ
count. The synthesizer emits that many CZ gates; the existing candidate path
rejects the raw proposal before cleanup. Reusing that negative classification
can remove redundant work without changing delivered routes or acceptance.
The existing Qiskit-derived KAK/Weyl implementation retains its attribution in
[the upstream reuse record](COMPILER_UPSTREAM_REUSE.md). This change reuses that
implementation's existing result; it does not port another external compiler.

This differs from the rejected fresh Weyl precheck: it performs no new
classification, does not affect incumbent realization and does not change
production rejection reasons. Reducing blocks still use actual synthesis,
operation-growth rejection, local cleanup and reconstructed residual checking.

## Implementation boundaries

The existing route-selection-local matrix-cost table is passed only to the
shortlisted alternative. Keys contain the exact ordered complex matrix bytes;
there is no rounding, cross-selection state or fixture-specific dispatch.
Only native unannotated RX/RZ/CZ blocks without matrix overrides qualify. A
known CZ count at least as large as the block's current count skips a proposal
only when the matrix passes the existing unitary predicate. Unknown costs,
empty tables, nonnative blocks, annotations and nonunitary inputs use the
original candidate path. The initial operation budget is checked first.

No production compiler source, public API, default, schema, SDK adapter,
dependency or acceptance threshold changes. The ordinary restored-layout
compilation path remains authoritative. Independent topology replay, artifact
construction and SDKBench's producer-free evaluation remain enabled.

## Prospective protocol, November 23, 2026

Publish implementation, tests and this protocol before application or latency
observations. Dependency: the bounded matrix-reader change in PR #1314 must integrate first.
The comparison base is `67c33e4c2bcd1510c82a04e8cac414c7c85e2b95`.
Freeze the base and implementation commits, source hashes,
driver hash, dependencies, compiler seed, options and input nonces. Production
source must match the base exactly. ASTs outside the exact matrix-key helper,
cost estimator, realization and route-choice function must match. Retain the
[untimed diagnostic census](evidence/native-negative-screen-census.json), including the discarded diagnostic whose counting
table outlived a route selection; neither census is latency evidence.

Use unchanged SDKBench `9aa9260920984e2ca05bcb293abb885fccd9c144`,
`flagquantum-target-artifact`, seed 11 and the 20,000-operation native growth
budget. Run every width 6/8/12, five-family, four-topology combination for both
variants in alternating order. Use the inspected nonce `01` repeated 32 times,
then a fresh 32-byte nonce frozen after publication. The fresh draw is not an
independently blinded held-out test. Preserve failures and attempts.

Require byte-identical operation/layout payloads, audits and metrics for every
old/new pair, identity final layout and unchanged evaluator acceptance. Width
six compares complete operators; widths eight/twelve use three existing probes
at `1e-8`. Replay every output without importing producer SDKs in a fresh
process. Any mismatch blocks admission.

After both quality draws and replay pass, measure complete adapter execution
for 6-wire QFT/grid, 6-wire repetition/ring and 12-wire QFT/grid. Include
production, old experiment and new experiment contemporaneously. Use two
warmups per variant, nine rotating three-variant rounds and one separately
traced Python-allocation round per variant following garbage collection.
Validate every output and retain all observations. No concurrent local tests.

Admit this experimental implementation only if every new/old median ratio is
at most `1.05`, every new Python peak is at most `1.10` times old and at most
128 KiB above old, and at least two controls have ratios at most `0.90`.
Report production comparisons separately. Experimental admission does not
admit a production default. No favorable rerun, retrospective threshold
change, dropped case or bypass of independent replay is permitted.

Bound each quality draw to 600 seconds, replay to 180 seconds, resources to
300 seconds and reports to 16 MiB, with process-group cleanup. Shared macOS
observations cannot certify the formal Linux protocol, RSS, significance,
SDK rankings or scalability. Formal qualification remains deferred. No reboot,
service restart or kernel upgrade is authorized.

## Verification and supported user workflow

Tests compare cached and original realization for nonreducing and reducing
blocks, reverse operand order, annotations, unknown costs, operation budgets
and nonunitary failure behavior. Full operators verify semantics. A positive
cost estimate never replaces synthesis acceptance.

The existing supported workflow is unchanged:

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

circuit = fq.Circuit(4)
circuit.h(0)
circuit.cx(0, 3)
compiled = compile(circuit, coupling_map=CouplingMap.line(4))
assert compiled.metadata["routing"]["mapping_restored"] is True
```

Expected: topology-legal compilation with restored logical output order.
This experiment does not enable native-cost route selection in that workflow.

Frozen complete-workflow control driver SHA-256: `b9f741ca24831b545087a8e8fc51da4b1f93c48d99df6eb9b06ffe9dc29c7e77`.
The evidence archive must retain the exact source. Before application
observations, all 280 focused unit/integration tests pass. Ruff, Black and
strict mypy pass both changed Python modules. Full final-head CI remains
required; the predecessor local full-suite origin failure is documented
separately and no whole-repository passing result is claimed.
