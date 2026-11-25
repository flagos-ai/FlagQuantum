# Reusing negative native synthesis classifications

Status: **rejected by the prospective resource filter**. The evaluated
implementation is retained in history and raw evidence; the benchmark
implementation is restored exactly to the comparison base. Only gate-class
boundary and nonunitary-failure regression tests remain.

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

## Evaluated implementation boundaries

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


## Completed observations and rejection

The implementation and protocol were published at
`80122e0dd48ffc1b560077f99fe7cc49e0ae8ed0` before application observations.
The [inspected draw](evidence/native-negative-screen-inspected.json) and
[fresh draw](evidence/native-negative-screen-fresh.json) each contain 120
accepted executions. All 120 baseline/candidate pairs have byte-identical
operation/layout payloads, audits and complete metrics. The fresh nonce is
`152df9bbb5aae0bcdf381fa6e5462a236a3f15e2cb8283fda05ae197a38ca373`.

[Producer-free replay](evidence/native-negative-screen-replay.json) accepts all
240 outputs without importing FlagQuantum, Qiskit or Cirq. Maximum residuals are
`6.072657351995283e-14` for the inspected draw and
`3.1447209754817776e-14` for the fresh draw. Each draw has 40 full-operator rows
at width six and 80 three-probe rows at widths eight/twelve. Probe checks do not
prove full-operator equivalence for the larger widths.

The [complete resource record](evidence/native-negative-screen-resources.json)
retains all nine timed observations per variant, every warmup and traced
observation, validated outputs, audits, source hashes and exact drivers.
The production implementation tree is unchanged from the verified integration
commit `959f25462c5e869a05b1957cc0cd68ef8900680c`. These local macOS
measurements do not certify the formal Linux benchmark protocol.

| Control | Production median ms | Baseline experiment ms | Candidate ms | Candidate/baseline | Candidate/production |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6-wire QFT/grid | 90.560 | 179.381 | 171.009 | 0.9533265 | 1.8883434 |
| 6-wire repetition/ring | 102.589 | 179.105 | 170.349 | 0.9511097 | 1.6604907 |
| 12-wire QFT/grid | 572.517 | 573.189 | 581.637 | 1.0147389 | 1.0159296 |

| Control | Production Python peak bytes | Baseline Python peak bytes | Candidate Python peak bytes |
| --- | ---: | ---: | ---: |
| 6-wire QFT/grid | 2199118 | 2268606 | 2263348 |
| 6-wire repetition/ring | 2567429 | 2600089 | 2599633 |
| 12-wire QFT/grid | 10118234 | 10118164 | 10118214 |

All memory bounds and the per-control maximum time ratio pass. **Zero controls
meet the required ratio of at most `0.90`; the prospective filter requires at
least two.** The control runner therefore exits 2 with `completed=true` and
`admission_passed=false`. There is no timeout, omitted case or favorable rerun.
The 12-wire control has no native-cost alternatives, so its timing difference
cannot be credited to the evaluated shortcut.

Correctness preservation and roughly 4.7%/4.9% local median reductions do not
justify retaining this additional implementation under the published rule.
The experiment is rejected; no default or production admission is proposed.
It remains about 89% and 66% slower than the contemporaneous production path
on the six-wire controls. Native-cost selection still needs a sufficiently
inexpensive complete implementation and unchanged independent qualification.
The number of skipped proposals in an untimed census does not establish the
fraction of total adapter latency those proposals consume.

Quality and resource archives store the production Git tree explicitly as a
tree object, not as a source revision. Their reconstruction notes restore the
original manifest field order and payload chunks. All three original reports
reconstruct byte for byte with their recorded SHA-256 hashes. All observed
values, failures, source identities and threshold decisions remain unchanged.
The evaluated implementation remains obtainable at the published commit above.

The retained version adds eleven gate-class boundary and nonunitary-failure
regression cases. All 254 focused unit/integration tests pass. Ruff, Black and
strict mypy pass both checked Python modules. The authoritative revision/origin
checker passes all eleven new evidence records, and local documentation links
resolve. The benchmark module is byte-identical to the comparison base, as is
the complete production tree. Full final-head CI is still required before
integration; no whole-repository passing claim is made.

After PR #1313 integrated, merge the updated prerequisite/main baseline while
retaining both independent test additions. All 259 combined focused
unit/integration cases, Ruff, Black and strict mypy pass. The evaluated benchmark
source and production tree remain byte-identical to the comparison base. This
integration update requires fresh final-head CI.
