# Application-stage replay and rejected operand-order widening

Status: inspected development diagnosis; the candidate is rejected for default
use. Owner: Compiler. Classification: `single_device_fast_path`. No production
compiler source, public API, routing objective or acceptance threshold changes in
this record. The archived implementation is an experiment, not a supported pass.

## Why the two existing workflows differ

The prospective study in [PR #1298](https://github.com/flagos-ai/FlagQuantum/pull/1298)
selected twelve-wire QFT on a sparse graph as the next ordinary-workflow witness.
Its runtime is integrated main `2ea5b78b6449b837db151e0d29c90895f6b51a4e`;
SDKBench remains unchanged at `9aa9260920984e2ca05bcb293abb885fccd9c144`.
The retained [stage replay](evidence/compiler-application-stage-replay.json)
records both that witness and six-wire repetition on its sparse graph, using
that study's single frozen nonce. These are inspected inputs, not a new unseen
campaign. Wrappers observe the original functions and return their results
unchanged. Additional legalization for stage accounting uses the same declared
abstract CZ target; it is outside the original adapter timing interval.

| QFT boundary | Source spelling | CZ count after legalization | Two-body depth |
| --- | --- | ---: | ---: |
| Ordinary source | 12 H, 132 RZ, 66 RZZ, 18 CX | 150 | 45 |
| Ordinary pre-routing optimization | 12 H, 22 RZ, 66 RZZ, 18 CX | 150 | 45 |
| Ordinary route | Above plus 88 SWAP | 414 | 193 |
| Ordinary post-routing optimization | Same counts | 414 | 193 |
| Target-artifact route from original source | Original plus 82 SWAP | 396 | 182 |
| Target-artifact native block output | 34 accepted rewrites | 328 | 156 |

The source optimization merges local RZs without reducing the two-qubit work.
The resulting instruction dependencies change the router's lookahead and chosen
trial. A separate replay of the same numeric IR before/after optimization confirms
82 versus 88 restored SWAPs. Thus the 18-CZ pre-stage difference is six extra
SWAPs, while the subsequent native stage saves another 68 CZs. This identifies
pipeline sensitivity; it does not establish that RZ merging is generally harmful,
that disabling optimization is a fix, or that all differences come from synthesis.

The archived per-trial accounting retains every existing deterministic trial,
three tie seeds and the layer-lookahead trial. For raw QFT the layer trial uses
62 forward and 20 restoration SWAPs, winning at 82. After source optimization it
uses 62 plus 30; the seed-2 trial wins at 61 plus 27, or 88. Forward routing and
final restoration must both be assessed.

Repetition starts with 24 CXs. Ordinary source/post-routing optimization does not
remove any. Both workflows insert 12 SWAPs and deliver 60 CZs/depth 57; the target
stage accepts zero rewrites. The incumbent uses ten forward plus two restoration
SWAPs. The layer trial has seven forward plus five restoration SWAPs: reducing
forward routing alone does not reduce delivered cost. The independent six-wire
acceptance compares the complete operator. None of this proves a globally optimal
route or that additional two-qubit synthesis alone closes the gap.

## Reused mechanism and candidate scope

The generic collector currently splits reversed operand orders into distinct
blocks. The existing private target-CZ collector already handles them by
conjugating the matrix with the basis-index permutation `(0, 2, 1, 3)`.
The candidate reuses that exact mechanism in the common matrix reader, allows
mixed orders in generic blocks, and removes the now-duplicated native-stage
conversion. Control/target roles remain explicit. A declined widened block reruns
the original ordered collector once, preserving its local-gate folds. Routing
annotations and trainable parameters retain their existing refusal boundaries;
there is no cross-call cache or new dependency.

This follows the existing Qiskit/Cirq collect/consolidate pattern already owned
by these modules; it is not a new external compiler port. The complete modified
sources, source hashes, patch, tests, stage runner and trial diagnostic are in the
[candidate archive](evidence/compiler-operand-order-candidate.json). Its baseline
is integrated main `97c8da4690fb7ce3a606a1e24a94277e843b299c`, which has the
same compiler implementation as the prospective study runtime. The screen
verifies the two compiler source hashes against that archive. No archived code is
imported or admitted by the production optimizer.

The candidate passes 85 focused optimization/operator/gradient tests and 231
native-stage, physical-evidence, routing, decomposition and splitting tests.
Coverage includes all 1,554 words of length one through four over both operand
orders of CX/CZ/SWAP, exact full-operator comparisons including phase, 96 mixed
three-wire circuits, trainable gradients and mutable-matrix rereading. The
ordered collector is also an explicit non-lengthening reference.

The initial 40-test invocation had 38 passes and two failures. One old test
required mixed-order CZs to remain separate; its successor explicitly checks
correct symmetric cancellation and asymmetric control/target preservation against
the independent runtime operator. A new test initially demanded one-call
idempotence; fallback can expose a subsequent shorter fold, so its corrected
assertions require strict shortening, exact operator preservation at every step
and bounded fixed-point convergence. Both logs are retained. No acceptance
threshold, test count or application input was reduced.

## Complete fixed application screen

The [raw screen](evidence/compiler-operand-order-screen.json) retains all 120
outcomes in five independently hashed application-family files: all 60 inspected
width/family/topology inputs from PR #1298, the same
ordinary and target-artifact workflows, and seed 11. FlagQuantum does not support
native compiler seeds here, so this does not select a favorable stochastic seed.
Original SDKBench deadlines, 20,000-operation cap, normalization, target edges,
identity final placement and `1e-8` semantic threshold are unchanged. Every
outcome is accepted; no exception or unsupported outcome is dropped.

| Workflow | Equal count/depth | Improves without regression | Regresses without improvement | Count/depth tradeoff |
| --- | ---: | ---: | ---: | ---: |
| Ordinary restored | 49 | 6 | 3 | 2 |
| Target artifact | 60 | 0 | 0 | 0 |

Every changed ordinary outcome is QFT:

| Width / topology | Baseline CZ / depth | Candidate CZ / depth |
| --- | ---: | ---: |
| 6 / line | 117 / 74 | 111 / 72 |
| 6 / ring | 93 / 60 | 87 / 60 |
| 6 / grid | 75 / 52 | 69 / 42 |
| 6 / sparse | 105 / 76 | 111 / 78 |
| 8 / line | 224 / 120 | 224 / 121 |
| 8 / ring | 188 / 95 | 182 / 102 |
| 8 / grid | 176 / 110 | 170 / 112 |
| 8 / sparse | 164 / 102 | 164 / 86 |
| 12 / ring | 438 / 165 | 432 / 162 |
| 12 / grid | 408 / 191 | 420 / 193 |
| 12 / sparse | 414 / 193 | 396 / 181 |

Twelve-wire QFT/line and every non-QFT input are unchanged. Repetition's selected
counterexample remains unresolved. Favorable source folding on the selected
QFT witness does not justify three regressions elsewhere. The existing native
collector already handles reversed operands, explaining why this candidate
leaves all target-artifact results unchanged. The experiment is not enabled in
the default path, nor presented as a remedy for the native target-workflow gap.

## Independent replay, resources and reproduction

A separately started process imports no FlagQuantum, Qiskit or Cirq. It replays
all 120 archived candidate payloads and all 12 routed stage payloads through the
unchanged evaluator, with exact metric agreement. Its [proof and complete script](evidence/compiler-operand-order-replay.json)
retain both original artifact digests. A storage-only reconstruction verifies
that the family files restore every original screen byte and compact stage JSON
restores every original pretty-printed stage byte; no numerical values change.
Maximum residual is `4.40152024415209e-14`.
Pre-routing stage records show legalization cost only; they do not claim target
edge legality before routing. Six-wire candidate outputs use complete operators;
widths eight/twelve use three probes and do not prove universal equivalence.

Collection and replay each run in bounded, supervised process groups with one
numerical thread, unchanged stage deadlines and retained terminal supervision.
The shared macOS host does not qualify formal cgroup peak memory, exclusive
physical cores, stable latency, hardware behavior, publication eligibility or
scalability. No time/memory admission is claimed. Formal SDKBench qualification
remains deferred; no machine, service or kernel was restarted or upgraded.

The archived stage output renames its initial `runtime_commit` field to the
existing `implementation_commit` vocabulary and adds the SDKBench source pin.
The independent replay reads the final archived JSON. Numerical observations
are unchanged; the original proof and original runner remain in the candidate
archive with their original digests.

For reproduction, use disposable source checkouts at the named baselines and a
clean frozen SDKBench checkout. The candidate archive's `sources` entries are
complete file contents; write them only into the disposable candidate checkout,
then execute its archived tests and screen script with the recorded manifest.
The final replay script also records its exact input paths. Relocate those paths
to the corresponding archived JSON files when reproducing elsewhere. These are
fixed inspected controls, so rerunning them never creates unseen evidence.

The existing user workflow supported by this diagnosis remains:

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

source = fq.Circuit(3).cx(0, 1).cx(1, 0).cx(0, 1)
compiled = compile(source, coupling_map=CouplingMap.line(3))
assert compiled.metadata["routing"]["mapping_restored"] is True
```

Expected: equivalent legal compilation with restored placement, using the
unchanged production optimizer. The archived experiment does not change this
workflow's implementation or promise a shorter result.

The next production change must address the delivered routing/native-cost
interaction and retain restoration, independent verification and resource gates.
Neither this failed widening nor the previously resource-rejected native-cost
selector may be enabled by default on the strength of one improved witness.

Archive validation checks all new file sizes, revision origins, field vocabulary,
English and local documentation links. Sparse-checkout evidence-suite attempts
initially passed 54 and then 56 tests while missing tracked corpus files. After
materializing the complete corpus, 57 tests pass; one ambient-history origin
check remains unqualified because this development repository has additional
refs resolving 16 historically declared revisions. Its declarations and
assertions are unchanged. These attempts are not full-suite passes; exact-head
CI must qualify the complete pull request.

## Subsequent two-body layer projection: also rejected

Cirq 1.7.0's `RouteCQC._get_one_and_two_qubit_ops_as_timesteps` assigns local
operations around a circuit of two-qubit timesteps. Reviewing that existing
upstream mechanism exposed a narrower FlagQuantum difference: mixed-instruction
ready batches can split an otherwise unchanged two-body layer when local gates
are subdivided. The additional candidate drains ready nonblocking dependencies
before scoring the ready two-qubit layer, using the existing DAG indegrees and
unchanged eight-layer/160-operation/32-candidate bounds. Within that budget,
local subdivisions cannot split the projected layer. Classical joins still wait
for all predecessors. Pure two-qubit projection and the existing Cirq-derived
ranking mechanism remain unchanged. No new dependency or public contract is
introduced; the existing Cirq Apache-2.0 attribution remains with the source.

The [complete candidate and tests](evidence/compiler-layer-projection-candidate.json)
are frozen for inspection. All 21 focused tests pass, including 300 seeded
local-subdivision comparisons across widths 4/6/8, input immutability, classical
joins, unchanged projection/search caps, existing Cirq source comparisons and
shared-gradient routing. This is a projection invariant, not proof that delivered
native compilation improves.

The [complete fixed screen](evidence/compiler-layer-projection-screen.json)
uses the same 60 inspected inputs, seed 11, both workflows and unchanged
SDKBench acceptance. All 120 outputs pass. Ordinary compilation has 59 equal
results and one improvement: twelve-wire Ising/ring depth 86 to 85, with 168 CZs.
Target-artifact compilation has 52 equal, three improvements, three regressions
and two tradeoffs. Every changed native result is retained below.

| Target-artifact input | Baseline CZ / depth | Candidate CZ / depth |
| --- | ---: | ---: |
| 6 / QFT / line | 75 / 52 | 103 / 68 |
| 8 / QFT / line | 156 / 77 | 184 / 97 |
| 8 / QFT / ring | 142 / 78 | 134 / 73 |
| 12 / Ising / ring | 162 / 82 | 162 / 81 |
| 12 / QFT / line | 396 / 137 | 398 / 135 |
| 12 / QFT / ring | 322 / 131 | 330 / 114 |
| 12 / QFT / grid | 352 / 150 | 346 / 144 |
| 12 / QFT / sparse | 328 / 156 | 402 / 194 |

The selected ordinary QFT/sparse witness and both repetition workflows remain
unchanged. This projection candidate is also rejected for default use, and its
code is removed from executable production paths. Improving representation
invariance or the restored-SWAP objective alone does not guarantee a better
post-synthesis native cost. A prospective release study or resource measurement
would not cure these already-observed quality regressions.

The [independent replay](evidence/compiler-layer-projection-replay.json)
checks all five family-file hashes, complete input/workflow accounting, exact
reconstruction of the original observation bytes, and all 120 unchanged evaluator
results without importing producer SDKs. The same inspected-input, probe-only
larger-width and shared-host limits apply; this adds no latency, memory or SDK
ranking claim. Both candidate mechanisms remain rejected, rather than being
selected according to their best examples.
