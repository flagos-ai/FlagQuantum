# Native compiler optimization

Status: implementation in progress; no performance parity or superiority claim.
Execution classification: `single_device_fast_path`. Compiler work partitions no
numerical workload and establishes no distributed scalability claim.

## Objective and ownership

Combine the strengths of mature quantum compilers, including Qiskit and Cirq,
within FlagQuantum-owned IR and compilation stages. Reuse existing algorithms,
implementation components, and dependencies before introducing new mechanisms.
Independent workload evidence, including regressions and unsupported cases,
determines whether an implementation improves the product.

Compiler owns program transformation. Core owns IR and target contracts;
Deployment owns executable packaging; Simulation supplies execution references.
SDKBench remains an independent evaluator with participant-neutral acceptance.
Changes to protected contracts require their own approved integration change.

## Existing components reused

| Requirement | Existing authority | Work in this slice |
| --- | --- | --- |
| Target-independent optimization | `compiler/pipeline.py` | No built-in optimization pass changes |
| SABRE search | `compiler/topology/sabre.py` | No second SABRE implementation |
| Automatic routing | `compiler/topology/routing.py` | Existing candidate selection reused as the approved native compile default |
| Device and source routing costs | `compiler/topology/routing_cost.py` | Shared channel, topology, and orientation counting remains beside the existing device cost model |
| Compilation orchestration | `compiler/lowering/pipeline.py` | Retain stage ownership and router replacement through the entry facade |
| Numeric instruction matrices | `compiler/optimization/two_qubit_optimization.py` | Reuse matrix reading and multiplication; normalize reversed operands |
| Two-qubit KAK synthesis | `compiler/optimization/two_qubit_synthesis.py` | Reuse existing CZ synthesis |
| Basis cost | `compiler/lowering/basis_translation.py` | Price the actual per-instruction CZ decomposition |
| Explicit transformation admission | `compile(..., passes=...)` | Exercise the existing extension point without adding a public API |

No external code has been copied and no production dependency has been added
in this slice. The existing synthesis implementation retains its source history
and attribution. Any later component import requires a separate license,
maintenance, supply-chain, boundary, and replacement review. Reusing an external
compiler wholesale is not a substitute for integrating proven mechanisms into
FlagQuantum's own pipeline.

### Mature compiler review

Documentation reviewed on 2026-11-18:

| Source | Relevant mechanism | Adoption decision |
| --- | --- | --- |
| [Qiskit SabreSwap](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.SabreSwap) | Frontier candidates, lookahead and decay; documented LightSABRE enhancements and multiple trials | Keep the existing FlagQuantum-owned planner first. It is deterministic and is not an implementation-equivalent substitute for Qiskit's enhanced search. Evaluate search changes separately from the default-policy change. |
| [Qiskit ConsolidateBlocks](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.ConsolidateBlocks) | Collect local blocks, consolidate their operators, and resynthesize against a target | Reuse the existing native numeric reader, basis translation, and KAK implementation. Keep local block ownership in Compiler and verify each accepted candidate. |
| [Cirq CZTargetGateset](https://quantumai.google/reference/python/cirq/CZTargetGateset) | Full or partial CZ targets, two-qubit matrix decomposition, and optional moment-structure preservation | Admit full CZ only in this slice. Preserve instruction dependencies and treat total gate count/depth as separate outcomes. Do not assume this pass reproduces Cirq's target optimizer or scheduling policy. |

These are mechanism references, not performance evidence. No claim that the
implementations or defaults are identical follows from their shared use of KAK
or SABRE. Cirq's documented numerical tolerance and negligible-operation
postprocessing are not imported into generic FlagQuantum optimization; this
slice has its own explicit synthesis residual bound and preserves the existing
exact-identity policy outside this opt-in target transformation.

## Call-local routing plan reuse

Native `compile` retains the SABRE plan produced when pricing candidates and
materializes that same plan if SABRE wins. This removes a second identical
search within one automatic compilation. Selection objectives, candidate order,
calibration exclusions, and final-layout restoration are unchanged. There is no
cross-call plan cache: a different program or device is planned independently.
Separate public calls to `select_routing_strategy` and `route_to_topology`
continue to plan independently. A caller-supplied facade router remains in
control and receives the selected strategy; its planning work cannot be reused
implicitly. Path-cache telemetry records the operations
actually performed; fewer planner calls are not a measured wall-time speedup.

Regression tests compare automatically routed instructions with the explicit
selected strategy, check statevector equivalence and deployment-plan replay,
and exercise changed programs, changed devices, and calibrated-device exclusion.

## Bounded CZ block transformation

`compiler/optimization/_cz_blocks.py` is a private, opt-in transformation for a
full-CZ target with arbitrary RZ and RX rotations. It collects static operations
over a pair, allowing commuting operations on disjoint qubits between members.
Two open pairs cannot overlap. A different pair sharing a qubit closes the old
block. Reversed operands are converted into the same ordered matrix basis.
When a block closes, it can draw in contiguous, readable leading one-qubit
operations on its pair, using the existing native collector's exclusive-membership
rule. Operations already owned by an open or closed block cannot be claimed again.
Leading members use only remaining space within the 64-operation bound, preserving
the existing entangling-block partition rather than closing it earlier.

Each candidate must strictly reduce CZ count relative to recursive native-basis
translation. Already-native instructions contribute their actual cost without
requesting a rewrite. Both the translated baseline and synthesized candidate
must contain only target opcodes; an unresolved basis leaf cannot be counted as
a zero-cost interaction. Its reconstructed 4 by 4 operator must match the source within
`1e-11` per entry after accounting for one unit-modulus global phase. The
candidate may add at most 32 instructions relative to the translated baseline.
This objective can trade single-qubit gates and depth for fewer CZ gates; it
does not claim to minimize device duration, error, or total instruction count.

Each block has at most 64 source instructions. Matrix storage is constant per
block; output and replacement storage grow with program size. The number of open
blocks is at most half the qubit count. This bounds local numerical synthesis,
not whole-process wall time or memory. External evaluation must still enforce a
process timeout and memory budget. KAK work is attempted only for blocks whose
translated baseline contains at least two CZ gates.

Annotations, custom matrices, nonunitaries, trainable or batched angles, and
operations wider than two qubits form boundaries. Zero-qubit directives close
all blocks. Annotated routing SWAPs are retained, so layout evidence is never
invalidated by deleting a movement operation. Program metadata, observables,
measurements, and final assignment are retained; a logical SWAP is never
absorbed into an implicit final permutation.

### Shortest developer path

Run from the repository root with the project dependencies installed:

```python
from flagquantum.compiler import compile
from flagquantum.compiler.optimization._cz_blocks import resynthesize_cz_blocks
from flagquantum.core.ir import CircuitIR, Instruction

source = CircuitIR(2, (
    Instruction("swap", (0, 1)),
    Instruction("cx", (1, 0)),
))
result = compile(
    source,
    optimization_level=0,
    passes={"experiment.cz": resynthesize_cz_blocks},
)
assert sum(op.name == "cz" for op in result.instructions) == 2
```

This is a private developer experiment, not a new supported user API. The
unchanged per-instruction CZ translation costs four CZ gates for this source.
The two-CZ result preserves the identity final assignment and the full operator
up to global phase. It does not establish a workload-level speedup.

The focused checks are:

```bash
python -m pytest tests/unit/test_compilation_target_cz_resynthesis.py -q
```

The tests independently embed Simulation gate matrices into the complete
register, including reversed operands and disjoint spectators. They also check
incorrect candidate rejection, instruction-growth rejection, long blocks,
trainable parameter gradients, ordering boundaries, and the existing explicit
compiler-pass workflow. They require admitted outputs to use the target basis,
including after per-instruction native legalization, and reject an equivalent
candidate containing non-target interactions. Compiler matrix reconstruction
is not the test oracle.

## Limits and remaining integration

- The pass preserves semantics up to global phase. IR currently has no
  authoritative global-phase field. Do not admit it to generic `optimize`,
  phase-sensitive controlled construction, or an exact-amplitude contract.
- Conservative annotation boundaries mean it does not yet consolidate routed
  SWAPs with neighboring gates. This slice does not resolve the complete routed
  block optimization issue. Integration must construct correct layout and
  source-to-output evidence, rather than strip metadata to unlock a rewrite.
- Leading one-qubit collection is contiguous and conservative: it stops at an
  already claimed operation, annotation, unreadable value, or operation outside
  the pair. It does not perform general commutation or symbolic synthesis.
- Only full CZ cost is implemented; arbitrary target gates, calibration-aware
  cost, symbolic synthesis, and approximate synthesis are not admitted.
- Automatic routing already considers deployable SABRE. The approved native
  compile default is now `auto`; explicit old-policy calls remain supported.
  See the separate API proposal and authorization record.
- `fq.compile` still rejects native target selection; the existing expert route
  is `flagquantum.compiler.compile`. No new root workflow is claimed here.

Next evidence must compare explicit existing routing strategies and the new
transformation on held-out tasks using unchanged SDKBench acceptance. Report
target basis, topology, initial and final layout policy, versions, seeds,
optimization settings, CZ count, dependency depth, compile time, memory,
timeouts, failures, and unsupported cases. Separate default-policy effects from
algorithm and adapter effects. Gate savings alone do not establish superiority.

## Development verification on 2026-11-18

Environment: macOS arm64, Python 3.12.14, PyTorch 2.14; CPU execution with
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. This is development correctness
evidence, not SDKBench resource-isolated performance evidence.

- Compiler unit files (`tests/unit/test_compilation*.py`), Compiler team tests,
  deployment routing evidence, routing candidate contracts, and the optimization,
  weighted-routing, synthesis, target-aware, and artifact-compilation integration
  golden paths: **2,759 passed** in 66.85 seconds after correcting target-basis
  admission and integrating main's
  compiler stage refactor at `2219064cee1e90a19a71239e9f22d1cb2aa36686`.
  Router replacement remains effective through the facade; the native fast path
  still searches SABRE once. The 28 pass-admission contract tests also passed.
- After adding leading-local collection, **2,764 distinct compiler, deployment,
  routing-candidate, and focused integration tests passed**: 2,753 in the main
  focused run (52.45 seconds) plus 11 routing-plan candidate cases. The 28
  pass-admission contracts also passed. Five new cases verify both operand
  orders, exclusive ownership across neighboring pairs, annotation preservation,
  and the shared resource bound; the CZ file now has 36 passing cases. All eight
  changed Python files pass Black 26.10.1 and Ruff. These focused results do not
  replace the incomplete broad qualification described below.
- Vocabulary census and vocabulary contracts: **173 passed** after correcting a
  new comment's terminology and selecting this checkout's Git work tree.
- Public API migration baseline and the target-aware compilation example passed.
  Ruff lint and formatting checks passed for the changed Python files.
- Implementation PR #1219's next Linux quality and dependency-bounds jobs
  stopped at Black 26.10.1 formatting in the new default-routing test, before
  reaching later checks. The formatting was corrected; Black now accepts all
  eight changed Python files, and the 45 default-routing and CZ-block tests pass.
  This correction does not resolve the separate evidence-origin failure below
  or establish that the dependency-version test suites have passed.
- The following Linux quality run reached the architecture checker and rejected
  `compiler/topology/routing.py` at 1,285 lines against the unchanged 1,250-line
  ceiling. Three unchanged private pricing functions moved into the existing
  `routing_cost.py`, leaving the router at 1,220 lines and the cost module at 714.
  Public types, signatures, planner replacement, and serialized identities remain
  in place. The architecture check passes; no ceiling was raised or waived.
  All 2,764 focused compiler, deployment, routing-candidate, and integration cases
  passed again after this move in 50.23 seconds.
- The earlier full smoke/unit run was **not green**: 20,344 passed, 15 failed,
  6 errors, and 785 skipped. The routing assertion was migrated to the explicit
  approved previous policy and subsequently passed in the compiler suite.
  The terminology regression and incorrect Git work-tree selection were
  diagnosed and corrected, rather than updating expected vocabulary counts.
- Six launch-plan failures and six MPS process-group setup errors reproduced
  on the unchanged baseline `95f839f8fd43641143caaa5cca93f715cbf9183e`
  in the same sandbox: port binding was denied and Gloo could not resolve a
  local address. CPU process-group setup errors are not scalability evidence.
- The checked-in evidence revision gate still fails against the local clone's
  available history/origin declarations. It has not been waived or repaired in
  this compiler change. No full-suite passing claim is made.
- Authorization PR #1218's Linux `quality` job reported 92 passed and one
  failure: `artifacts/development/qec_colour_distance_acceptance_20261118.json`
  records a producing revision without an obtainable reference or an origin
  declaration. This is a separate observed CI failure, not a waived gate.
- The broad runtime tier exceeded its earlier 600-second budget and was
  terminated. Targeted local integration checks passed; the full runtime tier
  and Linux SDKBench evaluation remain merge/release qualification work.

Target legalization's per-instruction decomposition records cannot truthfully
describe a many-to-many block rewrite. Connecting the private CZ pass to that
workflow requires a reviewed, versioned lineage/equivalence contract first.
Removing routing annotations or attributing a whole block to one source gate
would corrupt mapping and source evidence and is not an admissible shortcut.

## Subsequent prerequisite and negative-test verification

Integration PR #1222 passed its complete required CI after retrying an artifact
download timeout and a cancelled wheel job, and merged as
`4ba10f67d87fd93d19d86371ff8acdcbabcc600a`. This implementation branch has
synchronized that main revision. The earlier Linux evidence-origin failure above
is historical; its resolution does not certify this compiler branch or waive its
remaining checks.

Review found that the instruction-growth negative test proposed non-native X
gates, so target-basis rejection could conceal a missing growth check. The test
now starts from the actual two-CZ reduction, appends native zero-angle RZ gates,
and independently verifies complete-operator equivalence before injecting that
over-budget candidate. It requires rejection before residual work. All 36 CZ
tests pass. A process-local mutation removing only growth admission makes the
new regression test fail at the forbidden residual check; no production source
was modified by this diagnostic. Black, Ruff and whitespace checks pass for the
test change. This strengthens development resource-admission coverage and makes
no process-isolated SDKBench performance claim.

## Numerical convergence regression

A development diagnostic with four-wire interleaved blocks found a supported
numeric circuit that exhausted all 100 Weyl diagonalization attempts. The Jacobi
solver compared its sum of squared off-diagonal entries with `1e-16`, allowing
individual entries near `1e-8` to remain, while M2 reconstruction required
`1e-13`. The stopping condition now uses `(_M2_ATOL * 0.01) ** 2`; the existing
reconstruction threshold, 60-sweep bound, and 100-attempt bound are unchanged.
No exception is caught or converted into a successful optimization.

Three independent symmetric-unitary reconstruction cases exercise the precision
boundary. The `1e-9` and `1e-11` cases and both operand orders of a reduced
ten-instruction circuit reproduce the failure before the fix and pass afterward.
The focused Compiler, native legalization, artifact,
deployment-evidence and local integration selection passed 2,814 cases. The
development diagnostic passed all 80 circuits from seeds 71000 through 71079,
with up to 410 source instructions, maximum full-operator residual
`5.120605827706243e-14` and maximum global-phase modulus error
`2.1094237467877974e-15`. These are unisolated correctness checks, not comparative
performance evidence. The separate Qiskit count-anchor test refused the local
Qiskit 2.6.2 installation because this repository certifies 2.0/2.5 lanes; its
failure is retained rather than changing the lane or expected counts.

The convergence repair alone did not establish accuracy for every
near-degenerate unitary. A separate direct synthesis probe for CRX with angle
`1e-5` still had a phase-adjusted maximum entry residual of approximately
`3.30e-6`. The CZ block pass rejected that proposal through its unchanged
`1e-11` residual check and retained the original block. An end-to-end regression
checks the complete operator without requiring the numerical limitation to
persist.

The second diagnosis localized this error to single-qubit Euler phase
extraction, rather than the Weyl coordinates: the reconstructed KAK operator
for that source was accurate to approximately `2.88e-16`, but independently
rounded, nearly zero diagonal entries of a local factor had unreliable relative
phases. Euler extraction now obtains the diagonal phase relation from the
determinant instead of the bottom-right entry's phase. This is algebraically
equivalent on a unitary and avoids amplifying absolute roundoff in a small
entry into a rotation of a large entry. Existing diagonal and anti-diagonal
branches, target opcodes and acceptance thresholds are unchanged. Regression
cases cover perturbed near-anti-diagonal single-qubit matrices and complete CRX
synthesis with both RX and SX pulses.

After those two repairs, floating-point fidelity ties remained a separate issue:
a CRX angle of `1e-9` could still select zero entanglers and produce a residual
near `2.50e-10`. The block pass rejected that proposal outside its `1e-11`
threshold. The subsequent coordinate-based selection below addresses this
specific failure mode; these tests do not certify the direct synthesis helper
to that stricter threshold for all unitary matrices.

With both numerical repairs present, the focused selection passed **2,827
cases**, and the three direct synthesis/CZ test files passed **125 cases**.
All six near-anti-diagonal regressions fail when only the previous Euler phase
expression is restored in a process-local mutation; no production file was
changed for that check. The same 80-circuit development diagnostic passed again,
with maximum full-operator residual `4.365542661283488e-14`.

Broad qualification remains incomplete. The smoke/unit attempt for the Jacobi
repair reached its explicit 600-second limit at 58% without a final summary.
That invocation incorrectly inherited a checkout-specific `GIT_WORK_TREE`,
which also broke temporary Git repositories created by evidence tests. Removing
that environment override in a focused evidence-revision run yielded 57 passed
and one failure: the local clone's available references disagree with 98
checked-in origin declarations. Neither the environment-induced setup errors
nor the remaining origin failure is counted as passing verification. The broad
runtime attempt also reached its 600-second bound, at 17%, and its process group
was terminated. Required Linux CI remains a separate qualification requirement.
These incomplete broad attempts do not certify the subsequent Euler repair.

## Exact entangler-class selection

Ranking `(4 + |trace|**2) / 20` in floating point can round a nonzero interaction's
fidelity to one. Selecting the first maximum then omits interactions rather than
merely choosing a different exact presentation. Independent operator checks
reproduced this at all three shorter-form boundaries: a local unitary, a
one-entangler unitary, and a two-entangler unitary with a small third Weyl
coordinate. The problem also occurs with arbitrary local gates before and
after the interaction, so it is not specific to a gate spelling.

The chooser now tests representability directly in Weyl coordinates. Local
coordinates select zero entanglers; equality with the basis coordinates selects
one; a zero third coordinate selects two; otherwise the existing three-entangler
closed form is used. The absolute zero threshold is `1e-13` radians. Coordinates
below that numerical resolution are not claimed to be mathematically zero.
The unused fidelity-ranking method was removed. Qiskit's closed-form factors,
their attribution, target choices and iteration bounds remain in place. The
block-level `1e-11` operator gate still decides whether a proposed rewrite is
admissible. Preserving a small interaction may cost more entanglers than the
previous rounded-fidelity result; this is a correctness tradeoff, not a gate
count improvement claim.

The updated direct synthesis file passed **63 cases**, including twelve
operator-and-cost cases spanning the four classes with and without surrounding
local gates. A process-local restoration of the previous chooser fails eight
of those twelve cases. The broader focused selection passed **2,835 cases**
before adding the six surrounding-local-gate variants; those variants then
passed in the direct file on the same production code. The 80-circuit
development diagnostic also passed again. Black, Ruff, architecture boundaries
and the protected public API baseline pass. These focused checks do not replace
the incomplete broad qualification above.

The earlier Qiskit anchor restriction was addressed with a separate certified
Qiskit **2.0.3** test import and a complete SymEngine **0.13.0** wheel installed
only in a temporary target. The cached SymEngine directory was incomplete;
attempting to use it produced import/setup errors and is not counted as
verification. With the complete dependency, all **15** tests in
`tests/benchmark_contract/test_compiler_two_qubit_synthesis.py` passed, including
the six-basis, eighteen-opcode count anchor, with no skips. Both sides use the
recorded unitary inputs; Qiskit uses `pulse_optimize=False`. The anchor checks
entangler counts, not compilation speed or identical Euler instructions. Its
certified-lane requirement and expected counts were not changed. This local
anchor is separate from the frozen SDKBench comparison cohort and does not
promote a participant-neutral benchmark result.

## Application checks on 2026-11-19

Six development checks at `f838428114bb183a9f57ada84a844473a123a98d`
compile the existing four-wire QFT, five-wire four-layer MaxCut, and six-wire
five-layer Ising examples. Each is checked without topology and on a line,
using the expert compiler default, recursive RZ/RX/CZ basis translation, and
the private opt-in block transformation. The complete source and output
operators are reconstructed by the independent Simulation-matrix embedding
oracle used in the block tests. These examples were selected during development;
they are not a preregistered held-out evaluation corpus.

| Application | Topology | Baseline CZ | Output CZ |
| --- | --- | ---: | ---: |
| QFT | Absent | 18 | 18 |
| QFT | Line | 42 | 42 |
| MaxCut | Absent | 48 | 48 |
| MaxCut | Line | 102 | 102 |
| Ising | Absent | 60 | 60 |
| Ising | Line | 150 | 150 |

All six complete-operator checks pass at `1e-10` absolute tolerance up to one
operator-wide unit-modulus phase. The largest observed entry residual is
`4.822014874533835e-15`; the largest phase-modulus residual is
`7.66053886991358e-15`. Line outputs satisfy adjacency and retain the existing
identity-final-layout evidence. No CZ reduction is observed in any case.
The current pass conservatively refuses annotated routing operations; that
boundary remains in place until the approved many-to-many lineage stage can
validate and retain their mapping evidence. Annotation refusal explains why
those operations cannot participate, but does not prove how much a future
stage will save. The unrestricted cases also show no reduction, so routing
annotations do not explain every unchanged count.

The new application regression path preserves full operators, target basis,
CZ non-increase, routing metadata, and line adjacency without pinning these
counts as optimal or preventing later improvements:

```bash
python -m pytest tests/integration/test_compiler_application_cz.py -q
```

This is local macOS CPU correctness evidence with one numerical thread per
process. Compilation time and memory are not controlled comparison results.
It does not qualify the automatic target-artifact stage or replace SDKBench's
Linux resource, independent acceptance, revision-registration, and reporting
gates. The old benchmark adapters and frozen outcomes remain unchanged.

## Complete tier qualification status on 2026-11-19

The complete smoke/unit selection at `f838428114bb183a9f57ada84a844473a123a98d`
ran as four official disjoint shards with localhost process-group permission
and one numerical thread per process. The raw outcome is **20,423 passed,
3 failed**, with per-shard skips of 250, 242, 229, and 230; module-level
collection skips can recur across shards. Two failures came from vocabulary
census operations selecting another checkout through shared Git configuration.
A correction confined to this worktree's Git configuration restored the intended
checkout; the 173 vocabulary/census cases then passed. The evidence-origin test
still reports 98 discrepancies against locally obtainable revisions and origin
declarations. No origin declaration, historical artifact, or acceptance gate
was changed to waive those discrepancies. The complete raw run remains failed.

The complete local integration selection was attempted as four official shards,
selecting 3,026 / 3,029 / 3,036 / 3,021 tests. Each shard reached its explicit
1,200-second bound and was terminated with return code 124. They were progressing
through the synthesis application catalog at approximately 18–19%; none reached
a final pytest summary. This is incomplete qualification, not a passing runtime
tier or evidence of a diagnosed compiler regression. The runner waited for each
terminated child. Focused checks and the six application scenarios remain
separate evidence. Required Linux CI and complete runtime qualification are
still outstanding.
