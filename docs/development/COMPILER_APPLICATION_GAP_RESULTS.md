# Prospective application quality gaps

The integrated compiler passes every acceptance in this local diagnostic, but
it does not dominate Qiskit or Cirq. Its ordinary restored workflow still loses
substantially on some application inputs. Its target-artifact workflow removes
many of those costs, while retaining real routing and depth gaps. These are
workflow-specific quality observations, not a global SDK ranking, resource
qualification or a claim that all larger operators are equivalent.

## Frozen protocol and complete accounting

The [protocol](COMPILER_APPLICATION_GAPS.md) and runner were published at
`db244bf28f86810ee3439054c6243df0202f86a4` before the single nonce draw.
The [freeze manifest](evidence/compiler-application-gaps-freeze.json) retains
that commit, exact configuration/version/hash records and nonce. Runtime code
equals integrated main `2ea5b78b6449b837db151e0d29c90895f6b51a4e`; clean
SDKBench is pinned at `9aa9260920984e2ca05bcb293abb885fccd9c144`.
The study changes no SDK implementation, participant setting, task recipe,
acceptance rule or resource cap. It does not measure the later unmerged PMH
experiment or claim an outcome for the later dependency-sharing change.

All 60 inputs, five distinct declared compiler seeds and four workflows completed:
1,200 accepted observations, zero exceptions and zero unsupported outcomes.
Maximum semantic residual is `9.912746458906366e-14`, below `1e-8`.
Four hundred observations compare complete six-wire operators up to one global
phase; 800 observations check three probes at widths eight/twelve. The latter
are not universal operator-equivalence evidence. Every delivered layout is
identity and every interaction satisfies its declared coupling graph.

Fifty-one source/target pairs differ from the inspected `01`-repeated nonce;
nine QFT line/ring/grid pairs repeat their operations and target edges despite
new probes. The 51 count means changed relative to that specific reference,
not proven disjointness from every historical development draw. All drawn inputs
are now inspected development cases and cannot be reused as unseen qualification.

The [raw observations](evidence/compiler-application-gaps.json) retain every
row and all 298 distinct normalized payloads as canonical JSON strings keyed
by SHA-256. The index refers to ten separately hashed payload files, each below
the repository's unchanged 1.8 MB per-file ceiling. Joining them reconstructs the
original 13,940,983-byte observation exactly, including its recorded SHA-256.
The independent archive replay checks every file digest, rejects duplicate
payload keys, reconstructs those exact bytes, and replays all 1,200 observations.
A row's original compile-wall value reconstructs the complete acceptance payload. An independently started process imported no FlagQuantum,
Qiskit or Cirq and replayed all 1,200 rows through the unchanged evaluator,
checking fixture/payload hashes, complete combination accounting, layouts and
exact metric agreement. Its [proof and script](evidence/compiler-application-gaps-replay.json)
retain the original observation digest and complete replay scope.

## Quality comparisons without independent-seed inflation

The [derived analysis](evidence/compiler-application-gaps-analysis.json)
contains every paired result, changed/reference-control groups, source script,
all positive Qiskit gaps and the prescribed next witness. Compare CZ count and
two-body dependency depth jointly. A workflow dominates only when neither
metric is worse and at least one is strictly better. A tradeoff means one
metric improves while the other worsens; no weighted score resolves it.

The following counts use 60 source/target inputs. A category is assigned only
when all five seed comparisons have that category; otherwise the input is
mixed. These are not 300 independent experiments.

| FlagQuantum workflow | Compared workflow | Dominates across all seeds | Equal across all seeds | Dominated across all seeds | Tradeoff across all seeds | Mixed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary restored | Qiskit restored | 20 | 1 | 26 | 2 | 11 |
| Ordinary restored | Cirq restored | 24 | 0 | 34 | 2 | 0 |
| Target artifact | Qiskit restored | 29 | 8 | 4 | 2 | 17 |
| Target artifact | Cirq restored | 33 | 12 | 12 | 3 | 0 |

FlagQuantum and Cirq each produced one distinct normalized output per input
across the five requested seeds, consistent with their reported lack of native
compiler-seed support in these configurations. Qiskit produced one distinct
output in 26 inputs, two in 13, three in six, four in three and five in 12.
Retain those variations rather than choosing a favorable seed.

## Prespecified next witness: twelve-wire QFT on a sparse graph

The largest positive FlagQuantum CZ excess against Qiskit on a changed
source/target pair is the ordinary restored workflow at seed 37: 414 CZs and
depth 193 versus 322 CZs and depth 173. This follows the selection rule frozen
before the draw. QFT source operations repeat; its sparse target graph differs
from the inspected reference. This selected counterexample is not a population
estimate.

| Workflow | Seed 11 CZ/depth | Seed 23 | Seed 37 | Seed 51 | Seed 73 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FlagQuantum ordinary restored | 414 / 193 | 414 / 193 | 414 / 193 | 414 / 193 | 414 / 193 |
| FlagQuantum target artifact | 328 / 156 | 328 / 156 | 328 / 156 | 328 / 156 | 328 / 156 |
| Qiskit restored | 340 / 154 | 350 / 170 | 322 / 173 | 352 / 174 | 378 / 196 |
| Cirq restored | 328 / 132 | 328 / 132 | 328 / 132 | 328 / 132 | 328 / 132 |

The target-artifact audit reports 396 CZs before native block optimization,
34 accepted rewrites and 328 delivered CZs: 68 fewer at that stage. Its earlier
396 already differs from the ordinary workflow's delivered 414. Do not attribute
all 86 CZs of the between-workflow difference to block synthesis; the workflows
also differ before that stage. The native result still exceeds the lowest
Qiskit count by six, while its depth is 17 lower. Cirq matches its count and
has 24 lower two-body depth. Those are distinct tradeoffs and gaps.

Source inspection confirms ordinary compilation currently defaults to
optimization level **2**, with optimization before and after routing. Its
adapter then performs native legalization; it does not call the target-artifact
block stage. Target-artifact compilation routes, legalizes, verifies native
rewrites and serializes an executable artifact. The difference is not explained
by an assumed FlagQuantum level-1 default. Neither pipeline may silently be
replaced by the other in the declared study.

## Remaining target-workflow counterexample

The largest positive target-artifact CZ excess against Qiskit among changed
inputs is six-wire repetition on a sparse graph at seed 11: 60 CZs/depth 57
versus 44/depth 41. This input has complete-operator acceptance. The other
Qiskit seeds yield 56/52, 58/56, 62/59 and 60/55; Cirq yields 56/52 throughout.
FlagQuantum's ordinary and target-artifact outputs both yield 60/57. The native
audit records 60 baseline CZs and zero accepted block rewrites. This is a
concrete unsolved case after native synthesis, not evidence that enabling more
local block optimization alone necessarily fixes it.

## Next implementation requirements and limits

First replay the prescribed QFT witness's source optimization, route,
legalization and local-block boundaries to separate pre-stage cost from native
savings. Preserve the independent acceptance and both original workflows.
In parallel in scope, retain the six-wire repetition witness as a small
complete-operator routing/synthesis regression. A production change must improve
a general compiler mechanism and pass the complete fixed regression set and a
new declared prospective check; special-casing either fixture is forbidden.
Do not enable the prior resource-rejected native-cost selector or PMH experiment
on the strength of this quality-only study.

Any unification of public entry points, defaults, result schemas or provenance
must follow the protected API proposal and approval process. This record grants
no additional authorization. Compiler-owned IR transforms must remain separate
from artifact construction and runtime execution.

Calls used one numerical thread and the original adapter deadlines/acceptance
caps within a 900-second supervised process group. The complete collection
finished within that budget. The shared macOS host has no formal cgroup peak
memory or exclusive-core qualification. Compile-wall values are retained only
as acceptance inputs; no timing ratios, process-memory rankings, hardware,
scalability or publication eligibility are claimed. Formal host-qualified
SDKBench remains user-deferred. The existing runnable public workflow and its
expected restored-layout result are in the frozen protocol.

## Follow-up stage diagnosis and rejected widening

The [stage replay and operand-order experiment](COMPILER_OPERAND_ORDER_GAPS.md)
separates the six-extra-SWAP source-optimization effect from native synthesis
savings. It also retains a rejected candidate: all 120 inspected application
outputs pass semantics, but three ordinary outputs regress and two make
count/depth tradeoffs. No candidate optimizer code is enabled in production.

## Integration baseline and CI source preparation

The documentation branch is synchronized with integrated main
`5a84f706ab9aeaab0ad3a5472bac292616fe39fe`. Its production `flagquantum/` tree
equals that main tree; this synchronization does not rerun or relabel the frozen
study measured at `2ea5b78b6449b837db151e0d29c90895f6b51a4e`. Recorded source
pins, inputs, numerical outputs and replay digests remain unchanged.

CI for documentation head `d8106c17ee186b3b7df12982d454708c8bc13145` failed
before tests: the API source-restoration step exceeded its existing ten-minute
budget, and the fallback checkout recorded TLS connection termination before
the preparation job was cancelled. One unchanged failed-job retry also failed
source preparation. Required aggregate checks consequently failed; they are
not test passes. Synchronization supplies the current integrated source baseline
for a fresh exact-head run without modifying CI, budgets, acceptance tests or
recorded evidence. Full CI remains required before merging this documentation.
