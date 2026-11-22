# Controlled-gate commutation results

Owner: Compiler. Classification: `single_device_fast_path`.

The [implementation and protocol](COMPILER_CONTROLLED_COMMUTATION.md) were
published at `1b859d7cf1de0929b9e5ce53c515c663058d004e` before observation.
The control is production main `46ba1523ed0c47fcf304d64d33b34d3f6c4b9629`;
the only production difference is the structural commutation extension. The
experiment patches old/new checker functions for equal lifetimes. It does not
enable the separate commuting-frontier selector or native-cost experiments.

## Complete application observations

Both declared draws completed: 240 observations using the inspected nonce and
240 using one fresh nonce generated after publication and frozen before either
draw ran. Each draw covers all 60 inputs, both complete workflows and both
variants, with compiler seed 11. All 480 observations pass unchanged SDKBench
`9aa9260920984e2ca05bcb293abb885fccd9c144` independent acceptance.
There are 160 complete-operator checks at width six and 320 three-probe checks
at widths eight/twelve; the latter cannot establish universal equivalence.

| Draw | Workflow | Improved CZ/depth | Equal CZ/depth | Regressed CZ/depth | Tradeoff |
| --- | --- | ---: | ---: | ---: | ---: |
| Inspected | Ordinary restored | 0 | 60 | 0 | 0 |
| Inspected | Target artifact | 0 | 60 | 0 | 0 |
| Fresh | Ordinary restored | 0 | 60 | 0 | 0 |
| Fresh | Target artifact | 0 | 60 | 0 | 0 |

Do not interpret this as closing the existing entangler/depth gap. These
application circuits do not demonstrate the three-to-one CX cancellation of the
focused uncomputation example. Each draw does change the normalized output for
the 12 ordinary repetition cases: each has one fewer RZ instruction, with
unchanged CZ count and dependency depth. The remaining 108 paired outputs per
draw are byte-identical after normalization. Total instruction count is an
additional output diagnostic inspected after observation, not a replacement for
the declared CZ/depth objectives. The target-artifact outputs do not improve.

The source proof and real uncomputation workflow support broader structural
coverage. These application results establish bounded before/after acceptance
and disclose limited benefit; they do not establish a global SDK ranking,
significance, broad performance improvement or independently administered
qualification. All source pins, manifests, payloads and original compile-wall
acceptance values must be retained.

## Resource follow-up declaration

Declare this follow-up after the complete quality observations and before its
resource measurements. Preserve the frozen implementation and use inspected
QFT6/grid, repetition6/ring and QFT12/grid, both complete workflows and both
checker variants, compiler seed 11. Keep the existing adapter configurations,
120-second deadlines, operation budgets and independent acceptance.

Each of six case/workflow pairs uses two warmups and nine alternating retained
samples per variant, fresh adapters, equal checker-patch lifetimes and one
numerical thread. Measure adapter execution outside independent acceptance.
Measure one additional traced Python-allocation peak per variant after garbage
collection, keeping adapter creation outside tracing. Verify every retained
final output and audit against its frozen quality result and replay independent
acceptance outside timing. Keep all raw samples and failures, with no favorable
rerun or case removal. Freeze the driver and this declaration by source hashes.

Use a 240-second process-group supervisor and no simultaneous local test suite.
No resource admission threshold is introduced after measurement. Shared-host
medians and Python allocation peaks cannot certify significance, a latency
ranking, native allocations, peak RSS or formal Linux cgroup memory. Formal
host qualification remains user-deferred; no machine or service may be restarted.

## Archives and independent replay

The raw [inspected index](evidence/controlled-commutation-inspected-screen.json)
and [fresh index](evidence/controlled-commutation-fresh-screen.json) each retain
three hashed payload chunks. They reconstruct the original 3,828,280-byte and
3,886,439-byte reports exactly. Each draw retains 116 distinct normalized
outputs and every one of its 240 observation rows. The
[analysis](evidence/controlled-commutation-analysis.json) lists all output changes,
including the additional instruction-count diagnostic.

A fresh process [independently replayed](evidence/controlled-commutation-replay.json)
all 480 observations without importing FlagQuantum, Qiskit or Cirq. Maximum
observed residuals are 6.072657351995283e-14 for the inspected draw and
2.758150189067026e-14 for the fresh draw against the unchanged 1e-8 threshold.
Fixture hashes, complete case/variant coverage, output hashes, layouts and exact
returned metrics are checked. Three-probe rows retain their limited coverage.

## First resource observations and limitations

The [complete resource record](evidence/controlled-commutation-resources.json)
retains every timed sample, allocation observation, driver and protocol snapshot.
The first run completed within the supervisor budget and was not rerun.

| Input | Workflow | Baseline median ms | Candidate median ms | Candidate/baseline | Baseline Python peak bytes | Candidate Python peak bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 06/qft/grid | ordinary-restored | 7.778 | 7.713 | 0.9917 | 472558 | 471969 |
| 06/qft/grid | target-artifact | 93.818 | 91.605 | 0.9764 | 2198408 | 2198392 |
| 06/repetition/ring | ordinary-restored | 6.712 | 8.055 | 1.2000 | 552720 | 562802 |
| 06/repetition/ring | target-artifact | 121.201 | 121.713 | 1.0042 | 2566909 | 2566859 |
| 12/qft/grid | ordinary-restored | 30.475 | 30.396 | 0.9974 | 2113695 | 2115138 |
| 12/qft/grid | target-artifact | 565.080 | 562.130 | 0.9948 | 10118879 | 10118879 |

The ordinary repetition6/ring median grows by about 20.0%, despite removing only
one RZ. Do not hide this adverse observation or infer significance from nine
shared-host samples. Other ratios close to one do not establish parity or an
improvement. The raw field named `compiler_stage_python_peak_bytes` covers the
complete adapter execution window, not an isolated compiler transform. It
excludes native allocations and is not peak RSS.

The resource driver validates the last timed output and audit per variant
against quality hashes and independent acceptance. It does not retain or validate
each intermediate timed payload or the traced payload. Thus it does not fully
meet the declaration's per-sample validation intent; these observations do not
certify complete resource-protocol conformance or resource admission. Retain
this limitation with the original driver and data rather than rewrite the
protocol or rerun for a favorable median.

An [untimed diagnostic](evidence/controlled-commutation-work-census.json) examines
the adverse inspected ordinary repetition6/ring case without replacing its
measurements. Both outputs and audits still match the frozen quality hashes.
The baseline makes four commutation-analysis calls and 396 predicate calls; the
candidate makes six analysis calls and 610 predicate calls. Its initial widest
block grows from three to ten, and initial logical pair queries grow from 82 to
121. Broader blocks and additional fixed-point work are concrete costs of the
newly enabled simplification. This census does not isolate how much of the
observed wall-time difference comes from each cost or from shared-host noise.

## Decision

Keep this implementation under draft review. It proves useful structural
coverage and the default uncomputation example, while these application draws
show no entangler/depth improvement and the resource observation includes a
material regression. Do not call this a solution to the remaining application
routing gap or enable the separate commuting-frontier policy on its strength.
The next implementation must address analysis/fixed-point work and evaluate
final entangler count, dependency depth and resources together under a new
frozen declaration. Formal Linux qualification remains deferred.


## CI attribution follow-up

The first full CI run found seven historical benchmark-contract failures. The
new controlled-gate reach composes with old rotation, identity and two-qubit
passes, so their historical population counts move even though their own rules
are unchanged. Keep those archived numerical pins and raw evidence unchanged.
Replay the three historical contract modules under an explicit substitution of
the earlier controlled-gate decision boundary, following their existing
rotation-merge and fold substitution instruments. This is scoped to historical
contract replay; normal benchmark commands and current production compilation
continue to use the current checker.

A separate test enumerates all 576 ordered controlled opcode/placement pairs
under that historical scope, verifies the old boundary, and proves the patch
restores the current checker and its three-to-one CX workflow afterwards.
Complete operator tests for the current 1,728-pair census remain active outside
the historical scope. This repairs attribution without changing a recorded
count, threshold or production algorithm. A vocabulary check also found the
word `wires` in a new explanatory comment; it is reworded to `qubits`. This
comment-only correction does not change frozen implementation behavior or
relocate any observation to a different source revision.


Follow-up validation uses the existing certified Qiskit 2.5.0 lane: all 89
focused current-rule and historical attribution tests pass. The initial local
2.6.2 attempt passes 87 and refuses the two Qiskit anchors because that version
is outside the repository's certified lanes; the refusal is preserved, not
patched out. Ruff, Black and strict Linux-targeted mypy for the complete
commutation module and current-rule test module pass. The live comment/docstring
vocabulary scan of the complete commutation module matches integrated main.
The seven historical count failures are addressed by substitution, with every
recorded count and application observation unchanged. The complete exact-head
CI run remains required before integration; the resource concerns still keep
this implementation under draft review.

The sparse default tier also passes 262 tests, with 96 deselected. This is
scoped local verification, not a claim that full repository CI has passed.

## Downstream constraint and live-resource follow-up

The exact-head CI run at `9fda5216fb1afee1966665821eb8f49e0162a4fe`
also exposes live synthesis expectations that were calibrated against earlier
compiler costs. For example, the branched arithmetic retain schedule now uses
95 operations instead of the test's earlier 103. An operation limit of 103
therefore admits both schedules; the unchanged width objective correctly
selects the narrower recomputation schedule. This is distinct from the
historical benchmark-attribution failures above. Do not disable optimization,
substitute the historical checker into live workflows, or alter archived data
to make those scenarios pass.

Update the live regression scenarios with stricter limits and exact current
costs while retaining their numerical references, exceptions, named-register
checks and clean-workspace requirements:

| Scenario | Earlier live expectation or cap | Current live expectation or stricter cap |
| --- | --- | --- |
| Branched retain-only operation limit | 103 | 95 |
| Combined recovery operation/depth limits | 102 / 72 | 94 / 68 |
| Uniform-input retain-only operation limit | 105 | 97 |
| Qmod predicate depth, 13 / 14 qubits | 69 / 53 | 67 / 49 |
| Water evolution native CX count | 4,176 | 3,694 |
| Exponential operation rejection limit | 2,398 | 2,386 |
| Exponential depth rejection limit | 2,045 | 2,038 |

The exponential test also checks positive acceptance at exactly 2,387
operations and depth 2,039, alongside rejection below those current boundaries.
The arithmetic recovery test still requires every candidate to report all four
violated limits and checks every combined recovery with a complete retry. The
preparation test still verifies all 100 cancelling input X operations consume
construction budget and checks the unchanged 205-operation source separately
from the 95-operation executable. These are live private-compiler cost
expectations, not changes to an API snapshot or a historical numerical artifact.
No SDKBench threshold, case, tolerance, deadline or resource limit changes.

The protected `tools/` selection walkthrough is repaired separately in
[Integration PR #1308](https://github.com/flagos-ai/FlagQuantum/pull/1308).
It explicitly minimizes depth for both requests using the existing objective;
its original qubit and operation limits and complete complex-state checks
remain unchanged. The same edited script passes on integrated main and this
controlled-gate candidate. Integrate that separate change before promoting the
complete compiler PR; the original walkthrough assertion remains a CI blocker
until its dependency lands and the compiler branch synchronizes main.

All 148 cases in the affected arithmetic, user workflow, Qmod predicate, water
and exponential-boundary selection pass locally, including complete numerical
and gradient assertions. The narrower first follow-up passed 58 cases. These
checks do not establish full CI success or performance admission.

A first strict Linux-targeted mypy run of the four initially touched test modules
reports 40 errors. Repeating the same command with all four files shadowed by
their exact pre-follow-up Git contents reports the same 40 errors. This is an
existing typing deficit, not a passing strict-type check or an exemption from
the engineering standard. Resolve the deficit in the touched modules with
typed callbacks and fixture arguments, explicit optional-result assertions,
runtime narrowing of resource records and the property descriptor, and an
explicit dynamic boundary for deliberately invalid immutable-mapping writes.
No ignores, casts or checker exclusions were added. Strict Linux-targeted mypy
then passes all five complete touched test modules, including the additional
exponential boundary scenarios. The resource regression
and the incomplete resource-protocol evidence remain unresolved, so this PR
continues as a draft.

The expanded default selection passes 311 tests with 239 deselected. A first
broader sparse-checkout run records 493 passes and three failures: the existing
selection example needs Integration PR #1308, and two other examples lack the
unchanged `selection_costs` source directory in that local sparse checkout.
Retain those failures; they are not numerical acceptance failures or a passing
whole-repository check.

After materializing the missing source, an explicit prospective combined-tree
check substitutes only the exact example file from Integration commit
`b124669d1568fdbcdf54975cbeed4a463b8515a1`. With the compiler follow-up tests
unchanged, `python -m pytest -m "smoke or unit or integration" -q` passes all
496 locally selected cases, with 54 deselected. This includes both independently
executed candidate schedules, complete state/phase/gradient checks and every
exponential rejection/acceptance scenario. The example substitution is then
removed byte-for-byte; it is not committed into the Compiler PR. This proves
the proposed combination locally, not that the dependency has landed or the
Compiler PR's exact-head CI is green. No production compiler or frozen evidence
file changes in this follow-up. Ruff and Black pass for all five complete test
modules, strict mypy passes all five, and the six edited artifact paths pass
the English-language check.


## Original thermal-reset live-cost regression

Exact-head CI at `e6dc85d30830563456c7e5735a4490100ebb848e` reports two
coverage failures. The selection example remains the already documented
Integration PR #1308 dependency. The second failure is the original 50-round
thermal-reset test, which stops at its old live operation-count assertion.
Neither failure is retried as infrastructure noise.

An untimed diagnostic synthesizes the unchanged licensed original twice,
substituting the existing pre-extension predicate boundary for the control.
The earlier boundary reproduces 22,200 operations and depth 19,901; the current
boundary produces 22,100 operations and depth 19,801. The only gate-count
difference is CX: 3,500 becomes 3,400. Both retain 400 amplitude-damping channels,
one workspace qubit and 11 total qubits. See the [complete counters and driver](evidence/controlled-commutation-thermal-cost-diagnosis.json).
This is a causal cost diagnostic, not a wall-time or channel-correctness claim.

Update only the live cost expectations to 22,100 operations and depth 19,801,
and explicitly assert 3,400 CX. Preserve the original source/license hashes,
25,000-operation construction cap, 20,000-operation refusal, density memory
refusal, all 400 channels, all named registers, complete workspace-tail checks,
and independent Kraus-channel density reference at absolute tolerance 1e-10
with zero relative slack. All seven thermal/reset integration cases then pass,
including the original 50-round complete-density comparison.

The first strict check of the unchanged thermal test reports eight existing
typing errors. The touched module is repaired using a typed external Torch
block-diagonal callable, checked filename/resource/state narrowing, explicit
synthesis keywords and concrete Program/ExecutionResult arguments. No casts,
ignores, exclusions, tolerance changes or production changes are introduced.
Performance admission and the separate selection-example dependency remain
unresolved; the compiler PR stays a draft.


The final combined local selection passes 318 tests: the existing selected
unit/compiler cases plus all seven thermal/reset integration cases. An earlier
command named an unmaterialized `tests/hybrid_compiler` directory and collected
zero tests; it is a setup failure, not verification. The corrected selection
uses existing directories and retains all numerical, phase, workspace and
refusal assertions. Strict mypy, Ruff and Black pass for the complete thermal
test module; the three added/edited artifact paths pass the English check.
