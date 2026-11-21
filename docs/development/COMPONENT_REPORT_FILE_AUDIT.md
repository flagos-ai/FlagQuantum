# Component report source closure

## Source and ownership

Q-0058 covers the complete untouched Qualtran v0.7.0
`dev_tools/qualtran_dev_tools/bloq_report_card.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The reference test verifies its
original Git blob against the immutable manifest. Google copyright and the
Apache-2.0 source attribution remain in the native files.

`tools/development/component_report.py` owns package labels, missing-example
sets and rows, ordered five-check collection, slow-check warnings, discovery,
sorting, pandas styling and status summaries. `component_cls` replaces the
source-domain column name. The caller supplies the existing domain class
predicate and checker because native components have no required shared base.

The existing register checks move unchanged from the test helper to
`flagquantum/testing/example_checks.py`. Existing tests import the same objects
through explicit re-exports. Pytest result policy and classical action assertions
remain test-owned. The report consumes a checker using these shared records. It
never selects a lowerer, performs numerical execution or imports a test package.
The native production numerical and Compiler boundaries remain authoritative.

## Complete behavior comparison

The source differential exercises all report functions and all six statuses,
missing and duplicate classes, all five check ordering, exact sorted tables,
status summaries, exact styled HTML after column/UUID normalization, implicit
slow-example exclusions and explicit overrides, slow warnings, empty-input
failure and one-shot-iterable exhaustion. Domain callback exceptions propagate.
Native register-program reports execute the same accepted five checks against
the existing three-X `circuit.append(op)` fixture. Existing reference-testing
acceptance supplies the complete complex basis columns and signed-adder
measurement/workspace evidence. Moving the checks changes no quantum stream.

The original accepts an Iterable but consumes examples twice. A generator loses
its example rows, and wholly empty inputs raise `KeyError('package')`. Both
boundaries remain explicit and preserved. Use a reusable list for normal reports.
The source default excludes the two named slow Hubbard examples. Native explicit
examples bypass these exclusions identically.

## Dependencies and remaining scope

Pandas and Jinja2 are existing optional report/documentation dependencies. Pandas
owns DataFrame/Styler behavior and is isolated to this developer tool. Neither is
added to Core or runtime execution. Existing dependency licenses apply, BSD-3-Clause
for pandas and Jinja2. The replacement interface is the report's example/check
input and tabular output. A future renderer replacement must pass the existing
table and style differential fixtures. No dependency manifest changes are needed.

Q-0937 remains open for generic source-object serialization, wire-symbol checks
and arbitrary typed graphs. This report closure does not assert that every native
component is physically lowered or that its five checks all pass. Q-0089 tensor
reporting remains a separate source file and numerical-execution boundary.
Measured primitive counts and model counts remain separate in their original
owners. This batch introduces no additional quantum resource estimate.

## Acceptance evidence

The concentrated regression passes 1446 cases with 36 original missing-model
warnings and zero skips. It includes all existing register-check consumers,
reference testing, native metadata/discovery and the complete library graph.
Strict typing passes four files. Black, Ruff, Verification/Integration scope,
architecture, internal IR, protected API and diff checks pass. Evidence logs are
`g2-component-report-{focused,regression,strict,black,ruff,scope-verification,scope-integration,architecture,ir,api}-20261121.log`
in the local locked-reference directory.
