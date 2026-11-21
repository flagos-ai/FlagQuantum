# Component report notebook source closure

## Complete source and ownership

Q-0040 covers the complete dev_tools/bloq-report-card.ipynb notebook at immutable
Qualtran v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389. The original Git
blob and cell-type sequence are verified. The native authored notebook is
examples/notebooks/component_report.ipynb. Existing component_report owns table
construction, styling and summaries. Existing Testing example_checks owns all
five checks. Existing WordSwap owns the example Program and its decomposition.
Existing register binding owns physical lowering. No new checker, registry or
component hierarchy is introduced.

Run all notebook cells in the locked development environment. The example uses
a two-bit WordSwap Program with its existing implemented decomposition. The
report checks construction, physical decomposition, instruction counts,
serialization and register typing. Native report labels use the actual package owners below flagquantum. The
real example therefore appears under core.ir.program, where ProgramRecord is
defined. Empty compiler.registers, algorithms.chemistry and algorithms.primitives
sections remain visible. This is an explicit native example report, not a claim
that every upstream example now has a native default checker.

## Complete views and source counterexamples

Every original code cell runs unchanged with only the report-module import
replaced by its already accepted native owner and a shared input table. The
native notebook runs over the corresponding native-domain table. All six check
statuses are represented in this comparison. All five styled views, all four per-owner summaries and the full summary
agree after the explicit package-name mapping, including both original query
quirks.

Two original query quirks are preserved and retained as negative evidence. The
factoring styled view uses equality while its summary includes child packages.
The Other query tests an unstripped chemistry prefix against already stripped
package names; it also includes factoring child packages. Native owner-name
substitutions retain these original equality/prefix predicates and the same
concrete overlap. Tests execute the original queries and verify both cases in
the unchanged source and native notebook. This is source-behavior parity, with
those query limits recorded for a future separately scoped change. It does not
claim a disjoint domain partition or silently repair the reference.

## Actual execution and resource evidence

The real native notebook executes in a Jupyter kernel. All five checks pass
for the owned WordSwap example. A separate direct execution checks the returned
table and full summary. Deliberately changing its declaration to five CX gates
causes the existing count checker to fail, demonstrating that the report uses
the actual lowering and independently supplied declaration.

| Quantity | Observed value | Evidence owner |
| --- | --- | --- |
| Measured emitted primitives | 6 CX | Actual register binding |
| Declared primitive count | 6 CX | Two bits times three CX per swap |
| Abstract resource model estimate | Not introduced | Existing model owners remain separate |
| Allocated workspace | None | Bound Program has no workspace events |
| Measurement branches | No measurements | Existing swap decomposition |

The notebook prints measured and declared primitive counts separately. The
ordinary circuit.append(op) path is exercised with the same WordSwap owner on a
complete five-qubit complex input, including a spectator. The output equals the
independent word-axis permutation exactly, and six emitted CX gates are verified.
Accepted original WordSwap decomposition evidence is reused. This batch does
not claim a new universal upstream-library execution census. The development
notebook remains outside the scientific publication catalog.

Broad default-suite NumPy dependency ownership and preceding-execution copyright
interaction remain open. Pending Proposals 116, 117 and 118 remain separate
protected-contract work.

## Batch verification

The final source-faithful concentrated regression passed 324 tests with 913
visible warnings and zero skips. Strict mypy passed the test harness and the
complete extracted native notebook code. Black and Ruff passed that same Python
scope. Integration scope, architecture, IR, API and whitespace checks passed.
The earlier disjoint-group draft was replaced before commit so that the final
native notebook preserves the unchanged source query behavior. Full default
suite failures remain separate open evidence.
