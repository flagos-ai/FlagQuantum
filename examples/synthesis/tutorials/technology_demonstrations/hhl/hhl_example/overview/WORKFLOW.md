# Complete HHL precision workflow acceptance

Run `main.py` as a module from `README.md` or execute all Notebook cells. Preserve
the original rounded 8 by 8 matrix, eight-element RHS and all seven precisions
2 through 8. Explicitly normalize the rounded RHS before unitary preparation;
its original norm is 0.9999999994356988. The classical solve keeps the raw RHS.
Squared overlap compares normalized solutions and is scalar-invariant.

Every precision requires complete independent complex states before and after
operator-exact CX/U3 conversion. Do not align away global phases. Check the
compiler workspace and report retained estimator leakage separately. Read the
actual phase-zero, indicator-one branch, retain the source 1e-12 filter and
precision-dependent scale, then compute the source squared overlap. Costs must
be measured from the actual converted program, not copied from a prior source.

The source optional Qiskit cells are fully commented. Preserve their arrays as
historical observations and do not claim new SDK executions. Inspect the actual
final figure with all seven points, labeled measured/historical series and width
annotations. No controlled timing, speedup or optimality comparison is claimed.

Literal fractional roots preserve the current annotation and ownership contracts.
No compiler/public API/dependency changes or original quantum-program edits are
part of this workflow migration. Source pins, fresh installed Jupyter, focused
regressions, quality checks, whole corpus and frozen default evidence are all
required before crediting complete Notebook coverage.
