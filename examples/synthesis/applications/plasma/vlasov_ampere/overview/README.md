# Vlasov-Ampere inverse

Run `python -m examples.synthesis.applications.plasma.vlasov_ampere.overview.main`, or execute `walkthrough.ipynb`.

All 27 original stages retain the eight spatial points, eight signed velocity values, two electric sectors, degree 251 and complete 18-bit inverse. The workflow reconstructs every column of the actual native block by execution, computes its minimum singular value, fits all 252 phases freshly and executes all 13,560 emitted instructions. The companion native Circuit workflow supplies the validated full plasma block encoding.

The CPU executor transfers the complete state through contiguous 64-instruction Circuit segments. This bounds execution-cache memory on the local 16 GiB machine. Every original operation executes in order, including all relative global phases. A separate singular-pair recurrence checks all 262,144 final amplitudes and the projected polynomial response. Resource counts use the native static matrix-operation basis.

The original degree-251 polynomial is an approximation to the inverse. Its actual linear-system error is reported alongside the independent polynomial and full-state errors. The plots retain all eight electric-field amplitudes. The original classical source includes an i phase that its quantum preparation explicitly omits. Both source conventions remain visible and the field plots compare absolute values as the source does.

The unprojected preparation extension is the declared Householder extension from the companion workflow. Full matrix reconstruction uses 128 basis-column executions to obtain the complete original 128 by 128 projected operator.
