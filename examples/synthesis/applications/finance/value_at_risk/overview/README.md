# Native fresh value at risk

```bash
python -m examples.synthesis.applications.finance.value_at_risk.overview.main
```

The complete Notebook retains the original seven asset bits,128point
log-normal grid,mu0.7,sigma0.13,target probability0.07,search tolerance0.007,
initial threshold32 and width budget28. Its distribution is freshly computed
from the closed-form density; no retained probability table supplies model data.

Native `qfunc` prepares the distribution and XORs the exclusive predicate
`asset < threshold` into an indicator. Reversible composition supplies each
structural Grover power. Full physical states are checked against an independent eight-logical-qubit
public Circuit and a direct-vector reference tensored with clean zero work,
with absolute phase, workspace-zero and named indicator checks. The dropped asset is retained private environment,
not clean compiler workspace.

The existing local count-only adaptive estimator uses epsilon0.05,alpha0.01,
at most64steps,powers0..16 and256fresh samples per batch. Each requested
structural threshold/power compiles once; actual local marginals supply fresh
seeded binomial draws. Only counts enter estimation. Exact CDFs are used in
classical comparison and independent validation, not quantum search decisions.
This is a documented bounded local profile, not proprietary Classiq IQAE,
runtime parameter binding, hardware timing or a speedup claim.

The source bisection stops when either tolerance is reached or its integer
search size is exhausted. Its search tolerance is tighter than its amplitude
estimation precision, so quantum convergence is not guaranteed and is reported
explicitly. No exact CDF correction or tighter estimator silently repairs it.
The source function prints an index but implicitly returns `None`; the native
diagnostic exposes its internal index and records `original_return=None`.
The source's first printed `v` uses `probs[index]`, unlike later grid values;
that observation is retained. Its plot legend incorrectly says5% although the
calculation uses7%; the native plot labels the actual7% calculation.

The power cap0..16 is an algorithmic bound, not a promise that every threshold
and power fits the physical construction budget. Each program retains the
default100,000operation limit and source width28, permitting up to20clean
workspace qubits beyond the eight logical qubits. Oversized combinations fail
closed before execution or sampling. Direct native predicate XOR reuses existing
compiler lowering; the threshold1,power2 probe uses13qubits including5clean
workspace qubits and48,269operations under these unchanged source budgets.
The earlier zero-workspace H/controlled-phase construction exceeded the default
operation budget on that case. Historical probe and refusal evidence are retained;
tests do not require that future optimized implementations keep refusing it.
