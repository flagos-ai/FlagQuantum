# Surface-code error scheme complete-source audit

The entire `surface_code/qec_scheme.py` reference is frozen at Qualtran v0.7.0
revision `bcb17bb25360ed68ce3bc084645db3e929799389`. Its source blob is
`00e2bd5acfc55309cb1b340edec25ceeb4183774`, with original test blob
`43b935b1dad82afa477556b536425e3771d1c8b7`.

## Ownership and complete definitions

`qec/resources/error_model.py` owns both source definitions. `QECScheme` maps to
`SurfaceCodeScheme`. `LogicalErrorModel` retains its domain name and binds a
physical error to that scheme. Its scheme binding is exposed as `qec_scheme`.
The two numerical fields, logical-error formula, odd-distance budget inversion,
physical-qubit formula and both named parameter factories are implemented.
Subclass construction and frozen fields retain the source behavior.

The scheme is an analytical physical-resource assumption. It takes scalar
physical probabilities, distances and budgets. It owns no quantum registers,
measurement outcomes, workspace or primitive emitters. Algorithm operations
continue to use their existing `circuit.append(op)` entry points. Primitive
counts and logical resource models remain independent inputs to the physical
estimates.

## Direct consumer replacement

The shared implementation replaces the prior local formulas in four production
modules. Gidney-Fowler factory estimates bind the distance callable. Physical
cost aggregation uses the shared logical error rate. The 15-to-1 density-matrix
schedule uses the same callable at its X/Z/measurement distances. Beverland
budget inversion and tile footprints use the shared scheme methods.

A replacement test substitutes the untouched reference scalar formulas at this
boundary and runs the same four consumer scenarios. Their complete estimates
remain equal, including factory output error, data error, qubits, cycles and
budget-derived distance. Existing consumer tests also compare native
five-qubit density matrices and source physical-model examples.

## Numerical acceptance and exceptions

All source formulas match bit-for-bit across both named parameter scenarios and
an independently chosen prefactor/threshold. Logical rates and bound callables
cover 32 distances and six physical error rates per scenario. Budget inversion
covers three physical rates and 39 budget exponents per scenario. Zero,
negative, equal-threshold and infinite inputs retain source arithmetic exception
categories. Native consumer input validation continues at each consumer's
existing boundary.

## Original rounding defect

For the Gidney-Fowler scheme, physical error `1e-4` and budget `1e-7`, the frozen
logarithmic inversion selects distance 5. Its calculated rate is
`1.0000000000000002e-7`, which exceeds that exact floating-point budget. The native
scheme preserves this result exactly. A dedicated test records the original
violation. The reference is unchanged, and no tolerance is used to classify
that budget as satisfied.

Q-0923 closes as a complete source migration with every definition and direct
consumer replacement verified. On 2026-11-26 the user explicitly accepted
Q-0924's complete original-test responsibility under the
[approved disposition](G2_UPSTREAM_DEFECT_DISPOSITION.md). Its four unchanged
tests and recorded consumer evidence are reused. The floating-budget
counterexample remains in regression; a stricter globally conservative policy
requires a separately specified numerical change.

## Evidence

The batch passes 77 focused and physical-consumer cases. All six changed Python
files pass strict mypy, Black and Ruff. Core-team scope, architecture, IR and API
checks pass. The four untouched original tests pass. Logs are stored under
`g2-surface-code-scheme-*` in the reference evidence directory. Prior algorithm
and package regression evidence is reused at its verified publication SHA.
