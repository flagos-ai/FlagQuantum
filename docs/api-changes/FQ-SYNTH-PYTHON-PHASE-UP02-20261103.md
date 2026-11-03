# Native program phase admission for UP-02

The user requested on 2026-11-03 that UP-02 add finite real coefficients,
multi-bit phase, constant global phase, a public export and acceptance tests
after the initial phase capture PR passes CI and merges. This authorizes a
single addition to the unreleased synthesis namespace. Release authorization
and human trials remain separate.

`flagquantum.synth.phase(expression, coefficient, /)` multiplies the amplitude
for an unsigned basis expression by `exp(i * coefficient * expression)`.
A finite signed real literal or static numeric binding may be used for a
constant expression or coefficient. Quantum expressions use the existing
unsigned arithmetic subset. The scaled angles must fit finite binary64.
Captured registers must be initialized, and scratch is restored after the
phase operation. Calling the marker directly in Python raises TypeError.

The change preserves the preceding synthesis contracts, existing Qmod
predicate phase and scalar phase entry points, and Stable Core exports.
