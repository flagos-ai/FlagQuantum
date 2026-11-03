# Native scoped cleanup and checked adjoint admission for UP-03

The user requested public within/apply exports, checked inverse composition
and acceptance tests after #1042 passes and merges. This admits three context
markers in the unreleased `flagquantum.synth` namespace. Release authorization
and human trials are separate.

`with within():` prepares sized unsigned locals from arithmetic expressions.
An immediately following `with apply():` uses them for phase or static
Const-only calls, including checked adjoints. The compiler verifies unchanged dependencies and restores
and releases the locals in reverse order. The declared local is unavailable
after the pair.

`with adjoint():` contains static calls to Const-only quantum functions.
The admitted effects are phases and verified compute/use/uncompute scopes.
Call order is reversed, finite phase angles are negated, and local arithmetic
retains its cleanup contract. Calls with Output effects are rejected.
Invoking these markers outside captured qfunc source raises TypeError.
