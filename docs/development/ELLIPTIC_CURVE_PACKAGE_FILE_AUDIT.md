# Elliptic-curve package source disposition

The complete Qualtran v0.7.0 `bloqs/cryptography/ecc/__init__.py` has frozen
Git blob `537b429e07bb865f6b51f659f766a5be75018ef5`. It contains the Apache-2.0
notice, a mathematical introduction and five relative import statements exposing
six class aliases. There are no constructors, numerical kernels, decomposition
methods, resource formulas, side effects or executable examples in this file.
The audit checks the full AST and actual imported alias identities.

The introduction defines affine curve addition, scalar multiplication and the
public-key relation Q = [k] P. The existing point owner and its complete small
curve tests cover that algebra. The double-phase owner composes the two quantum
phase estimations used in the discrete-log construction. Native ownership follows
FlagQuantum's algorithm primitives. Each source alias has one existing owner.

| Source alias | Existing native owner | Existing acceptance evidence |
| --- | --- | --- |
| ECPoint | `algorithms/primitives/elliptic_curve_point.py`, ECPoint | Every point pair on two small curves, nonnegative multiples and symbolic coordinates; source negative-scalar defect recorded separately |
| ECAdd | `algorithms/primitives/elliptic_curve_addition.py`, EllipticCurveAddition | All six arithmetic stages, source signatures/calls, operands and feedback, coherent channels and release witnesses |
| ECAddR | `algorithms/primitives/elliptic_curve_fixed_point.py`, EllipticCurveFixedPointAddition | Scalar control signature and original classical callback; original physical decomposition is unavailable |
| ECWindowAddR | `algorithms/primitives/elliptic_curve_window_addition.py`, EllipticCurveWindowAddition | Original tables and calls, concrete lookup gates, small coherent execution and bank erasure |
| ECPhaseEstimateR | `algorithms/primitives/elliptic_curve_phase_estimation.py`, EllipticCurvePhaseEstimation | Symbolic calls and concrete windowed operands, every tested terminal projection and clean work |
| FindECCPrivateKey | `algorithms/primitives/elliptic_curve_double_phase.py`, EllipticCurveDoublePhase | Original initialization and dirty exits, two shared-coordinate phase stages and all tested joint readouts |

The source package's flat alias surface is disposed into these existing direct
module imports. Its initialization adds zero physical primitives, workspace or
measurement feedback. Resource estimates remain owned by the referenced child
modules. The package has no independent physical action to compare. Existing
`DynamicCircuit.append(op.make_on(...))` examples and coherent execution tests
are reused directly. No additional algorithm or public alias is needed for
package initialization.

Q-0455 covers this complete initializer disposition. Q-0457, Q-0458, Q-0460,
Q-0461 and Q-0466 retain their existing open algorithm boundaries. In particular,
source arithmetic cleanup defects, unavailable fixed-point gate decomposition,
symbolic lookup execution and inherited provider breadth are independent gaps.
Package alias acceptance does not close any of those algorithm parents.

Validation is recorded in the shared reference environment under
`g2-ecc-package-*-20261121.log`. The new package audit is run together with the
existing point, window-addition, six-stage addition, phase-estimation and
joint-phase suites. Strict typing, formatting, Algorithms scope, architecture,
IR and protected API checks cover this batch. The frozen initializer hash is
asserted at runtime before its AST is used as evidence.

All 38 concentrated cases pass without skips. Strict typing for the new audit,
Black, Ruff, Algorithms scope, architecture, IR and API checks pass. The initial
strict check identified a missing local list annotation. The accepted file adds
that annotation, with the earlier diagnostic retained in the execution history.
