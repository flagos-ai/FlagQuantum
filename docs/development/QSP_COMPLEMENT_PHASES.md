# Root-based QSP complements and numerical phases

Simulation owns `numerics/qsp_complement.py` and `numerics/qsp_phases.py`.
They adapt the unchanged `qsp_complementary_polynomial` and `qsp_phase_factors`
from Qualtran v0.7.0 `bloqs/qsp/generalized_qsp.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Source copyright is retained.
Phase recursion also uses the source `SU2RotationGate.rotation_matrix` expression
at lambda=global_shift=0. It does not import Compiler to perform numerical work.
Existing Compiler SU(2) decomposition remains the physical authority and is
unchanged. No new registry, dependency, public export, protected API or execution
engine is introduced.

## Local numerical chain

```python
import numpy as np
from flagquantum.simulation.numerics.qsp_sampling import (
    random_qsp_polynomial,
    check_gqsp_polynomial_pair_on_random_points_on_unit_circle,
)
from flagquantum.simulation.numerics.qsp_complement import qsp_complementary_polynomial
from flagquantum.simulation.numerics.qsp_phases import qsp_phase_factors

p = random_qsp_polynomial(3, random_state=np.random.RandomState(42))
q = qsp_complementary_polynomial(p, verify=True)
q_values = tuple(complex(c) for c in q)
check_gqsp_polynomial_pair_on_random_points_on_unit_circle(
    p, q_values, random_state=np.random.RandomState(17),
)
theta, phi, initial_lambda = qsp_phase_factors(p, q_values)
assert theta.shape == phi.shape == (3,)
assert np.isfinite(initial_lambda)
```

The normal physical workflow remains `circuit.append(op)`. Tests feed the
returned angles to the existing `su2_rotation_ir` and append its existing
`FieldOperation`, checking both signal basis inputs against original rotation
matrices and the idle clean wire. This batch does not add a physical
GeneralizedQSP emitter or claim that numerical coefficients are appendable.

## Retained source sequence

The complement builds R(z)=z**d-P(z)*conj(reverse(P))(z), computes NumPy roots in
their returned order, classifies unit/larger/smaller roots, pairs unit roots by
the original allclose rule, optionally verifies reciprocal root pairing, and
interpolates from paired-unit plus smaller roots with the original scale.
The nested source permutation verification is inlined locally without changing
iteration, match/removal, failure or arithmetic order. A typing-only cast at the
NumPy root result boundary makes numeric roots iterable to the checker; it does
not convert, reorder, filter or repair runtime values.

Phase extraction preserves reverse-degree traversal, arctan2 magnitudes,
1e-10 zero-angle checks, source 2x2 adjoint update, coefficient slicing and
initial lambda. The numerical rotation formula retains source scalar-phase
multiplication, including the sign on the bottom-right entry. Native lambda is
a Python float (the source returns an int or NumPy scalar despite its int return
annotation); its numeric value and all angle entries match the original.
Unequal lengths retain the source ValueError. Empty pairs retain empty angle
arrays and zero lambda. Complementarity is assumed, not validated by this function.

The typed inputs are finite complex coefficient sequences. The numerical roots
and allclose behavior remain dependent on the NumPy/LAPACK environment. Exact
array equality is demonstrated here, not guaranteed across BLAS versions or for
all ill-conditioned high degrees. No symbolic `Shaped`, foreign arrays, arbitrary
dtypes, or full `GeneralizedQSP` wrapper/API acceptance is claimed.

## Source failures and proof limits

A bounded constant such as P=(0.5,) fails at `Polynomial.fromroots([])` in the
reference environment with `ValueError: Coefficient array is empty`. Tail-zero
P=(0.5,0) with verify=True triggers the source zero-root assertion. Both native
and unchanged source retain these failures. Empty inputs, zero polynomials and
unit-root pairing boundaries are compared directly. No roots are perturbed,
trimmed, sorted or replaced to obtain a passing result.

`verify=True` checks the algorithm's root properties. It does not first prove
that the supplied P is bounded on the entire unit circle, nor repair source
numerical instability. The previously recorded finite-grid missed-peak example
remains open. Synthetic root injection tests exercise the diagnostic branches
and order only; they are not independent complement identities. Independent
pointwise identities are tested for actual computed small-degree complements.

## Verification and resources

Random real and complex inputs with 2, 3, 5 and 10 coefficients are compared to
the original output coefficients with exact array equality. Actual complements
also satisfy independently evaluated |P(z)|**2+|Q(z)|**2 on a deterministic grid.
Root diagnostic tests cover zero, unit pair/unpaired, reciprocal mismatch and
matching roots. Phase boundary cases cover zero/tiny coefficients and empty pairs.

Eighteen length/real/negative-power scenarios compare all angles, then assemble
the source GeneralizedQSP signal sequence using the native angles and original
SU(2) matrices. The full complex 4x4 matrix matches the unchanged source circuit,
and both polynomial blocks match the existing independent dense evaluator with
negative Laurent offsets. This is a numerical reconstruction, not a new native
GQSP circuit emitter. Separately the existing physical SU(2) emission is executed
for each angle and both basis inputs, checking absolute complex phase and clean
zero. No phase-insensitive distance weakens these checks.

| Quantity | This numerical batch |
| --- | --- |
| New physical primitive gates / work wires / measurements | 0 / 0 / 0 |
| Complement work | Root solving and polynomial reconstruction, NumPy-owned |
| Phase work | n small matrix updates and shrinking 2-by-n coefficients |
| Numeric storage | Polynomial/root arrays; phase recursion O(n) temporary storage, peak not measured |
| Source QEC gate model | Owned by the GeneralizedQSP wrapper, not these numerical functions |

The existing SU(2) instruction stream and costs are unchanged. The source wrapper
counts n rotations, max(0,d-k) zero-controlled U, min(d,k) controlled U-adjoint,
and max(0,k-d) unconditioned U-adjoint (d=n-1). These are wrapper model counts,
not measured native GQSP primitive counts. There are no measurement branches in
this slice, no new hardware/gradient/scalability claim and no full-parent/G2 closure.
