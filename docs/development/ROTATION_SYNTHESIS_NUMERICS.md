# Rotation synthesis numerical contracts

The internal synthesis scalar and arithmetic modules cover these Qualtran
v0.7.0 files at commit `bcb17bb25360ed68ce3bc084645db3e929799389`.

| Item | Frozen file | Git blob |
| --- | --- | --- |
| Q-0820 | `rotation_synthesis/_math_config.py` | `ea435fa802e0dabc272d3ac52065b709fd441b55` |
| Q-0821 | `rotation_synthesis/_typing.py` | `66bdc937ad691a24bb523d64c2d77cc34de36031` |

`flagquantum/synth/_rotation_types.py` supplies the numerical annotation
types and the runtime integral predicate. The predicate admits Python
bool and NumPy integer scalars as the source does. It rejects NumPy bool
and integer-valued floating and arbitrary-precision scalars.

`flagquantum/synth/_rotation_math.py` supplies immutable float, NumPy and
mpmath backends with the same callback injection, constant construction,
scalar operations and rounding. The NumPy constructor uses the platform's
`longdouble`. On the acceptance host that type has a 52-bit mantissa
field. The independent tolerance helper preserves the strict absolute or
first-operand relative comparison. The arbitrary-precision backend
preserves the inclusive decimal tolerance and the source's process-global
precision change. Previously created backends observe changes in later
operation calls. Tests scope this state with `mpmath.workdps`.

The native NumPy and float paths load without mpmath. Calling the
arbitrary-precision factory requires mpmath in the environment, as in the
locked reference environment. These files declare numerical operations.
Their direct rotation-synthesis consumers have separate ledger items for
channel, lattice, integer ring and matrix work. Circuit amplitudes,
primitive counts and scratch checks apply at those consumers.

## Integer square-root-two ring

| Item | Frozen file under `rotation_synthesis/rings/` | Git blob |
| --- | --- | --- |
| Q-0856 | `_test_utils.py` | `57d62f94b3aef1392b47487d8906c2cdbc246aa1` |
| Q-0857 | `_zsqrt2.py` | `c94516686151c5cda561e18eb045e0b54166d1f5` |
| Q-0859 | `zsqrt2_test.py` | `caa82503b022b450dc141b97b35e36f52e20d958` |

`ZSqrt2` stores the coefficients of `a + b*sqrt(2)` as immutable Python
integers. Ordering, norms, quotient rounding and the Euclidean GCD stay
in exact arithmetic. The port preserves the source's GCD unit selection,
prime factor order, constants and cached radii. Numerical evaluation uses
the configured square root of two. Prime operations require SymPy and load
it only when called. Integer-valued native elements hash like equal Python
integers, so either key finds the same dictionary entry.

All 3,858 original cases pass on the locked source and in the native
suite. Differential checks compare exact coefficient pairs, large-integer
ordering, Euclidean division, exceptional inputs and all three radius
backends. The original prime fixture remains test data. Complex ring,
lattice and matrix consumers keep their own acceptance items.

## Eighth-root cyclotomic integer ring

| Item | Frozen file under `rotation_synthesis/rings/` | Git blob |
| --- | --- | --- |
| Q-0855 | `__init__.py` | `4eb7949e5ab78f63a5d09ae391f8fd641d6b2552` |
| Q-0858 | `_zw.py` | `c8bf840eaa53797af105eb0c037193c294381868` |
| Q-0860 | `zw_test.py` | `cc136951d5e38da3bdfd56a42e71424049d13416` |

The internal ring namespace exposes `ZSqrt2` and `ZW`. The latter stores
the coefficients of the first four powers of the eighth root of unity.
Both conjugations, norm, Euclidean quotient and GCD, and the ordered prime
factor lists match the locked source. Subring conversion preserves the
parity bit that determines whether the eighth root is added. The numerical
evaluation and polar coordinates use the configured arithmetic backend.
Prime factorization uses SymPy on demand for the modular square roots.

The original 4,199 cases pass in both suites. Native randomized cases use
a fixed seed, and iterator parameter grids are materialized before pytest
collects them. The additional checks compare exact coordinates and large
integer GCDs against the source, both subring directions, type and value
of the three numerical backends, and the original errors on unsupported
exponents and scalar equality. No circuit or resource model is emitted
by this layer.

## Exact SU(2) Clifford+T matrices

| Item | Frozen file under `rotation_synthesis/matrix/` | Git blob |
| --- | --- | --- |
| Q-0840 | `_su2_ct.py` | `5096d82e23a7fc95ecbacd887952a0ed31946033` |
| Q-0844 | `su2_ct_test.py` | `9380fe8358cc30bbab4d0f2f41e91cf9414c7c58` |

`flagquantum/synth/_su2_ct.py` keeps exact scaled matrices over the
existing integer rings. It stores the circuit-order word when both
multiplication operands carry one. The source adjoint drops that word,
and scale reduction preserves it. Determinant, parametric coordinates,
phase selection for input pairs and the parity lookup operate on exact
coefficients. Float, NumPy and mpmath normalization use the existing
math configurations.

The complete original suite and the corresponding native cases pass.
The native tests use a fixed seed and materialize the parameter grids.
They compare coefficients, gate words, parity keys, primitive operands,
and both columns of the native CPU matrix against the locked reference.
The circuit tests bind words with the existing `Circuit.append(op)`
protocol and use one wire with no ancillary or measured registers.

The source normalizes H by `i`, S by `exp(-i*pi/4)`, X and Z by `i`,
Y by `-i`, and each positive axis T by `exp(-i*pi/8)` relative to
the physical gates or conjugation recipes. The tests apply each
specified factor, conjugating it for adjoints, before checking complex
amplitudes. The native X-axis recipe emits H, T, H in circuit order,
giving three primitives and one T. The Y-axis recipe emits S adjoint,
H, T, H, S, giving five primitives and one T. The Z-axis recipe emits
one T. The exact determinant model reports one T for each rotation.
The Z-axis rotation followed by its inverse emits T and T adjoint,
while the scale-reduced determinant model reports zero T.

The Clifford catalogue used by the original T-count test remains a
frozen test fixture until its generation module is ported. The exact
matrix module has no production dependency on that catalogue.
