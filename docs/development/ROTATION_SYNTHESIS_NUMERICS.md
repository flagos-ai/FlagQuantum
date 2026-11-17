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

## Exact Clifford and rotation generation

| Item | Frozen file under `rotation_synthesis/matrix/` | Git blob |
| --- | --- | --- |
| Q-0839 | `_generation.py` | `10822bf2efe2e2445bc6a593015bee3d98b227b0` |
| Q-0843 | `generation_test.py` | `2961ff16966b0ecd45e721ac39f5f22ea9c159f9` |

`flagquantum/synth/_rotation_generation.py` enumerates the 24 Clifford
representatives in source traversal order. It expands successive T levels
from the preceding frontier and excludes both an exact signed matrix
and its negation once either representative has been seen. The catalogue
is cached and supplies the exact matrix T-count test.

At levels zero through five the native and reference frontiers contain
24, 72, 168, 360, 744 and 1,512 matrices. All 2,880 comparisons cover
ordered words, exact coefficients, the determinant T exponent and both
complex columns. The words reconstruct their matrices with the same gate
order. This enumerator returns candidate matrices and words. Primitive
execution and phase correction remain covered by the one-wire circuit
checks above.

The progress indicator uses tqdm when requested and available. If tqdm
is absent, generation proceeds with the identical frontier. Verbose level
counts are still available. The native generator has no required tqdm
dependency.

## Exact Clifford+T words and export

| Item | Frozen file under `rotation_synthesis/matrix/` | Git blob |
| --- | --- | --- |
| Q-0838 | `_clifford_t_repr.py` | `62f821e1a048b374ec1c750c86523fba9a3fc93a` |
| Q-0842 | `clifford_t_repr_test.py` | `733c3e5921eacbd0ec94b68a42fcd3cd2b9e078d` |

Synthesis owns the exact Clifford lookup, XYZ parity stripping and XZ
search. Ecosystem owns Cirq operation export and Quirk cell tokens.
Cirq uses the existing optional loader. Quirk exports the original cell
tokens in both phase modes, including the SXSX prefix in the default
mode. The tests compare serialized cells directly. Browser execution
is outside that token encoder. Tests compare Cirq operation order, gate
parameters and target qubits to the locked export.

All 150 original cases pass in the native suite. The full zero- to
two-T frontier contributes 264 matrices for each word format. The
locked XZ search returns no word for 24 unrescaled entries at level two.
Calling the source's `rescale()` makes each of those entries
representable. The direct word/export functions retain the original
admission behavior. The native `Circuit.append(op)` binding applies
that same reduction before requesting XZ, as the original XZ test does.

The append binding emits the synthesized Clifford+T word on one existing
wire. It uses the accepted scalar phase primitive when the signed word
has a nonzero cumulative normalization. That correction emits X,
phase, X, phase with no T or scratch. The complete complex two-wire
operator checks cover the rotated wire and its spectator. For a single
Tz, the binding emits one T plus the four scalar operations. Its
determinant model reports one T separately. The unrescaled double Tx
emits two T in XYZ and zero T in XZ, where the matrix is first reduced.
Neither path allocates work or performs measurements.

## Analytical ZXZ decomposition and matrix exports

| Item | Frozen file under `rotation_synthesis/matrix/` | Git blob |
| --- | --- | --- |
| Q-0836 | `__init__.py` | `b449545fa983766987c78e335d781eb4476f55b0` |
| Q-0837 | `_analytical_decomposition.py` | `98faf18574ecb686b4961912e894c4a2712f78c8` |
| Q-0841 | `analytical_decomposition_test.py` | `b7766cfafa47afd7e8384c8ccef5415b684d33fa` |

The analytical helper accepts an SU(2) matrix and a math configuration,
then returns `theta1, phi, theta2` for the product
`Rz(theta1) @ Rx(phi) @ Rz(theta2)`. Append applies `theta2` first.
The helper preserves the locked cosine clipping and angle branches.
The caller supplies an SU(2) matrix, as in the original. Near a diagonal
matrix, source floating rounding can make the reconstructed operator
differ from the input by about 7.5e-9 in an off-diagonal entry. Native
CPU execution compares both complex columns directly to the locked
reconstruction at 1e-12 absolute tolerance. The 80-decimal configuration
also preserves the original object-array arithmetic.

The analytical conversion emits no circuit itself. Binding its angles
through native append in the execution tests emits three rotation
primitives on the target wire, with no scratch, measurement or
feedback. This check counts the emitted rotations before any
Clifford+T approximation.

The nine callable exports of the frozen matrix package have these
native owners.

| Exports | Owner |
| --- | --- |
| `su_unitary_to_zxz_angles`, `unitary_from_zxz` | `synth/_analytical_decomposition.py` |
| `to_sequence` | `synth/_clifford_t_repr.py` |
| `to_cirq` | `ecosystem/cirq/_rotation_export.py` |
| `to_quirk` | `ecosystem/_quirk_rotation.py` |
| `generate_cliffords`, `generate_rotations`, `generate_rotations_iter` | `synth/_rotation_generation.py` |
| `SU2CliffordT` | `synth/_su2_ct.py` |

The package audit checks the complete inventory. Synthesis callers use
the native math and gate-word modules directly. The ecosystem boundary
owns external representations and optional dependencies.

## Rotation synthesis channel models and execution

| Item | Frozen file under `rotation_synthesis/channels/` | Git blob |
| --- | --- | --- |
| Q-0822 | `__init__.py` | `ab2de75e6d347d04863ca14fae2055675b929ae3` |
| Q-0823 | `_channel.py` | `0e835f66ff2dbced69f31bbf368badf74d9444de` |
| Q-0824 | `channel_test.py` | `cca8d8d4825f0961bc86bea4c508d3a6d3030971` |

Synthesis owns all four original exports in `synth/_rotation_channels.py`.
The unitary model preserves the exact ring pair, determinant scaling,
phase selection and twirled and untwirled distance formulas. The
projective model preserves success probability, conditional correction
cost and distance. The probabilistic model preserves weight clipping,
matched channel kinds, interpolation and the mixed-kind distance error.
Twirled and probabilistic channels have source resource and distance
semantics. They lack an executable export in the locked implementation.

The Cirq and Quirk adapters stay in Ecosystem. Cirq preserves the
controlled frozen correction and the original measurement key. Quirk
URLs preserve the serialized cells, operand rows and phase modes.
The native append binding accepts an existing zero ancilla and a caller
chosen classical bit. It reuses the exact Clifford+T word binding.

The locked projective diagram shows a Y before correction, while both
locked exporters apply the correction directly. The native operation
follows the executable path and restores the ancilla with a conditional
X. With normalized rotation `U` and correction `C`, the unnormalized
target maps are `diag(U)` and `C @ (U - diag(U))` for outcomes zero
and one. The tests compare their complete complex columns and outcome
probabilities, with the ancilla at zero in both branches. This is
branch parity for the executable path. The analytical distance model
is compared directly to the locked numerical model.

For rotation Tx and correction Tz, the emitted path contains two CX
and one measurement besides the rotation and conditional correction.
Each word emits one T or T adjoint, with its own Clifford gates and
scalar correction as needed. The final conditional X adds one
primitive and zero T. The expected T model weights the second T by
the outcome-one probability. The measured bit retains its result
after cleanup. All 103 original test cases pass in the native suite.

## Exact lattice generation

| Item | Frozen file under `rotation_synthesis/lattice/` | Git blob |
| --- | --- | --- |
| Q-0825 | `__init__.py` | `62babd8ac6ef95657a690e1a6bb74ee764066b5d` |
| Q-0826 | `_geometry.py` | `946d4f89c86d28e0b109e2f4bb89cc6468d9088b` |
| Q-0827 | `_grid_operators.py` | `6d2f4020b7e988a83a82068256ad639d52ecea1a` |
| Q-0828 | `_ipe.py` | `fb54fb68cc0cf145bd91ce518ff135f08be3da46` |
| Q-0829 | `_state.py` | `67e66a6804e39459c8ff1d33c968b18960bec8d5` |
| Q-0830 | `_test_utils.py` | `e2e23df593af06af3bff4756658a8fcc73306fd5` |
| Q-0831 | `geometry_test.py` | `1a6aed7a6f4f671494532a609bd4abfb74c10d3e` |
| Q-0832 | `grid_operators_test.py` | `d4ebaa7ed4e6691bee16c30be22ea65a8dd2fbec` |
| Q-0833 | `ipe_test.py` | `8c890d4bd02bae53e9fe5c433e3c6e31a7477cfc` |
| Q-0834 | `state_test.py` | `f051a3cee6f04e7b016e2341db7d323831e69412` |
| Q-0835 | `test_utils_test.py` | `db26d880030601269f8d12f3ec3d2f19f5c879b8` |

The internal lattice package exposes the ten locked symbols. Geometry
owns scalar ranges, ellipse transforms and optional plotting. Exact grid
operators keep the source square-root-two scaling, coefficient parity,
composition order and conjugation. State transforms retain the ordered
action path and center transforms. Enumeration preserves the ring
coordinate order, including the two eighth-root cosets in the upright
case. The shared random fixtures live in tests.

All 6,246 original cases run natively. Differential checks compare
complete ordered point streams, exact grid coefficients and floating
state matrices under NumPy and mpmath. The plot check compares the
Matplotlib patch geometry and style. Iterator parameters are
materialized, the randomized geometry cases use a fixed seed and the
mpmath tests scope the precision change. Lattice generation is a
classical synthesis input and has no quantum primitive counts.

The locked grid power routine uses its existing behavior for negative
indices, which return the operator instead of its inverse. The locked
one-dimensional enumeration requires nonzero interval width and
propagates the configured logarithm's domain error. Native data classes
support an identity comparison for a reused ellipse. Comparing
separately constructed array-backed ellipses retains the NumPy
ambiguous-truth error. Use explicit array comparison for numerical
geometry checks.

## Clifford+T relative norm

| Item | Frozen file under `rotation_synthesis/relative_norm/` | Git blob |
| --- | --- | --- |
| Q-0852 | `__init__.py` | `7d471a7490ccbf507febcb821fab9bfca37cf32d` |
| Q-0853 | `_clifford_t.py` | `afee83bf529d5d59495b92c760d66d995351c40a` |
| Q-0854 | `clifford_t_test.py` | `f6d39297d0a67c3b72c9835eb35817cab5eb82a5` |

The internal solver reuses the accepted integer rings. It preserves the
ordered rational and ideal factors, the four root candidates, unit
selection and the locked inert-prime rejection. It loads SymPy when
factoring a nontrivial target. The 600 original cases pass and the
native random sample has a fixed seed. Additional cases compare exact
witness coordinates and verify each accepted pair by recomputing the
relative norm. For example, the target seven has no witness.

Negative one falls outside the relative norm image. The locked solver
returns one for this input, whose recomputed norm is positive one.
This out-of-domain behavior is preserved and tested as a diagnostic.
The approximation protocol's locked call site checks the witness
before accepting a point. The solver itself is classical and emits no
quantum primitives.

## Rotation approximation protocols

| Item | Frozen file under `rotation_synthesis/protocols/` | Git blob |
| --- | --- | --- |
| Q-0845 | `__init__.py` | `a869319efbbca01d1baa2b9c84bef56cb2c07eb5` |
| Q-0846 | `_clifford_t_synthesis.py` | `22a9071e3d26377deb39a149480c72b3a5a13cc9` |
| Q-0847 | `_diagonal.py` | `32362fde0005c870378186459e9e689bdc1f0a73` |
| Q-0848 | `_fallback.py` | `330ff3201fbcf41637c06e927c3f87f363006996` |
| Q-0849 | `_mixed_diagonal.py` | `ac3b5a49eb46b8e9be929338c24a0a0a4fb2fe99` |
| Q-0850 | `_protocol.py` | `5082ab37b791e5de48cb0d5cb13fcc5c605070b7` |
| Q-0851 | `clifford_t_synthesis_test.py` | `b2cefd61bac6d1636c57865f64464a59fd359955` |

The internal protocol package owns the three search regions, ordered
candidate stream, split-region tie rule and twelve locked exports.
The five synthesis entry points preserve the source cutoffs, selection
order and exact relative-norm witness check. The tests exercise all 90
original cases and compare exact ring coefficients, nested channel
shape, mixture probabilities, distance and T models directly against
the frozen source. Optional Matplotlib is loaded in the plot methods,
and patch geometry is checked for all three regions.

The finite-search exhaustion cases preserve the locked behavior.
Diagonal returns None. Fallback may attempt to iterate a missing result.
Mixed diagonal may attempt to unpack a missing result. Mixed fallback
reports the failed region with a ValueError. Its nested correction
searches use their default relative norm solver, as in the source.
The mixed-fallback cutoff retains the source mpmath default when mpmath
is installed and uses the same decimal value as a float otherwise.

The deterministic and projective results reuse the accepted native
append bindings. At theta 0.9, error 0.03 and success threshold 0.99,
the projective binding emits 47 unconditional primitives including the
measurement, and 45 primitives conditioned on outcome one including the
ancilla cleanup. The unconditional rotation and conditional correction
each emit nine T gates. The source weighted model reports about 9.01413
T. Both unnormalized complex columns match in each measurement branch,
with operand and condition checks and the ancilla returned to zero.
Mixed and twirled results keep the analytical channel model and
resource semantics described in the preceding section.

## Running the source examples

| Item | Frozen file under `rotation_synthesis/` | Git blob |
| --- | --- | --- |
| Q-0818 | `README.md` | `ffe88969b5c173869c7012b8111da7ecefd6de36` |
| Q-0819 | `__init__.py` | `9a8b46f674a9a464b37c146a4fb9ee9f21d57db5` |

The four original examples approximate `exp(i theta Z)`, which is
`Rz(-2 * theta)`, at theta 0.1 and error 1e-8. The internal synthesis
owners provide these entry points for integration. The example scopes
the requested mpmath precision because `with_dps` sets it process-wide.

```python
import mpmath
import torch

from flagquantum import Circuit
from flagquantum.dynamic import DynamicCircuit
from flagquantum.synth import _rotation_protocols as synthesis
from flagquantum.synth._clifford_t_repr import to_sequence
from flagquantum.synth._rotation_channel_operation import (
    ProjectiveChannelOperation,
    UnitaryChannelOperation,
)
from flagquantum.synth._rotation_math import with_dps

with mpmath.workdps(300):
    config = with_dps(300)
    theta = 0.1
    eps = mpmath.mpf(1e-8)
    diagonal = synthesis.diagonal_unitary_approx(theta, eps, 400, config)
    fallback = synthesis.fallback_protocol(theta, eps, 0.99, 200, config)
    mixed_diagonal = synthesis.mixed_diagonal_protocol(theta, eps, 300, config)
    mixed_fallback = synthesis.mixed_fallback_protocol(theta, eps, 0.99, 300, config)
    assert diagonal and fallback and mixed_diagonal and mixed_fallback

    for channel in (diagonal, fallback, mixed_diagonal, mixed_fallback):
        expected = float(channel.expected_num_ts(config))
        distance = float(channel.diamond_norm_distance_to_rz(theta, config))
        print(f"{expected:.6f} T, diamond distance {distance:.8e}")

    exact_matrix = diagonal.to_matrix()
    word = to_sequence(exact_matrix)
    circuit = Circuit(1, dtype=torch.complex128).append(
        UnitaryChannelOperation(diagonal, 0, "xyz")
    )
    measured = DynamicCircuit(2, dtype=torch.complex128).append(
        ProjectiveChannelOperation(fallback, target=0, ancilla=1, classical_bit=0, fmt="xyz")
    )
    print(len(word), len(circuit.to_ir().instructions), len(measured.to_ir().instructions))
```

| Protocol | Model T count | Model diamond distance |
| --- | ---: | ---: |
| Diagonal | 80 | 8.85416237e-9 |
| Fallback | 32.335168 expected | 8.35838888e-9 |
| Mixed diagonal | 40.645503 expected | 9.33668500e-9 |
| Mixed fallback | 18.982145 expected | 5.39736297e-10 |

The fallback model uses 32 T in the projection and 65 T in the
correction, which happens with probability about 0.005156. With the
XYZ binding above, the diagonal word has 82 labels and emits 248
primitives. The fallback IR contains 323 primitives, including
conditional correction and cleanup. Use `measured.to_ir().instructions`
to inspect the classical conditions. The mixed forms expose distance
and weighted T models for the channel mixture.

Optional exports of the deterministic and projective results use the
Ecosystem adapters. Running Cirq export requires the Cirq extra.

```python
from flagquantum.ecosystem._quirk_rotation_channels import (
    export_projective_channel_quirk,
    export_unitary_channel_quirk,
)
from flagquantum.ecosystem.cirq._rotation_channels import (
    export_projective_channel_cirq,
    export_unitary_channel_cirq,
)

diagonal_cirq = export_unitary_channel_cirq(diagonal, "xz")
fallback_cirq = export_projective_channel_cirq(fallback, "xz")
diagonal_quirk = export_unitary_channel_quirk(diagonal, "xz")
fallback_quirk = export_projective_channel_quirk(fallback, "xz")
```

The locked guide uses roughly `10 * log10(1 / eps)` decimal digits as
an empirical starting point. Low precision can miss a candidate or
trigger a numerical error. Higher precision increases runtime. The
finite-search return and exception behavior is described above.

The ten root exports map to the existing native modules.

| Exports | Owner |
| --- | --- |
| `NumpyConfig`, `with_dps` | `synth/_rotation_math.py` |
| `to_sequence` | `synth/_clifford_t_repr.py` |
| `to_cirq` | `ecosystem/cirq/_rotation_export.py` |
| `to_quirk` | `ecosystem/_quirk_rotation.py` |
| `diagonal_unitary_approx`, `fallback_protocol`, `magnitude_approx`, `mixed_diagonal_protocol`, `mixed_fallback_protocol` | `synth/_rotation_protocols/` |
