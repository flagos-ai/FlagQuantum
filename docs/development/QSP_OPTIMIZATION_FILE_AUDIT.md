# Complete optimization-based QSP complement source

The frozen Qualtran v0.7.0 commit is
`bcb17bb25360ed68ce3bc084645db3e929799389`. Q-0649 maps the complete
`qsp/fast_qsp.py`, blob `eada77ac614c12dc6c40def0e5a034678d05659d`.
Q-0650 maps its complete test file, blob
`e03d0a478f84ae0c3bcb6783b4498b3229592f70`. The native owner is
`simulation/numerics/qsp_optimization.py`, retaining Google LLC 2023
Apache-2.0 attribution. This is `single_device_fast_path` numerical work.

## Complete responsibility and interface mapping

| Original responsibility | Native responsibility |
|---|---|
| `FastComplementaryQSPHelper.__init__` | `ComplementaryPolynomialObjective` retains float64/complex128 assertions, original flip convolution and center coefficient replacement. |
| `loss_function` | Exact squared norm of the convolution difference in the original operation order. |
| `array_to_complex` | First-half real, second-half imaginary coefficient packing. |
| `conv_by_flip_conj` | Same NumPy convolution, flip axis and full mode. Axis `(0,)` supplies the typed equivalent of source `[0]`. |
| `complex_conv_by_flip_conj` | Same four split-real convolutions, subtraction and complex assembly. |
| `fast_complementary_polynomial` | Same ascending coefficients, explicit RandomState, real/complex Gaussian draw count, normalization, SciPy minimize call with `jac="3-point"`, tolerance, and returned iterate. |

SciPy is loaded when the numerical solver is called, matching existing optional
numerical dependency ownership. Algorithms' existing `GeneralizedQSP` consumes
its coefficients. Compiler owns signal rotations and unitary lowering. The caller
owns the unitary, target register order and scratch lifetime. The numerical
module consumes NumPy arrays and randomness, emits coefficient arrays, and emits
zero circuit instructions. Arrays use ascending polynomial degree. Input arrays
are preserved. Each objective uses linear storage and dense quadratic convolution
work per evaluation.

The normal circuit workflow remains

```python
q = fast_complementary_polynomial(p, np.random.RandomState(93))
op = GeneralizedQSP((("q", 1),), unitary_emitter, p, tuple(q))
circuit.append(op.make_on(signal=0, target=(1,)))
```

## Source and numerical boundaries

The source returns the optimizer iterate even when `success=False`. This behavior
is preserved and explicitly tested by replacing the optimizer at both boundaries.
Original invalid dtype and empty-array failures are preserved. Optimizer traces
match initial coordinates, RNG consumption, objective value, finite-difference
choice and caller tolerance. Default complex-mode and real-mode results match
source coefficients exactly.

A deterministic real-mode counterexample is retained. For polynomial RNG42 and
optimizer RNG91, lengths four/eight have maximum unit-circle residual about
0.02339667/0.00545494 on 129 fixed points. Native and unchanged source outputs are
identical. These inputs provide source-parity and failure evidence. Successful
fixtures are separately checked against the complement identity. The source's
high-degree complex error TODO remains an upstream boundary.

All four original test groups execute, including slow degrees20/30. The unchanged
source suite passes28 cases with one original imported notebook-marker warning.
A second run replaces only the solver with the native callable while retaining
the original tests and assertions. Its result and concentrated batch checks are
recorded below after execution.

## Append, amplitudes and independent resource accounts

Real and complex three-coefficient pairs feed the existing GQSP emitter. Complete
operators and a coherent complex input match unchanged source GQSP, including
absolute phase, for negative powers0/1/5. One idle qubit stays exactly zero. This
fixture borrows zero scratch wires and has one unitary branch with zero
measurements. Existing accepted measured-unitary GQSP evidence covers its separate
branching provider contract.

| Negative power | Actual emitted primitive inventory | Separate declared model |
|---|---|---|
| 0 | 10 X, 6 phase, 6 Rz, 3 Rx, 2 controlled phase | 3 signal rotations and 2 open-control U calls. Child model11 rotations, peak2. |
| 1 | 8 X, 6 phase, 6 Rz, 3 Rx, 2 controlled phase | 3 signal rotations, 1 open-control U, 1 controlled adjoint. Child model11 rotations, peak2. |
| 5 | 6 X, 9 phase, 6 Rz, 3 Rx, 2 controlled phase | 3 signal rotations, 3 bare adjoints, 2 controlled adjoints. Child model14 rotations, peak2. |

The child model declares three rotations per signal rotation and one per chosen
Z-power unitary. Its continuous-rotation counts are independent of synthesis
prices and of the physical phase/X inventory. The accepted GQSP model suite
checks source graph keys, multiplicities, child widths and inactive costs.

## Batch acceptance

The native solver passes all28 original cases with unchanged assertions and zero
warnings. Concentrated dependency regression passes390 cases without skips or
warnings. After annotation-only fixes, all40 new solver/FFT cases pass again.
Strict mypy covers both complete numerical owners and both tests. Black/Ruff,
Simulation scope, architecture, multi-level IR and protected API checks pass.

Q-0650 closes the original test responsibility. On 2026-11-26 the user explicitly
accepted Q-0649 as a complete faithful migration, with the two real-mode precision
counterexamples retained as separate optimization work. The existing complete
source adaptation, direct consumers and original-test evidence above are reused.
This does not certify successful optimization or a complement-identity bound for
the failing inputs. See the [approved disposition](G2_UPSTREAM_DEFECT_DISPOSITION.md).
The reference implementation and tests retain their frozen blobs.
