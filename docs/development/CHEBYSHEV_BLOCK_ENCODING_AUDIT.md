# Chebyshev block encoding source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen files under `qualtran/bloqs/block_encoding/` are Apache-2.0,
copyright Google LLC.

| Item | Source file | Frozen Git blob |
| --- | --- | --- |
| Q-0279 | `chebyshev_polynomial.ipynb` | `30fa0a9688a8704c7ed9cc32619ca7ed96c93ada` |
| Q-0280 | `chebyshev_polynomial.py` | `327d0ace4b411c9d8290d59d496e717b0c9e3915` |
| Q-0281 | `chebyshev_polynomial_test.py` | `2bc29ea862924b8e213d94b9f3f15ff1c53414b9` |

## Unscaled block

`ChebyshevBlockEncoding` retains the child's separately ordered system,
ancilla and resource ports, uses identity signal preparation, returns alpha
one and scales the child error by the polynomial order. The binding follows
the original reflected forward/adjoint pairs and its optional forward tail.
Degree zero emits no child calls. The zero-width ancilla omits reflections.
The exact adjoint reverses the call order and switches each child direction.
A positive outer control binds the reflection on the same predicate as the
supplied controlled child.

The reflection compiler owns its phase target, predicate workspace and
classical bits. Each reflection releases its work before the child call.
Separate child and reflection work reservations can overlap between these
sequential stages. The source forward and reverse call order and complete
complex output agree for orders 0, 1, 2, 4 and 5 across the tested layouts,
including all histories of the two-reflection, three-ancilla-bit case. The
unmodified original order-four matrix fixture gives the same projected block.
Independent reflection tests cover the primitive mark, feedback, phases,
work cleanup and source operand order.

The direct source graph omits the final odd reflection count when ancilla
width is nonzero. The executable source decomposition includes that
reflection. The model therefore exposes both routes. Physical emissions
retain the decomposition sequence. For order two with three ancilla bits
and a T/T-dagger child, the physical stream contains two child T/T-dagger
instructions plus 16 T/T-dagger instructions for the four measured
conjunctions. Four measurements are emitted, and three reflection work
qubits are reused. The model counts the two child T gates, four And
operations and four measurements separately.

## Scaled block

The scaled block retains the source power-basis conversion, coefficient-wise
multiplication by the child normalization, and conversion back to Chebyshev
basis. Its terms include zero coefficients and preserve degree order. Orders
zero and one directly expose the unscaled child and retain its normalization
one, including when the original input normalization differs. The model
delegates to existing LCU metadata and signal preparation. The caller binds
controlled terms through the shared LCU SELECT callback and appends the
resulting operation to the circuit.

The original fourth- and fifth-order fixtures match the native coefficients,
normalization and packed widths. The native second-order, one-bit
coefficient-precision fixture with one child ancilla matches complete complex
outputs for forward and adjoint across all 32 measurement histories per
direction. The native state has zero in every temporary work bit at the end
of each branch. Its direct gate model matches the locked source, including
58 Cliffords, four Fredkin gates, five And calls, five measurements and peak
width 11. The decomposition model includes one additional controlled
reflection at degree one, adding four modeled Cliffords. The executable
stream emits 48 T/T-dagger instructions from four seven-T Fredkins and five
four-T conjunctions, plus five measurements. The default source
T-equivalent rates assign four T equivalents to each Fredkin.

## Test coverage and remaining scope

The unchanged original non-notebook suite passes 29 cases with two mode
skips and three expected serialization failures. The original notebook
executes all 20 cells independently. The native 22-cell tutorial preserves
the four original fixtures, two signature figures, two call graphs, and the
unscaled/scaled derivations. It adds an executable append and inverse example
covering the full odd-block matrix.

Q-0279 closes with the native tutorial. Q-0280 remains open for symbolic-width
binding and generic source object and serialization protocols. Q-0281
remains open for native full-state execution of the original large scaled
fourth- and fifth-order fixtures and the entire random-matrix family.
The small executable scaled fixture provides independent evidence at its
own bindings and precision.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_chebyshev.ipynb
```
