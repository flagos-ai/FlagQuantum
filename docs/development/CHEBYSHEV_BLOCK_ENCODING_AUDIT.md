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

## Remaining scope

The concrete packed interface does not yet model symbolic-width
decomposition. The scaled coefficient transform, controlled-term
composition through the existing linear combination, and complete native
notebook are tracked in this same batch. The three file-level items remain
open until the scaled and tutorial ownership checks are complete.
