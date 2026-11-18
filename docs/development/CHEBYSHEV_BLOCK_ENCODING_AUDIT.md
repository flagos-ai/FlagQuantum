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

Q-0279 closes with the native tutorial. Both native constructors now expose
the packed through signature, including zero-width omission. The scaled
constructor also declares the target, selection and junk registers in source
order. Q-0280 remains open for symbolic-width binding and the generic source
object and serialization protocols.

Q-0281 closes with the original concrete tests and native counterparts.
All 20 matrices from the original seeded random-Hermitian family compare
both the projected block and the full eight-dimensional logical operator.
The full comparison starts with clean scratch for every logical column
and checks that scratch is zero at the output.

The original unscaled order-four and order-five fixtures compare complete
complex columns for both system basis inputs with zero ancilla and resource.
The order-four circuit covers every measurement history; the order-five
circuit has no measurement. For the original large scaled cases, the source
contracts a vector for each system basis input and keeps every logical
output. Sparse execution compares every populated amplitude and its index,
including any nonzero ancilla and resource outputs. The all-zero and
alternating measurement histories each agree, and every temporary work bit
ends at zero. This covers two histories per scaled case, where exhaustive
history counts would be exponential.

| Scaled order | Reserved native width | Emitted T/T adjoint | Emitted measurements | Source peak | Source And | Source Fredkin |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 39 | 426 | 61 | 37 | 61 | 26 |
| 5 | 32 | 178 | 34 | 32 | 34 | 6 |

Reserved width is the explicitly bound native register pool. The source
peak is the independent QubitCount graph result. Native T emission charges
four per measured And and seven per exact Fredkin. The source QEC model
reports those operations as distinct categories. The original non-notebook
suite includes four slow tests, two upstream skips and three expected
serialization failures.

## External recipe intake and remaining closure

The shared optional component snapshot adapter now invokes the unchanged
`TwoBitCSwap.to_clifford_t_circuit()` recipe with its explicit `ctrl`, `x`,
`y` named-wire mapping. This atomic source exposes no standard decomposition;
the adapter uses its own exact public recipe, not a synthesized matrix or a
replacement algorithm. Forward and adjoint captures preserve all 17 literal
instructions, including seven T/T-dagger gates, and the full complex Fredkin
matrix through native IR JSON roundtrip. The entire source swap file is pinned
to blob `5c2481f7a1ee0a592d1192999ddf7e6dba8facc3`.

Initial Chebyshev ordinary-direction intake failed at this atomic leaf. With
the source recipe available, the original unscaled even and odd factories
capture both directions: respectively 520 instructions/eight reserved wires,
and five instructions/one wire per direction. Their private workspace and
classical slot requirements are `(2, 8)` and `(0, 0)`. The original scaled odd
factory captures 989 instructions/32 wires in each direction with `(14, 34)`
requirements. These are capture statistics, not independent logical models or
complete numerical acceptance of those snapshots.

The original scaled even capture now reaches
`Controlled(SingleQubitCompare)`. Its compute/adjoint partial registers have
no source-controlled graph or Cirq decomposition. Direct probes test that
failure unchanged. The adapter does not fabricate a controlled isometry.
The original even factory protobuf check fails on `SpecializeOnCtrlBit.ONE`;
both scaled factory deserializations cannot resolve the original class name;
the unscaled odd factory serialization passes. These distinct outcomes are
recorded separately from native IR serialization.

Q-0280 remains open. Next requirements are consumer-level external child
replacement, symbolic declaration/resolution, and complete large-history
evidence reusing the existing native coefficient and reflection owners.
Three captured source roots alone do not close the file. In particular, the
large scaled cases above still have representative numerical histories, not
an exhaustive numerical history enumeration.

The recipe dependency passes 119 concentrated external-block, shaped-port,
feedback and sparse-oracle regressions; its final exact recipe and original
comparator-boundary tests pass separately. Strict mypy, Black, Ruff, owning
team scope, architecture, multilevel IR and protected API checks pass.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_chebyshev.ipynb
```
