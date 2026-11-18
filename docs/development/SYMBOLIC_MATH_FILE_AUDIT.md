# Symbolic mathematics whole-file audit

The reference is unmodified Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, covering the complete
`symbolics/math_funcs.py`, its tests, `types.py`, its tests and package exports.

Core `value_math.py` owns neutral compile-time scalar mathematics. The existing
source-ordered width functions, sums and products remain authoritative. This
batch adds the remaining natural logarithm, exponential, principal argument,
absolute value, square root, arccos, conjugate, pi selection and zero predicate.
All symbolic dispatch uses Core `is_symbolic_value`. SymPy remains a lazy
optional dependency. Concrete operations retain the original math or NumPy
choice and conversion order. Numeric type aliases describe static inputs and
introduce no runtime wrappers.

Simulation `value_dimensions.py` continues to own numerical shape and length
descriptors and their queries. Existing shape, classifier, width and resource
consumer evidence is reused. Package-level source names map to these existing
native owners without introducing a second package identity.

The source zero predicate can return `None` from symbolic `equals(0)`, despite
its bool annotation. The native annotation records that result. Only ValueError
is caught, as upstream. Unknown protocols, other exceptions, negative square
root inputs and NumPy out-of-domain warnings retain their source behavior.
Descriptor class names and tuple-validator diagnostics remain native as already
recorded in NUMERICAL_VALUE_DIMENSIONS.md. Physical widths must be resolved at
the existing admission boundary.

## Source-to-native ownership

| Source names | Native owner |
| --- | --- |
| All 17 exported mathematical functions | `core/value_math.py` |
| `SymbolicInt`, `SymbolicFloat`, `SymbolicComplex` | Core scalar type aliases, with native `SymbolicReal` naming and Python numeric widening |
| `is_symbolic` | `core/value_parameters.is_symbolic_value` |
| `Shaped`, `HasLength`, `slen`, `shape` | Simulation `SymbolicArrayShape`, `SymbolicSequenceLength`, `value_length`, `value_shape` |

Original math tests execute unchanged with only their imported helper bindings
replaced. Both values of the original parametrized symbolic-bit-length test and
all five original shape cases pass. Additional differential checks cover NumPy
scalar precision, arrays, exact expressions, domain errors, warning streams,
custom protocol order and undecidable or failing zero queries. Existing complete
width, classifier and shape suites cover the remaining source behavior.

The actual source GlobalPhase and SU(2) consumers execute with native pi,
exponential and argument bindings. Their accepted append scenarios retain exact
instruction order, operands, complete complex amplitudes, adjoint recovery and
JSON snapshots. The coherent carrier stays clean. These scenarios contain no
measurements. Existing QROM/QROAM and measured-control suites in the regression
retain branch and workspace evidence for width and shape consumers.

Scalar helpers emit zero quantum primitives and own no work wires. The phase
consumer emits two X and two phase operations for a nonzero exponent. SU(2)
emits seven primitives. Source QEC category models and logical peaks are checked
independently by the reused consumer scenarios. Neither those counts nor a
mathematical function call is treated as a synthesized-resource estimate.

The batch passed 1438 concentrated cases with no warnings or skips. Changed-file
strict typing, formatting, lint, Core scope, architecture, multilevel IR and
protected API checks passed. Reference blob hashes are asserted for all five
source files. Retained logs use `ftqc-reference-env/g2-symbolics-file-*.log`.

The lazy optional SymPy import policy is native. The reference imports SymPy
when its package loads. Missing-dependency behavior is therefore documented
at the native dependency boundary. Descriptor validation messages retain native
names. Compiler's existing budget-specific definite-zero policy remains a
separate consumer contract. Complete source mathematical behavior, original
tests and export ownership are accepted for Q-0932 through Q-0936 within these
recorded adaptations.
