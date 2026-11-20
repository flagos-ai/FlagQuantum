# Monomial unitary validation

## Decision and compatibility

Integration records this implementation-only optimization before the Core edit.
The existing explicit unitary contract remains the authority. Immutable built-in
complex rows, square shape, finite entries, the absolute 1e-10 tolerance and
existing diagnostic text retain their behavior. Public APIs, serialized payloads
and capability admission retain their current contract. Public API Protection
expressly permits private performance algorithms to evolve under that contract.

For a matrix with exactly one nonzero entry in every row, unitarity is equivalent
to unique occupied columns and unit squared magnitudes at the existing tolerance.
The column inner products have disjoint support. Their diagonal sums each contain
one nonzero term, so this proof preserves the original floating-point arithmetic
and tolerance. All rows must meet the monomial condition before this path decides
validity. Other matrices retain the complete column-inner-product check.

Core keeps the proof dependency-free and backend-neutral. Ecosystem continues
owning foreign matrix construction. Compiler, binding and simulation continue
owning execution. The optimization changes validation cost for dense-stored
reversible lookup/permutation matrices from cubic to quadratic in their matrix
dimension, matching the cost of reading the payload.

## Evidence required

Compare the complete previous validator on every signed/phased small permutation,
invalid repeated columns, norm perturbations around tolerance, zero rows, dense
unitaries, arbitrary dense matrices and malformed inputs. Reuse payload tamper
and lowering tests. Repeat the bounded actual controlled-QROAM intake and report
the next reached boundary. Gate streams, amplitudes and workspace remain governed
by their existing independent algorithm tests.
