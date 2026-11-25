# Polynomial view source acceptance

Scope: frozen Qualtran v0.7.0 `gf_poly_split_and_join.py` and its complete
57-line test file. The implementation parent Q-0533 remains open; the test-file
parent Q-0534 has a complete assertion and raw protocol mapping below.

## Ownership and interface

Compiler owns `field_polynomial_component_program` and coefficient layout;
Simulation owns field and polynomial numerical codecs. Existing typed ports
carry the packed polynomial and the coefficient field. Coefficients run from
highest degree to constant term, each in MSB order. Split and Join alias exactly
the same physical wires. Neither owns work qubits, measurements or feedback.
Users continue to append the bound component to a circuit.

## Verification

The concentrated polynomial/storage/control regression passed 114 cases;
the independent Cirq 1.6 path passed 57. The unchanged complete original test
file passed five tests, with six original skips and two original xfails.
Strict typing, Black, Ruff, Integration scope, repository language, links,
architecture, IR contracts and API snapshots passed. Kickmix 0.1's native
Simulator was verified present. Earlier accepted execution evidence is reused.

The two actual source factories have ten raw protocol outcomes: two PASS,
four NA, two MISSING counts and two FAIL serialization. Split fails on QGFPoly
and Join on QGF; these failures are retained rather than counted as passes.
No source or reference test was modified.

## Complete original test mapping

| Original assertion | Native or source evidence |
|---|---|
| Split and Join actual factory autotesters | `test_polynomial_view_source_protocols.py` observes all five raw protocols for both original factories, before skip policy. Native typed signatures and verifier are checked against actual source ports. |
| Symbolic polynomial degree constructor refusal | Existing `test_field_polynomial_file_audit.py` preserves both original constructor refusals. The unchanged original suite is executed independently. |
| Classical padded coefficients and round trip | The same existing owner calls both actual source factories, preserves `[0,1,2,3,4]`, and compares both native codec directions. |
| Both exact 64-dimensional operators | Existing full tensor and native complex-state identity assertions are reused. |

The ten raw protocol outcomes are two PASS (make), four NA (atomic
decomposition and qtyping), two MISSING (counts), and two FAIL (serialization).
Split rejects QGFPoly serialization; Join encounters QGF first. These are source
results, not successful native serialization claims. The unchanged original
suite reports 5 passed, 6 skipped and 2 xfailed.

The new connection proof checks every identity factor, operand index, coefficient
index and direction for the default degree-four GF(8) factories and GF(3), GF(5),
GF(9) profiles. It proves the full operator as an exact product of identities
without materializing a 32768-square default matrix. Existing actual-factory
tests independently check all 65536 complex amplitudes with a spectator,
reversed bindings and source-SDK-blocked `circuit.append(op)` execution.
Approved Proposal 125 storage evidence covers unused raw encodings as well.

## Resources and remaining boundaries

Measured view streams contain zero primitives, zero private work and zero
measurements. Source QEC estimates separately contain zero gate categories;
source QubitCount equals the packed width (15 for the default examples).
Alias wiring is an atomic view, not an invented source decomposition.

The previously recorded nonbinary polynomial tensor-index discrepancy belongs
to `ClassicallyControlled._tensor_data`, not these view tensor factors.
`test_nonbinary_classical_field_feedback.py` retains its GF(3)/GF(5) witnesses:
polynomial integer encodings differ from packed bit indices.

Q-0533 remains open for symbolic field metadata in logical Program declarations
and remaining generic source bindings. A read-only six-profile probe finds valid
original logical identities but native descriptor rejection. Proposal 125 only
authorizes concrete physical storage; no symbolic admission is implemented here.
Original codec failures remain explicit. Numerical finite-field arithmetic and
control are still governed by their separate binary contracts.

## Evidence

Reference receipts: `g2-polynomial-views-original-test-intake-20261125.log`,
`g2-polynomial-views-raw-protocol-20261125.json`, and
`g2-polynomial-view-closure-regression-20261125.log` in the external reference
workspace. The concentrated regression covers the view owners, complete
polynomial audit, approved raw storage and original classical tensor witnesses.
See [the full polynomial audit](FIELD_POLYNOMIAL_FILE_AUDIT.md) and
[approved raw storage](NONBINARY_QUANTUM_FIELD_STORAGE.md).
