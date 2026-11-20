# Monomial matrix validation evidence

Integration decision `IR_MONOMIAL_UNITARY_VALIDATION.md` precedes the Core
implementation in commit 62d24a2ae. This is a private validation algorithm change
under the existing explicit matrix contract.

All shape, built-in complex, finite-value, tolerance and diagnostic checks retain
the original behavior. Exactly one nonzero in every row permits the occupied
column and squared-norm proof. Every other input uses the original complete
column inner products. All rows are inspected before choosing the proof.

The independent test oracle retains the previous inner-product implementation.
It agrees on all 6,144 four-dimensional phased permutations, repeated columns,
coefficient perturbations on either side of the tolerance, seeded dense and QR
unitary matrices at dimensions 2, 4 and 8, zero rows and a mixed matrix whose
first rows are monomial. Existing immutable payload, tamper detection, diagnostic
and lowering tests run with these cases. No payload or physical stream changes.

The actual controlled ECWindowAddR consumer previously exceeded 30 seconds while
validating its controlled QROAM matrix. The optimized path completes that node
and reaches the next ECAdd step in 1.70 seconds total. The next step raises a
recursion error. A separate SDK-only decomposition reproduces an undecomposable
Controlled(And) leaf in that stage; its receipt is
`g2-ec-step-one-original-boundary-20261121.json`. The native recursive path
and original undecomposable-leaf error remain distinct observations. This intake proves the validation bottleneck is removed for the
recorded small consumer. Complete ECC execution remains open. The result retains
the original dense tensor fallback, so it supplies no scalable provider claim or
primitive-level QROAM parity. Native QROAM primitive evidence remains separate.

Evidence is in `g2-monomial-matrix-focused-20261121.log` and
`g2-monomial-matrix-ecc-followup-20261121.json` in the reference directory. The
baseline timed intake retains its own receipt. Validation emits no quantum
operations, measurements, workspace or independent QEC resource model.
