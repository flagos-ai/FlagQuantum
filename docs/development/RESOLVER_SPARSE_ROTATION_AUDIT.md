# Sparse rotation preparation constructor dependency disposition

Reference: complete `state_preparation/sparse_state_preparation_via_rotations.py`
and its original tests at Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`. The source and test Git blobs are
`e5d631263c199129e23c246bfab0fd36fd5f774a` and
`ea6a6748f0837615f45e7e6082188b5cc7cfc1d9`. Reference fixtures require actual
Qualtran 0.7.0 and Kickmix 0.1 Simulator; no reference file is modified.

## Ownership and interface

Compiler preparation composes the four literal source stages: Partition,
normalized dense rotation preparation, inverse Partition, and ordered basis
permutation. Existing layout, dense rotation, lookup, phase arithmetic and
permutation owners retain their respective responsibilities. Compiler does not
import Algorithms or execute a quantum numerical kernel.

The target is an MSB-first unsigned THRU word; the phase gradient is an opaque
THRU word supplied by the caller. High target bits, including a zero-bit high
partition, remain in place. Dense preparation owns its private lookup and carry
work. Each permutation cycle owns its marker and equality scratch and releases
them before the next cycle. Logical target and gradient are not scratch.

The existing equality and ordered permutation owners now accept an exact
adjoint flag. The shared typed graph adjointer uses explicit child inverse
pairs, including equality's reversed Split/Join calls and self-inverse
controlled XOR. Reversing a cycle tuple would not preserve the original generic
Adjoint call chain. Measurement operations themselves are never reversed.

Constructor recipes retain support ordering, explicit target width and NumPy
coefficient precision through the original padding and normalization. Dense
table construction reuses the accepted compile-time RotationTree owner.
Source partial-map cycle discovery is retained instead of substituting the
separate Algorithms optimized injection. No protected public API changes.

## Source and execution evidence

Both complete original concrete examples and their exact Adjoints retain the
original ordered recursive calls, operands and types. Eight-bit angle-table
comparison additionally covers float32 and complex64 input coefficients in
both directions. Comparison explicitly stops at the existing source atomic
AddIntoPhaseGrad boundary; execution underneath uses native exact arithmetic.

Widths one, two and three cover the complete joint complex operator in both
directions, including empty permutation and zero-bit high partition. The
shared erasure-history certificate covers every applicable raw history, while
every basis column checks zero final private work. Equality and permutation
inverse operators are independently compared to the original source.
Ordinary `circuit.append(op)` and JSON roundtrips use the emitted instructions.
Source-free resolution is checked without importing the upstream constructor.
Previously accepted eight-bit coherent gradient/spectator/cleanup evidence for
the dense PRGA child and complete original sparse-state tests is reused; this
batch does not claim exhaustive dense operators for the larger examples.

## Retained boundaries

The unchanged source loses the requested transition 1 to 0 for support (3, 0),
returning only cycle (0, 3). Interchange preserves that actual operator and
checks it numerically. The independent Algorithms corrected injection remains
separate. Singleton source preparation cannot bind its missing dense target;
the constructor retains a logical declaration rather than fabricating a body.

The original depth-one ragged permutation codec fails, as does symbolic
from_n_coeffs serialization using HasLength. Symbolic phase width alone keeps
the valid parent graph and an unexecutable dense child declaration. Fully
symbolic sizes cannot be physically bound. These retained boundaries do not
raise the complete generic constructor count or close the parent source item.

## Resources and acceptance

Actual primitive inventories, physical leases and source analytical estimates
are recorded separately in `g2-sparse-rotation-constructor-resources-20261124.json`.
Both directions of the original four-bit target emit 176 T-family, 322 CX,
six Rx and 44 measurements with 11 peak wires. Its source model is 40 And,
194 Clifford, 40 measurements and 12 logical peak wires. The six-bit target
emits 256 T-family, 442 CX, six Rx and 64 measurements with 14 peak wires;
its source model is 60 And, 294 Clifford, 60 measurements and 14 peak wires.

The two-bit target with eight-bit angles emits 80 T-family, 230 CX, two Rx
and 20 measurements with 26 peak wires. Its separate source model is 12
Toffoli, six And, 42 Clifford, six measurements and 20 logical peak wires.
All six rows return zero live private leases. Lease accounting complements,
and does not replace, the numerical small-domain cleanup evidence.
The source phase-add leaf is atomic; its model and native carry expansion are
not equal primitive streams. Source QubitCount is a separate logical model,
not the actual first-use physical allocation peak.

Acceptance receipts use `g2-sparse-rotation-constructor-*-20261124` under the
external reference ledger. They contain the concentrated regression, local
integration, complete unchanged original test run and strict quality checks.
The concentrated regression passes 160 cases without skips; local integration
passes 145. The complete unchanged original file passes nine cases, with two
original skips and two original xfails. Strict typing covers all five changed
Python files. Black, Ruff, Compiler scope, language, architecture, IR and API
checks pass. Dependency policy retains exactly the existing 720 findings, with
no additional findings from this batch.
No large-example all-history execution, symbolic physical synthesis, source
codec repair, GPU result or performance claim is implied by these receipts.
