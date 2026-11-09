# Standard-gate Notebook workflow

The reviewed Notebook contains nine independent quantum-function definitions
and synthesis calls, in this order: X, RZ, R, PHASE, RZZ, CX, CRX, SWAP and U.
The overview reuses the corresponding native FlagQuantum function from each gate's
example directory and synthesizes all nine. Native synthesis directly accepts the
quantum function, so no separate create-model string representation is needed.
It additionally executes each program and prints resources and named probabilities.
No circuit drawing, remote execution, training or classical data workflow occurs
in the source Notebook.

The source parameters are preserved, including `R(1, 2, q)` and
`U(1, 2, 1.5, 1.1, q)`. The RZZ counterpart uses a named two-bit QNum for the source's
whole-register QArray; no element indexing occurs. All other declared outputs
retain their names and order.

Every gate has an independent Circuit comparison for its complete complex state,
including absolute phase and zero workspace. The workflow acceptance executes
all nine syntheses and checks their readout order and distributions. A zero-state
controlled gate can leave probabilities unchanged; the separate coherent-input
and control regression tests protect its nontrivial action. Gate semantics are
not inferred from probability changes alone.

The [provenance record](provenance.json) binds every reviewed Notebook code cell.
