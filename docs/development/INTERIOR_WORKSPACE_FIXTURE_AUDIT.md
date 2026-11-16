# Interior workspace fixture

The Q-0493 implementation and Q-0494 test match their frozen blobs at
Qualtran v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389`.
The original test passed unchanged.

The native fixture composes the existing register builder, atomic SWAP
and workspace binder. The call sequence is Allocate, Swap, Swap, Free.
The independent source workspace reader binds the source allocation
and release to the same caller-owned slots. Widths 1, 2, 3 and 10 match
the source gate operands, ordered outputs and both lifetime events. The
symbolic shape retains two `n`-bit logical words and a `3n`-qubit peak,
including `n` work wires. The source model counts `2n` Clifford calls.
The native physical stream has `2n` SWAP instructions, zero T-family
instructions and zero measurements.

For widths 1 through 3, `circuit.append(op)` maps a coherent
superposition of all x values on the y=0 subspace to x=0 and y=old x,
preserving the full source complex vector. Every workspace bit is zero
at release.

The original classical callback rejects nonzero y at Free. In the
native physical stream the same input remains in the middle word,
because the binder tracks the lifetime and does not inspect the
quantum state at release. That input is outside the fixture's clean
ancilla contract. The difference is recorded by a separate test and
keeps the implementation row Q-0493 open. Q-0494 is covered by
the direct call, gate, ownership and resource comparisons.

The workspace regression passed 183 tests. The shared source workspace
reader's 144 serial shaped tests completed before the combined consumer
run timed out. The cross-pool consumer suite passed all 24 tests in an
isolated rerun. Strict typing, formatting, team scope, architecture, IR
and public API checks passed.
