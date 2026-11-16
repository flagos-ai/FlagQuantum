# Matrix-backed register fixture

Q-0498 and Q-0499 match their inventory blobs at Qualtran v0.7.0
commit `bcb17bb25360ed68ce3bc084645db3e929799389`. All three
original test cases passed unchanged. The native fixture is test-only
and uses the existing matrix-backed Circuit instruction.

The fixture freezes the matrix rows and verifies both products
with the adjoint at the requested tolerance. It reproduces the
reference Gaussian/QR sampler and diagonal phase normalization.
All 15 seeded matrices from the original test are identical entry
by entry, as is the next random draw. Validation also checks an
integer seed, shape and unitarity failures, and immutable input rows.

The raw tensor keeps all output bit axes before all input bit axes.
The append tests bind full matrices to reversed physical operands
and compare the complete coherent state against the original tensor.
The inverse matrix recovers the original state. At zero logical
width the scalar multiplies the identity on an existing ambient wire.
The source scalar and its inverse preserve the full complex state
of the entangled ambient circuit up to the specified phase.

Each tested positive-width application emits a single opaque matrix
instruction across one, two or three wires. There are no measurement
branches or work wires. The source declares the same logical width
but has no fault-tolerant decomposition or QEC count for an arbitrary
matrix. A stored matrix instruction is left opaque in the evidence.

The concentrated native suite passed 505 tests. Strict typing,
formatting, team scope, architecture, IR and public API checks passed.
