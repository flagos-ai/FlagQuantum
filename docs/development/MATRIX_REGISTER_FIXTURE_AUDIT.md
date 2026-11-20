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

## Actual typed graph matrix handoff, November 21

The complete original `matrix_gate.py` and its direct typed-graph consumers now
have automatic Ecosystem handoff coverage. This adds no new numerical kernel or
public contract. The earlier complete constructor, validation, RNG, tensor-axis,
adjoint, zero-width scalar and native append evidence remains accepted.

The new graph importer initially compared immutable declaration records using
ordinary equality. Array-valued constructor arguments made a legal original
MatrixGate fail with NumPy's ambiguous truth-value error. Both declaration
collision detection and live leaf matching now compare bytes produced by the
existing deterministic component-library serializer. This preserves exact array
dtype, shape and values without adding an array comparison vocabulary or loading
constructors named by serialized text. Unequal live objects with identical
serialized declarations still fail explicitly. Constructor-reference leaves
retain their existing explicit-factory boundary.

`test_matrix_graph_handoff.py` uses the unchanged original five seeded random
matrices at each width one, two and three. Each five-leaf source graph imports
to owned Program. Every imported matrix and reversed physical operand tuple
matches the source exactly. Every matrix column and a dense coherent state
entangled with an external spectator pass through `circuit.append(op)`. The
actual source graph adjoint restores the complete initial complex state. Output
ownership, no workspace events and exact serialized CircuitIR roundtrips pass.
Repeated equal matrices and distinct array-valued declarations select their
correct actual live leaves. Source objects remain confined to conversion.

Measured execution emits five opaque matrix instructions, zero measurements and
zero work allocations for each five-leaf graph. The original logical peak is
one, two or three according to the source word width, excluding the external
spectator. The source matrix atom has no FT decomposition or QEC gate estimate.
Matrix-atom counts and logical peaks therefore remain separate from any
fault-tolerant model. The earlier zero-width scalar source/native disposition
is reused without claiming a new zero-wire static SDK contract.

All three unchanged original test cases pass. Concentrated native regression
passes 217 cases, local integration passes 17, and strict typing passes both
changed Python files. Black, Ruff, Ecosystem scope, architecture, IR and protected
API checks pass. Q-0498 and Q-0499 retain their accepted status. This batch fixes
a shared adapter regression and adds direct graph-consumer evidence, with no
additional parent acceptance counted.

The Cirq 1.6.1 compatibility lane passes all 73 cases with 692 original SDK/NumPy
shape-deprecation warnings retained. Five frozen source blobs match the pinned
inventory. Dependency policy retains 666 existing repository findings and zero
changed-path findings. The complete evidence receipt is
`g2-matrix-graph-evidence.json` in the external reference evidence directory.
