# Split-coordinate periodic Laplacian

The source splits its three-bit position `j` into low one-bit `j0` and high
two-bit `j1`. The native program declares those coordinates directly. Readouts
`j1,j0,selector,dimension` preserve the original packed `j,l,k` wire order.

Each coordinate uses whole-register QFT, a coherent conditional modular-shift
phase and adjoint QFT. The low selector bit enables subtraction when zero; the
high bit enables addition when one. The dimension bit selects the coordinate.
The original selector sign preparation and final Hadamard interference remain
explicit high-level operations. There is no handwritten Fourier gate sequence,
new view API, or claim that arbitrary subregister QFT is supported.

Tests compare all 64 packed basis initializations and a phase-weighted coherent
state with an independent permutation Circuit. They include both wraparound
boundaries, every selector and dimension, global phase, exact readout order and
zero hidden workspace. Resource limits fail closed.

Current native costs: 6 physical qubits, 0 workspace, 62 operations, depth 41.
Untouched source costs: 7 physical qubits, 21 operations, depth 17. This explicitly
trades more gates for one fewer physical qubit. The quantum program is accepted
separately from the related complete Notebook workflow.
