# Toeplitz block encoding

The native program retains the source's four table coefficients and eight logical
bits in `data,s,j,dlt` order. Named qfuncs express the centered rotation table and
pre-update overflow marker. Hadamards around `angle*(data-0.5)` implement the
complete RX unitary with its absolute phase. Fourier phase arithmetic performs
the reversible modular update `j += s-1`.

The source captures signed `tmp=j+s-1` before updating `j`. The native program
XORs the flag before that update: underflow is `j+s<1`; overflow is `j+s>16`.
This preserves the original snapshot, including nonzero initial flags, while
avoiding hidden signed storage. It does not introduce general snapshot or drop
semantics.

The independent Circuit builds direct RX matrices and enumerates the signed
classical snapshot permutation. Acceptance compares the untouched source's full
state and every native unitary column for all 256 inputs, including nonzero data
and flags, negative underflow and positive overflow. Two further nonzero angle
bindings, coherent phase-weighted vectors, unitarity, absolute phase, bit order,
zero workspace and budget refusal are checked. Complete Notebook workflow
acceptance remains separate.

Current native costs: 8 physical qubits, 0 workspace, 5769 operations, depth 4691.
Original: 11 physical qubits, 151 operations, depth 128. This native path saves
three qubits while using substantially more operations. Efficient bounded
predicate and modular arithmetic lowering remains a concrete optimization gap;
no performance or optimality claim follows from semantic equivalence.
