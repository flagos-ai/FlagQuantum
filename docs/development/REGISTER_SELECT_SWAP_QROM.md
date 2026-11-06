# Select-swap QROM with clean and borrowed banks

The register emitter ports the locked Qualtran v0.7.0 SelectSwapQROM
component graph. Addresses and words are big-endian. The high address
selects a zero-padded row-major block and the low address routes one
word in each bank to cell zero. The emitter XORs those words into
caller-owned destinations, which may start nonzero. An all-ones
control pattern enables the table load. The borrowed-bank correction
also cancels the XOR on inactive branches.

The clean route loads the batch, routes the selected words, copies them
by XOR, unroutes and reverses the load. The borrowed route appends a
second route/XOR/unroute to cancel the original bank contribution.
A unit-size block uses the same sequence with empty routing. The
relative-phase exchanges cancel around each XOR because their phases
depend on the bank and address, which the XOR preserves. Measured And
scratch starts and ends at zero. The borrowed bank can start nonzero
and entangled, and is restored with its phases and correlations.

The original gate oracle checks allocation order, partition aliases,
QROM and SwapWithZero components, XORs, frees and classical feedback.
The pinned source uses dirty Allocate and default Free flags for its
borrowed bank. The explicit FlagQuantum bank binding restores the input
state, as verified by complete complex output vectors. Fully batched
and mixed omitted high or low selector layouts fail in the source
graph and are rejected by the shared table validation.

The concrete default block chooser follows the source function body.
It doubles the target cost weight for the clean route and uses the
original scalar dataset factor. The source docstring describes the
opposite direction for the clean/dirty factor. The source function's
result is checked independently for both modes and multiple datasets.

The tests cover both control values, both zero and nonzero destinations,
valid addresses and padded cells, every borrowed bank basis value
for banks up to four wires, and representative larger bank values.
Complex vectors include a coherent spectator. Small cases force every
raw measurement history. Larger cases compare complete sparse
trajectories and use an exact And-channel certificate for both loads.

| Shape, block logs, mode | Physical wires | Bank wires | And work | Literal T-family | Literal CX | Literal measurements | Source model T / measurements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (3), (0), clean | 7 | 1 | 2 | 16 | 33 | 4 | 16 / 4 |
| (3), (0), borrowed | 7 | 1 | 2 | 16 | 34 | 4 | 16 / 4 |
| (5), (1), clean | 12 | 4 | 2 | 32 | 66 | 4 | 32 / 4 |
| (5), (1), borrowed | 12 | 4 | 2 | 48 | 90 | 4 | 48 / 4 |
| (3,4), (1,1), clean | 17 | 8 | 2 | 72 | 134 | 6 | 72 / 6 |
| (3,4), (1,1), borrowed | 17 | 8 | 2 | 120 | 202 | 6 | 120 / 6 |

The source peak-qubit estimates equal the physical wire counts in the
tested layouts. The source model aggregates Clifford gates, so the CX
column records literal native instructions only.

Run `python -m examples.algorithms.register_select_swap_qrom` to verify
a controlled lookup and borrowed bank restoration through
`circuit.append(op)`.
