# Unsigned product accumulation

The emitter implements the concrete Qualtran v0.7.0
PlusEqualProduct arithmetic. It adds or subtracts an unsigned
product modulo the accumulator width, preserving both inputs.
The accumulator must have at least the sum of the input widths
and can start in any basis value or superposition.

The shared column compressor computes the partial products and
carries once. The native Add or Add adjoint consumes the live
parity wires directly and restores them. The column network
then erases the products and carries with measured feedback.
This uses no separate product output word. The add carry slots
and the product workspace finish clean.

The pinned PlusEqualProduct and its adjoint expose the full
arithmetic and tensors. Their gate decomposition raises
DecomposeNotImplementedError. Tests compare the source Add
component's literal gates and feedback, the separately verified
product prefix and suffix, and both complete arithmetic maps.
All accumulator values, including overflow and underflow, are
checked in six layouts with every measurement history certified.
The one-bit-input case projects every raw history. Larger small
cases also compare sparse primitive trajectories with the
source tensors and a coherent spectator. A native
`circuit.append(op)` round trip covers measured slot reuse.

| Input widths / accumulator width | Native T-family / CX / measurements / work wires | Pinned approximate T count |
| --- | --- | ---: |
| 1×1 / 2 | 8 / 14 / 2 / 2 | 8 |
| 1×2 / 3 | 16 / 32 / 4 / 4 | 32 |
| 2×2 / 4 | 36 / 73 / 9 / 9 | 32 |
| 2×3 / 5 | 52 / 113 / 13 / 13 | 72 |
| 3×2 / 6 | 56 / 124 / 14 / 14 | 72 |

Native counts are per direction and include both cleanup stages.
The source estimate is `8*max(input_widths)**2` and does not
account for the accumulator width.
