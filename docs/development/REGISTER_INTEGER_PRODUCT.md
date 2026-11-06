# Unsigned integer product

The emitter binds the pinned Qualtran v0.7.0 Product register
signature. Inputs are big-endian unsigned words of independent
widths. The result has twice the wider input's width. Each
partial product contributes to column `i+j`, following
Appendix G. The same weighted-column emitter serves squares,
sums of squares and products. It copies the parity word into
the result by XOR, then clears all carries and partial products
with immediate measurement feedback. Applying the stream to a
matching result clears it. [Appendix G, Lemma 10](https://arxiv.org/pdf/2105.12767#page=81)

The pinned Product supplies the register layout and model.
Its decomposition and tensor methods raise and its classical
method returns NotImplemented. The executable value oracle
uses the unchanged PlusEqualProduct arithmetic on a zero
result and checks its inverse on the computed subspace.
Tests verify the ordered literal And stream for the partial
products, then certify every measurement history on all input
values through the tested 4×6 layout. The certificate checks
forward and reverse XOR with occupied result words. Raw
projection and sparse primitive trajectories verify complex
amplitudes, including a coherent spectator, and a native
example test uses `circuit.append(op)`.

| Input widths | Native T-family / CX / measurements / work wires | Pinned Toffoli model |
| --- | --- | ---: |
| 1×1 | 4 / 7 / 1 / 1 | 1 |
| 1×4 | 16 / 28 / 4 / 4 | 4 |
| 2×3 and 3×2 | 36 / 73 / 9 / 9 | 9 |
| 3×3 | 60 / 132 / 15 / 15 | 15 |
| 4×6 | 168 / 410 / 42 / 42 | 42 |

Native counts include cleanup in either direction. The pinned
formula counts abstract Toffolis. The native stream computes
one four-T And per partial product or carry and erases each
using measurement.
