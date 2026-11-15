# Trigonometric source family whole-file audit

Q-0223–Q-0228 cover the entire arithmetic trigonometric package at
Qualtran v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389`. The
six original Git blobs match the frozen inventory.

## Ownership, word contracts and source disposition

| Original definition | Native ownership and tested scope |
|---|---|
| Package initializer | Its only statements re-export ArcSin and ArcTan. ArcSin routes to Simulation for scalar values and Compiler fixed point for Appendix D lowering and resource declarations. ArcTan routes to Compiler lookup for exact bounded words, fixed point for the independent arithmetic approximation, and the shared trigonometric resource model for the original estimate. These routes use existing package boundaries. Foreign class exports remain an interface difference of the implementation files. |
| ArcSin fields, validation and signature | The complete ten-bit factory signature retains THRU x and result, both unsigned fixed point with seven fractional bits. The native neutral dtype validates fixed-point shape. The example declares the original iteration and degree defaults through its model. Generic foreign constructor admission remains open. |
| ArcSin.on_classical_vals | Simulation preserves the scalar full-width fractional interpretation, binary64 domain check, truncation and occupied-result XOR. The declared fractional width is absent from this source calculation. Word 15 at width four produces 19 before output encoding. The native scalar result remains unmasked. Verified all valid words at widths 1/2/3/4/5/8/10 under several declared splits. |
| ArcSin.build_call_graph, factory and documentation | Compiler retains the independent Appendix D Toffoli equation, checked over 36 concrete parameter combinations and unrestricted symbolic expressions. The complete source instance declares 13425 Toffoli. No source primitive decomposition exists for it. |
| ArcTan signature, registers and with_registers | The example declaration retains the opaque select and target and QBit sign, including symbolic widths. The original Cirq register tuple and symbolic/rebinding failure boundaries are recorded in the tests. Native execution binds wires explicitly. |
| ArcTan.apply and self-adjoint | The existing bounded QROM preserves big-endian unsigned select, sign XOR and fractional target XOR. All six original width pairs now compare the complete supported input domain at zero and occupied outputs to the locked callback. Certified primitive branches verify the outputs, zero work and two applications restoring the input. Existing three smaller source-derived QROM cases retain direct gate, operand, feedback and complex-state evidence. |
| ArcTan.build_call_graph | Retain one PlusEqualProduct with operand widths n/n/2n and fresh source multiplicity represented by the original single-query symbol. The symbolic legacy estimate is 8 c n². The original has no corresponding product gate decomposition. |

The independent Appendix D ArcSin circuit accepts a big-endian
nonnegative input in [0, 1] with one integer bit and produces a separate
one-integer-bit output. It uses a quantized factor polynomial and
explicit half-pi, XORs its result into a bound target, and restores clean
scratch. The signed wrapper, fixed-point reduction, inverse square root,
Horner and measurement feedback evidence are reused. Coefficients are
independently reproduced with a Chebyshev projection. The original
minimax coefficient table is still missing. The original scalar callback
and its overflow witness establish a different word contract from this
executable approximation.

The independent ArcTan arithmetic path retains its existing reciprocal
reduction and Horner evidence. Four of 256 input words in the previous
eight-bit comparison differ from the original by +1 ulp. The bounded
QROM path supplies exact locked-word parity within its admitted range.

## Original test and resource accounts

Four original test functions execute, including all original slow
ArcTan cases. The frozen run reports 10 passed and 3 skipped. The skips
are ArcSin decomposition, count comparison without a decomposition,
and dtype checks requiring decomposition. No reference file was edited.

| Path | Recorded account |
|---|---|
| Original ten-bit ArcSin factory | 13425 Toffoli in the declared model |
| Independent Appendix D teaching circuit | 6856 T/TDG, 16918 CX, 88 CCX, 1714 measurements, 196 work wires and 13 reusable slots |
| Original symbolic ArcTan | 8 c n² T in the product estimate |
| Native three-bit-to-three-bit ArcTan QROM | 16 T/TDG, 43 CX, 4 measurements, 2 work wires and 2 reusable slots |

The Appendix D teaching circuit is emitted through `circuit.append(op)`
and reported as IR. Its branch and coherent sparse execution evidence
comes from the dedicated native tests. The notebook independently
simulates the complete complex ArcTan input/occupied-target state and
its restoration through append. Emitted primitive counts, source
declared costs and polynomial factor error are distinct accounts.

## Teaching asset and closure

The original ten-cell notebook covers ArcSin only. The native
fourteen-cell notebook preserves all original topics, signature and
graph, and adds the scalar/Appendix D distinction and an ArcTan
resource/lookup example. Both assets pass independent kernels. The
native kernel blocks foreign reference imports.

Q-0223 closes through the complete two-export responsibility map.
Q-0228 closes as a complete teaching asset. Q-0224–Q-0227 retain the
original decomposition, generic interface, approximation and original
test-mode boundaries above.
