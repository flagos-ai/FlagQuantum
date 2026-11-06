# Scale an integer by an unsigned fixed-point fraction

The native emitter binds the pinned Qualtran v0.7.0
ScaleIntByReal signature. The input fraction has `r`
fraction bits, the unsigned integer has `n` bits and the
`r`-bit result has `r-n` fraction bits. The supported layout
has `1 <= n <= r`.

The circuit starts with the integer's least significant bit
and computes the controlled copy into a zero result. Each
remaining integer bit controls an addition of a longer
prefix of the input fraction. Each term is truncated before
addition. An addition uses its retained width plus one
carry position because the sum of all lower-weight terms
fits in that suffix. The adjoint reverses the controlled
additions and erases the initial controlled copy by
measurement. [Appendix D4, equations D6 and D8](https://arxiv.org/pdf/2007.07391#page=70)

For input words 15 and 3 at widths 4 and 2, term truncation
gives output word 10. Truncating the full product gives 11.
Tests independently evaluate the term formula and bound
the error by the number of selected terms times the result's
least significant unit. The locked ScaleIntByReal provides
its signature and cost model. The gate and tensor methods
raise and the classical method returns NotImplemented.

Tests compare literal CAdd and CAdd adjoint gates with
their pinned child decompositions and verify every history
with exact And and erase certificates on all inputs of the
tested layouts. Small cases project every raw history.
The 4×2 case compares sparse primitive trajectories with
a coherent spectator. The local trajectory executor also
runs a forward and erase round trip via `circuit.append(op)`.

| Fraction / integer widths | Native forward T-family / CX / measurements | Native erase T-family / CX / measurements | Work wires | Pinned Toffoli model |
| --- | --- | --- | ---: | ---: |
| 4 / 1 | 12 / 18 / 0 | 0 / 0 / 3 | 0 | 3 |
| 4 / 2 | 32 / 62 / 6 | 24 / 50 / 8 | 6 | 8 |
| 6 / 4 | 104 / 216 / 24 | 96 / 204 / 26 | 10 | 26 |
| 12 / 4 | 272 / 576 / 60 | 240 / 528 / 68 | 22 | 68 |

The model counts abstract Toffolis for the paper's forward
schedule. The native erase measures the initial controlled
copy and therefore uses fewer T gates and more measurements.
