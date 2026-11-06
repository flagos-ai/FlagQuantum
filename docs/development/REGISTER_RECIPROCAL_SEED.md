# Fixed-point reciprocal seed

The circuit prepares the initial reciprocal estimate in Figure 4.
For an input with a nonzero integer part of bit length `p`,
the output represents `2**(-p)`. Both registers are big
endian. The result is a pure fraction with at least one bit
for every input integer position. A zero integer part gives a
zero seed as a native extension to inputs below one.
[Reciprocal initialization, Figure 4](https://arxiv.org/pdf/1511.08253#page=10)

The input's highest set integer bit selects a seed bit through
the figure's negative-flag-controlled Toffolis and CNOT updates.
An all-zero-controlled X on the seed, followed by X on the
flag, clears the flag. The adjoint reverses the logical gates.
The all-zero control reuses the pinned MultiControlX emitter
with measured scratch erasure in both directions. All temporary
work returns to zero.

Tests compare the figure gate sequence and the unchanged
Qualtran MultiControlX gate stream. They verify all valid basis
words and measurement histories by exact primitive block
certificates, including the zero-integer extension. Small
raw projection tests and larger sparse trajectories check full
complex amplitudes with a coherent spectator. A two-integer-bit
round trip runs through `circuit.append(op)`.

| Integer bits | Native CCX / T-family / CX / measurements / work wires |
| ---: | --- |
| 1 | 0 / 0 / 3 / 0 / 1 |
| 2 | 1 / 4 / 10 / 1 / 2 |
| 3 | 2 / 8 / 17 / 2 / 3 |
| 4 | 3 / 12 / 24 / 3 / 4 |
| 8 | 7 / 28 / 52 / 7 / 8 |

Counts are per direction. CCX is counted as a native primitive
and is separate from the emitted T-family gates. Work includes
the flag. For `m` integer bits, the pinned initial estimate
model lists `m-1` Toffolis, `m+1` CNOTs, one X and one
MultiControlX. The original complete call graph raises for
total input widths one and two because it constructs
SquareRealNumber with insufficient width. This seed is
independent of the later Newton steps and their precision.
