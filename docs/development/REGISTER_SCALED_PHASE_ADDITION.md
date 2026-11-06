# Scaled fixed-point phase-register addition

The shared emitter ports the concrete Qualtran v0.7.0
AddScaledValIntoPhaseReg schedule. The input and coefficient magnitude
have explicit unsigned fixed-point formats. The magnitude is truncated
and saturated to that format. The sign of the coefficient determines
the sign of each addition.

The scheduler visits set coefficient bits from most to least
significant. It binds the original input slice and right shift for
each retained term. Each term truncates at phase-register precision
before accumulation. This preserves the original repeated-add result,
including cases where rounding the full product would differ.
The adjoint visits the same slices in reverse and inverts each
addition. The stage workspace is shared at the maximum of the leaf
requirements, and every work wire returns to zero.

The source oracle compares the original Cirq generator's leaf
operands, widths, shifts, signs and order. It also checks the Soquet
graph where that graph binds. The pinned source rejects some
full-width slices of a mixed integer/fraction input in its Soquet
export because the leaf returns a purely fractional dtype. The
generator still provides the slice, and the native output is
checked against both that leaf sequence and the original high-level
numerical action. The executable leaf uses the accepted measured
phase addition, and its emitted instructions are compared with the
pinned Add and Add-adjoint gates.

Tests cover positive and negative coefficients, saturation, zero,
integer and fractional input slices, and truncation. They compare
complete complex vectors with a coherent spectator and both
directions of the map. Exact And certificates cover all measurement
histories on the tested basis support, with direct raw projections in
small layouts and sparse trajectories for wider layouts.

| Input format, gradient, coefficient | Source generator terms | Physical wires | Clean work | Literal T-family | Literal CX | Literal measurements | Source model Toffoli / qubits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (3,3), 4, +0.75 | 2 | 10 | 3 | 24 | 61 | 6 | 4 / 7 |
| (5,2), 3, +1.875 | 4 | 10 | 2 | 28 | 72 | 7 | 3 / 8 |
| (8,8), 8, -0.375 | 2 | 23 | 7 | 56 | 151 | 14 | 12 / 16 |

Formats are (width, fraction bits). The pinned source model caps the
number of scheduled terms and has no measured leaf implementation.
In the middle case it models three additions despite generating four.
The literal columns count the expanded native instruction stream.

Run `python -m examples.algorithms.register_scaled_phase_addition`
to exercise scaled phase kickback through `circuit.append(op)`.
