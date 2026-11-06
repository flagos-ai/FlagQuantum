# Inverse-square-root initial estimate

The compiler component implements the untuned estimate in Eq. C1.
Given an unsigned big-endian source with f fractional bits, it finds
the first set bit. If that bit represents 2^j, set k=floor(-j/2) and
evaluate x₀ = 3·2^(k−1) − a·2^(3k−1). The circuit XORs the
truncation into a caller-owned big-endian result with b fractional bits.
A zero input yields a zero mask. The caller supplies clean quantum
workspace and reusable measurement slots. The source is preserved and
the workspace returns to zero. [Appendix C, Eq. C1](https://arxiv.org/pdf/1805.12445#page=8)

The result requires at least floor(f/2)+1 integer bits. Its fractional
width is explicit and may be zero. The internal guard width makes each
shift in both terms exact. For each leading bit, a one-hot control
loads the corresponding constant and subtracts the shifted source
suffix using the shared exact controlled-add adjoint. The source bits
preceding that suffix are zero on the selected branch. After copying
the requested high bits, the inverse adds the shifted suffix back,
clears the constant and reverses the first-bit scan.

For positive a, the chosen power satisfies 1/2 ≤ a·2^(2k) < 2.
Consequently the exact seed obeys 1/2 ≤ a·x₀² ≤ 1. Truncation at the
output precision can lower the seed further, including to zero at
coarse precision. Tests check the exact rational oracle and all
measurement histories for every input in six layouts. They also
exercise full coherent trajectories with an entangled spectator,
force all 16 raw measurement histories of the smallest layout, and
check the existing `circuit.append(op)` round trip.

| Source bits / source fraction / result fraction | T/T† | CX | CCX | Measurements | Work wires | Classical slots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / 0 / 0 | 16 | 35 | 2 | 4 | 6 | 2 |
| 2 / 2 / 1 | 64 | 143 | 4 | 16 | 14 | 6 |
| 4 / 2 / 3 | 208 | 465 | 8 | 52 | 24 | 11 |
| 5 / 0 / 7 | 304 | 682 | 10 | 76 | 26 | 12 |
| 7 / 4 / 5 | 568 | 1,272 | 14 | 142 | 36 | 17 |

The native counts include the output copy and complete erasure.
Each measurement has one reset. The paper's n-bit seed model gives
3n² + 5n Toffolis for its initial guess, or 182 at n=7, under its
fixed-width addition assumptions. Qualtran v0.7.0 gives no gate
decomposition or standalone cost for this seed. Its chemistry module
uses a distinct coefficient-based initialization. The tuned C2
constants and the complete iterative wrapper remain separate.
[Appendix C cost model](https://arxiv.org/pdf/1805.12445#page=8)
[Locked chemistry source](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/trotter/grid_ham/inverse_sqrt.py)
