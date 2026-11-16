# Hamiltonian simulation by GQSP

The native Algorithms owner binds a coherent walk emitter, its named logical
ports, and a prepared signal. The bound operation runs through
`Circuit.append`. The caller supplies clean outer PREPARE junk and walk
workspace. Outer junk stays live across GQSP. Walk workspace is restored
after each walk invocation and can be reused between calls.

The unchanged Qualtran v0.7.0 source at
`bcb17bb25360ed68ce3bc084645db3e929799389` declares Allocate, Always(PREPARE),
GQSP, Always(PREPARE†), and Free. FlagQuantum maps allocation and release to
caller-provided slots. The Jacobi–Anger degree uses signed scaled time and
half the specified precision. The cosine coefficients use negative scaled
time, grid normalization, and the source's secondary factor
`1-2*precision`. GQSP uses the existing complement and phase recursion at
half precision with the source's `1e-4` verification tolerance.

## Evidence

For a one-bit selection and a one-bit target, the native path binds the
existing walk owner on permuted physical wires. The test checks source
polynomial coefficients, the complementary polynomial, each SU2 angle
and its placement around the walk, and all ordered walk control operands.
It verifies both a positive-controlled inverse walk and a zero-controlled
forward walk. The test compares all eight complex columns to the
unmodified source and checks the projected block against `exp(-iHt)`.
Outer and reflection junk return to zero across all columns. This fixture
is coherent and has no measurement branches.

| Time | Precision | Degree | Declared walk calls by polarity | Declared SU2 calls | Emitted primitive count |
| --- | --- | --- | --- | --- | --- |
| 0.75 | 1e-5 | 7 | 7 each | 15 | 249 |
| -0.75 | 1e-7 | 9 | 9 each | 19 | 317 |

The native primitive totals include both outer PREPAREs, decomposed absolute
signal phase and rotations, and physical walk reflection gates. The source
graph also declares one forward and one inverse PREPARE. Both concrete and
symbolic Hubbard factories agree on all five direct declared costs.

The unchanged upstream non-slow test suite completed with 35 passes,
3 expected failures, and 4 slow cases deselected. The locked 12-cell
notebook executed all seven code cells in an independent kernel.

## Outstanding file evidence

Q-0539 needs the native 12-cell notebook, including the concrete Hubbard
signature and the symbolic depth-two cost table. Q-0540 needs the native
Hubbard end-to-end binding and its measured PREPARE/SELECT dependencies.
Q-0541 needs the native random-unitary and Hamiltonian test matrices,
Hubbard controlled-resource coverage, and the slow reference cases. The
full matrix test here covers the small bound fixture and two parameter
pairs. It uses the existing controlled walk and SU2 emitters for native
primitive lowering.

## Native notebook and complete source suite

Q-0539 closes with the native 12-cell notebook. It contains both locked
2-by-2 Hubbard examples, the concrete and symbolic signatures, the
concrete one-layer resource graph, and the symbolic depth-two leaf
table. The native helper uses the existing Hubbard signature and
PREPARE cost authority. The preserved source takes a set union
of the selection registers, so the native display uses deterministic
Hubbard order and the comparison checks every named port, its width,
side, and bounded-integer range. Signal and target remain at the ends.

For both examples, the reference test checks every edge and leaf at
both depths. The expansion includes controlled inverse SELECT and
reflection, separate controlled Ry directions and PREPARE inverses,
CSwap widths, and rotation-angle-generalized SU2 children. The concrete
example has degree 127 and 255 SU2 calls. The independent native kernel
ran all seven code cells and rendered two SVGs.

The four slow source cases also passed. Together with the earlier
non-slow run, the original suite has 39 passes and three expected
failures. Q-0540 and Q-0541 retain the native Hubbard measured append
path, projected execution evidence, and full native randomized matrix
coverage.
