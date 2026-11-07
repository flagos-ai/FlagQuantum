# In-place binary and unary conversion

G2-KLUT01 ports `gen_binary_to_unary` and `gen_unary_to_binary` from locked
Kickmix `src/kickmix/gen/lookup/gen_lookup.cc`, commit
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.

The compiler register module owns the source schedule. Target is a distinct
big-endian tuple of positive power-of-two length `L`; the reversed tuple is
the source low-first buffer. The last `log2(L)` target wires initially hold
the binary word. All other target wires are reset by the forward schedule.
The forward result is one-hot, with logical bit `value` set. The inverse
requires a valid one-hot superposition, reconstructs the binary word in the
same ports and clears every other target wire. `L=1` is the zero-bit logical
word, embedded in a one-qubit buffer. Empty and non-power-of-two buffers fail.

No quantum wires outside the supplied buffer are allocated. Inverse erasure
reuses a caller-owned classical result slot. Real swaps and the separate
virtual permutation of the inverse are preserved, as are source CSWAP/CX,
HMR/CZ and final conditional NEG. Inverse is a measurement-assisted channel
on the one-hot subspace, not the unitary adjoint on arbitrary buffer inputs.
Operations bind through the existing reversible-phase/KMX/Core adapter;
users continue to append an operation to a circuit. Source primitive counts,
expanded Core counts and analytical T models are distinct evidence.

## Local acceptance

The original source blob `d527eeeddf06c10ffe69ab6408594c60fd248182` was
verified against the pinned upstream Git object. An independent C++ harness
linked the unchanged native objects. All 56 complete primitive streams match
at binary widths 0 through 6 (buffer lengths 1 through 64), both directions,
two physical strides and two offsets. Only declarations and informational
tags are excluded. Portable fingerprints retain resets, operand order and
reused classical slots in `tests/fixtures/kickmix_binary_unary.json`.

All reset and HMR branches are enumerated separately through binary width
three, with entangled spectators and two layouts. Both conversion directions
match the full unnormalized complex vectors, including branch probabilities
and global phases. Full forward/inverse round trips are enumerated through
binary width two. Every inverse clears the nonbinary buffer ports. Larger
buffers have gate-stream evidence, not dense numerical evidence.

For an eight-wire buffer, forward primitives are 5 R, 3 SWAP, 1 X,
4 CSWAP and 7 CX; inverse primitives are 3 SWAP, 11 CX, 5 HMR, 4 CZ and
1 conditional NEG. Inverse binds to five Core measurement operations and
stays within the same eight-wire buffer. A zero count of literal T opcodes
at this unsynthesized IR layer is not a zero T-cost claim for CSWAP synthesis.

`python -m examples.algorithms.binary_unary_conversion` checks the coherent
forward result and round trip via `circuit.append(op)`. Source packages are
not required at runtime. The shared test projector now exposes reset Kraus
outcomes for this module and the previous unary cursor tests without changing
production execution semantics.

Acceptance: 30 new tests; 329 focused regressions passed with no skips.
Strict mypy (two production/example files), Black, Ruff, compiler scope,
architecture and diff checks passed. Native lookup/unlookup and the original
full fuzz/performance suite remain separate, open ledger items.
