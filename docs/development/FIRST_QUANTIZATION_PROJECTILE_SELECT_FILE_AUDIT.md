# Projectile first-quantization SELECT leaf audit

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, under
`qualtran/bloqs/chemistry/pbc/first_quantization/projectile`.

| Inventory | Source file | Git blob |
| --- | --- | --- |
| Q-0384 | `select_t.py` | `efa2e943f706cf62b9af7a622b9ce2bf671c8327` |
| Q-0385 | `select_t_test.py` | `f0752fda4113d5b233e72aeb53baa668ec2bfb24` |
| Q-0386 | `select_uv.py` | `6ef6283cfb5e8fba459e6dc8c1214020d35e0d17` |
| Q-0387 | `select_uv_test.py` | `8856e756a80b972c398cc2285615d628442fa653` |

## Ordered ports and calls

SELECT T has three single-bit controls, separate three-bit `w` and
`w_mean` words, and three projectile axes in its `p` bank. Its
declaration adds the source mean-momentum selection Toffoli.
The original eight-bit fixture declares 38 Toffoli and peaks at
49 logical qubits.

SELECT UV has through flags, an independent nuclear index `l`,
nuclear coordinates `rl`, and three separate axis banks. Each axis
of `p` uses the projectile width, `q` uses the electron width, and
`nu` uses projectile width plus one. A single atom leaves an explicit
zero-width opaque index. Independent electronic and projectile
conversion and addition counts come from the existing compiler cost
models. Equal-width children combine in the source graph query. The
nuclear phase uses projectile width.

For the original 6/9-bit, 10-atom, 18-bit nuclear fixture, the UV
model gives 828 Toffoli, 45 And operations, and a peak of 99 logical
qubits. The source test combines the first two categories as 873
Toffoli equivalents and adjusts its analytical expression for the
different additions.

## One-bit boundary and verification

The locked one-bit signed conversion declares a negative Toffoli
contribution but fails when the qubit-count query constructs its
dtype. The native gate-count entry point uses the same declared
children while the peak-bearing model retains the source range
restriction. Tests cover both positions for a one-bit word, including
the case where both words are one bit.

Neither source SELECT leaf supplies a composite or executable
decomposition. No emitted primitive count, measurement branch
comparison, or numerical state comparison is attributed to these
leaves. The separate momentum exchange has its own executable
`circuit.append(op)` contract and gate evidence.

The original suite had six passes and six skips. Q-0384 through
Q-0387 close the original declarations and runnable test scope.
