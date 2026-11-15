# First-quantization SELECT leaf audit

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, under
`qualtran/bloqs/chemistry/pbc/first_quantization`.

| Inventory | Source file | Git blob |
| --- | --- | --- |
| Q-0390 | `select_t.py` | `ca7938daa3a19126a2c6b0a4749038249f384ba2` |
| Q-0391 | `select_t_test.py` | `58dec6f7effb1464a34b1cce8d3f391913af44ee` |
| Q-0392 | `select_uv.py` | `86bd0fdb40cdc5bf0ab1ad160198dd7f73e0305b` |
| Q-0393 | `select_uv_test.py` | `cc689cb207554e6e35b4853b440cf58c1d7d2941` |

The algorithm owns the SELECT T, SELECT UV, and nuclear-phase register
contracts. The models and tutorial reuse their order, dtypes, and axis
shapes. The kinetic words `r` and `s` stay opaque at one bit. UV retains
the empty nuclear index for one atom. The phase child uses `Rl` and a
three-axis `nu` bank one bit narrower per axis than the parent.

The six-bit kinetic fixture declares 27 Toffoli and peaks at 35 logical
qubits. The six-bit, ten-atom, eight-bit nuclear UV fixture declares 261
Toffoli, 36 And, 234 Clifford, and 36 measurements, with a peak of 71
logical qubits. Its original test combines Toffoli and And and accounts
for the analytical convention for six additions separately. Both
nuclear-phase formula branches and the case where the phase child
dominates the parent peak are covered. A one-bit momentum word retains
the original declared gate categories, while the source peak query
fails on the signed conversion dtype.

The locked leaves supply no complete executable decompositions. These
gate counts are model categories; there is no emitted gate, state, or
workspace measurement for the leaves. The shared executable momentum
exchange belongs to the top-level batch and keeps its separate evidence.

The unchanged tests report six passes and six skips. The native tutorial
and 193 concentrated regression cases pass, with independent type,
format, ownership, architecture, IR, and public API checks. Q-0390
through Q-0393 close the original declarations and runnable test scope.
