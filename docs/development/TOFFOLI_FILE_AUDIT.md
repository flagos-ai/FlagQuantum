# Toffoli source family

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The source files retain their
Apache-2.0 attribution.

| Inventory | Frozen Git blob | Disposition |
|---|---|---|
| Q-0265 `toffoli.ipynb` | `dc66c9de450ddd86367e983e3342bc5515b44768` | Closed by the complete native tutorial. |
| Q-0266 `toffoli.py` | `a007be2ef25aa7dc5fda4d793498041a8f6b5a48` | Native atomic and grouped control paths verified. Generic external control-object and drawing protocols remain open. |
| Q-0267 `toffoli_test.py` | `ff83e793112a8c7503ec526715e76647ce6f2196` | Original suite runs unchanged. Generic source factory and diagram harness equivalence remain open. |

## Signatures and control binding

The atomic operation has the through-going `ctrl` QBit array of shape `(2,)`
and the scalar through-going `target` QBit. Native physical order is the two
controls followed by the target. It is self-adjoint. The one-level graph has
one atomic leaf and zero edges. Existing primitive, matrix and classical
tests verify both control bits, retained ports, and the target flip.

The original `get_ctrl_system` puts the added groups before the original
positive control pair. The native tutorial renders both one-bit polarities
and the signed eight-bit comparison against 6. It then binds the original
unsigned eight-bit prepared example to the shared control-register binder and
the same Toffoli emitter. The original and native order is the eight MSB-first
word bits, the two original controls, and the target. The new source-graph
tests compare both signed and unsigned declarations through the real source
adder, retaining the dependency order, exact angles, feedback slots and
output bindings. The small-control regression already compares every tested
forward and inverse history and full complex spectator states.

The eight-bit tutorial uses one actual trajectory of the prepared input. It
checks the complete complex output and the nine zeroed work slots before
applying the matching integer effect. This large example demonstrates one
bound input with the word held at 6, while the existing small-input tests
provide the exhaustive history checks.

## Emission and resource accounting

| Path | Emission | Declared source model |
|---|---|---|
| Atomic Toffoli | One `ccx`, no work or measurements | One Toffoli, four T-equivalents under the default convention |
| Eight added control bits | 24 X, 27 H, 18 T, 18 Tdg, 55 CX, 9 S, 9 measurements, 9 CZ and 9 reset, with nine work wires | 9 ANDs, 34 Cliffords and 9 measurements, peak 20 for the controlled body |

The native layout replaces source drawing and external adder objects with
explicit ordered register bindings. Factory metadata, foreign diagram
callbacks and generic control object equivalence remain open. The native
tutorial retains all 18 original cell topics and adds two cells for the
atomic matrix and exact inverse through `circuit.append(op)`.

Both notebooks passed independent kernel execution. The native kernel
rejected Qualtran, Quimb and Cirq imports. The two SVGs and five score PNGs
were inspected. The unchanged suite recorded 11 passes and three mode skips.
All 922 concentrated and 115 integration cases passed. Strict typing,
formatting, Compiler scope, architecture and public API checks passed.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_toffoli.ipynb
```
