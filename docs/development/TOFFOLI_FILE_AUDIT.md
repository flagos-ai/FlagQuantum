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

## Whole-file method and original-test closure

The remaining source protocols now terminate at their native owners. Drawer
owns `toffoli_wire_symbol`, preserving the empty operation title, both indexed
control circles, target ModPlus and unknown-register ValueError. Its complete
score geometry is compared against the actual source composite, retaining
`ctrl` as a QBit array of shape `(2,)` and `target` as a scalar. Ecosystem
captures both source directions into existing native named emitters. Compiler
and Simulation continue owning execution and full complex-state validation.

| Complete source surface | Accepted evidence |
|---|---|
| Signature, atomic decomposition refusal, self-adjoint identity and one-leaf graph | Actual `_toffoli` factory and frozen source capture, with no work, measurement or binding swaps |
| `my_tensors` CTRL/COPY and XOR factorization | Full source contraction against native 8-by-8 atomic operator and existing classical-action tensor suite |
| `on_classical_vals` | Reused original target assertion, ordered control short circuit, unchanged control-array identity and all basis actions in `test_classical_controlled_bits.py` |
| `as_cirq_op` | Actual named capture retains one `ccx`; direct Cirq round trip and both consecutive gates reproduce the original diagram assertion |
| `as_pl_op` | Actual original PennyLane operation equality, full matrix and native bidirectional conversion |
| `wire_symbol` | Complete source score geometry, both indexed control callbacks and exact unknown-register refusal |
| `get_ctrl_system` and its returned adder | Reused `test_register_controlled_not.py` actual source adders, both polarities, grouped controls and signed/unsigned eight-bit declarations, operand order, feedback, complex branches and work cleanup |
| `_toffoli`, `_TOFFOLI_DOC` and original generic factory harness | Actual factory result and documentation example identity, frozen source/test blobs, native concrete snapshot JSON and full operator |

The new `test_toffoli_atom_source_binding.py` rebinds the atom to reordered
physical wires with an entangled spectator. Both actual source directions
preserve every complex amplitude and restore the input after the second
application through `circuit.append(op)`. No atomic measurement branches or
scratch exist. Existing outer-control branch evidence remains applicable and
is included in the batch regression.

The emitted atom is one `ccx`. The independent source QEC model is one Toffoli
with four default T-equivalents and peak three. These counts have distinct
meanings. This batch makes no physical Clifford+T synthesis assertion.

Q-0266 and Q-0267 close with the complete method and original-test disposition.
The original generic harness skips its source serialization modes. Concrete
native snapshots pass JSON restoration, without claiming arbitrary foreign
object serialization. Source control adders retain their tested declarations
and the previously recorded eight-bit single-trajectory boundary. No reference
file is modified.

The concentrated whole-file batch passed 3,202 cases with no skips. Its 28
warnings are the paired source/native NumPy overflow cases in the reused CNOT
classical callback checks. Strict typing passed both changed Python files.
Black, Ruff, Compiler scope, architecture, multilevel IR and protected API
checks passed.
