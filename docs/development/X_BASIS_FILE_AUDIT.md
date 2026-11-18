# X-basis source family

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The original files retain their
Apache-2.0 attribution.

| Inventory | Frozen Git blob | Disposition |
|---|---|---|
| Q-0268 `x_basis.py` | `8b903d048e307789d71539441088efeec2868f99` | Actual Pauli binding and complete native symbols are verified. Only the recorded source controlled-measurement phase callback failure remains open. |
| Q-0269 `x_basis_test.py` | `cf8ce512efec89ede19cddf60a27cdaff6d1cddf` | Closed. All ten source tests have native evidence. |

## Bindings and original tests

| Original tests | Native counterpart |
|---|---|
| `test_plus_state`, `test_plus_effect`, `test_plus_state_effect` | The X-basis state and effect use right and left scalar `q` ports, respectively. Their complex vectors match and the prepared plus ket contracts to scalar one. |
| `test_to_cirq` | The locked plus/X and minus/X compositions have the same native gate operands and complete complex outputs. The original minus-state Cirq container is decomposed to its H/X leaves for the gate comparison. |
| `test_pl_interop` | Native X exports as the original PennyLane PauliX, round-trips and matches its complete matrix. |
| `test_x_truth_table` | The native scalar domain, X value callback and formatter produce the exact locked two-row table. |
| `test_controlled_x` | The original predicate `QUInt(8) == 1` is bound to a reversed wire list. The full primitive sequence, bit significance, operands, classical slots, output map and model match the original. The original kept graph has 7 And, 7 And-adjoint, 28 X and 1 CNOT. |
| `test_meas_x_classical_sim`, `test_meas_x_basis_state_phase` | Existing native tests cover both bit values, identical seeded outcomes and fixed branches, the delayed `(-1)^c` phase for input one, and invalid-value diagnostics. |
| `test_meas_z_supertensor` | Despite the original test name, the subject is MeasureX. Existing tests preserve pure contraction refusal and tensor-channel semantics. The prepared Zero, One and Plus cases give the same complete two-by-two classical output density as the locked superoperator. |

Preparation uses a clean physical wire. Plus emits H and Minus emits X then
H. Effects contract the unnormalized complex overlap on the consumed wire.
The native controlled X uses the shared conjunction ladder with seven work
wires, one target CX, 28 T/Tdg and seven measurements. The kept resource
graph is recorded separately from these physical instructions. Existing small
selection tests verify all tested cleanup histories and exact inverse action.

Logical MeasureX consumes `q` and returns classical `c`. The retained physical
slot can be cleared using the same result, which adds conditional X after H
and measurement. The source model counts one measurement and zero T. Earlier
evidence compares the exact measurement tensors and full complex measurement
branches with spectators, including preserved phase and work cleanup.

All ten unchanged source tests passed. The 913-case native profile passed
without skips and included Cirq and PennyLane integration. Strict typing,
formatting, team scope, architecture, Program IR and public API checks passed.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_measurement.ipynb
```

## Whole source-interface continuation

The shared basis symbol owner now covers the empty state/effect title, plus
and minus directional labels, the X title and ModPlus glyph, and the unchanged
MeasureX default name/port labels. Complete geometry is compared for all four
actual vector factories, X and the actual measurement factory, retaining the
classical output lane. X's unknown-register fallback remains ModPlus.

The actual X atom is captured in both source directions and restored through
native JSON. It emits exactly one X with no work, measurements or binding swaps.
Its complete matrix, coherent spectator state and inverse restoration pass via
`circuit.append(op)`. Source Plus/Minus effects retain their direct physical
conversion ValueError. The bidirectional emitter exposes that same failure
when asked to capture a preparation and its effect. Existing native vector
emission and tensor contraction remain authoritative for those numerical
paths. Existing classical values, phase descriptors, exact controlled-X adders,
measurement superoperators, all small raw histories and clean work are reused.

The vector field-hiding transformer and concrete factories preserve the source
single-qubit signatures. Their source `n != 1` refusal remains separate from
integer-word preparation. MeasureX consumes quantum q and returns classical c,
so its physical execution uses the dynamic measurement owner, while named
through-port capture refuses the sided signature. Measurement primitives and
its independent one-measurement model remain separate as previously recorded.

Q-0268 stays open only for the previously recorded source controlled-MeasureX
right-output phase callback failure. Existing tests preserve both the actual
source exceptions and native callback behavior across both control values and
both basis inputs, with and without the supplied classical output. No failure
is reclassified as executable support. Q-0269 remains closed.

The shared X/Z whole-file batch passed 3,574 cases without skips, with 32
paired source/native NumPy overflow warnings. Strict typing of both changed
Python files, Black, Ruff, Compiler scope, architecture, multilevel IR and
protected API checks passed. Initial callback checks caught the source CZ
filled-circle default and an invalid dataclass replacement of native Register.
The callback now reuses directional glyph output directly, and all source
geometry passes. The integer-adjoint test also distinguishes wrapper creation
from its later source decomposition failure.
