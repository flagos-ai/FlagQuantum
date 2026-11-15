# X-basis source family

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The original files retain their
Apache-2.0 attribution.

| Inventory | Frozen Git blob | Disposition |
|---|---|---|
| Q-0268 `x_basis.py` | `8b903d048e307789d71539441088efeec2868f99` | The native vector, Pauli and measurement owners cover the verified operations. Foreign object and wire-symbol dispatch, and the source controlled-measurement callback failures, remain open. |
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
