# T gate source family

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The original files retain the
Google LLC 2023 Apache-2.0 attribution.

| Inventory | Frozen Git blob | Disposition |
|---|---|---|
| Q-0262 `t_gate.ipynb` | `cbdf39584e2062b270487bdcee8e27465c44b108` | Closed by the complete native T gate tutorial. |
| Q-0263 `t_gate.py` | `77cc66bbed4f0fcf35d2565e338075890b0889f6` | Atomic matrices, T/adjoint resources and positive external conversion verified. Adjoint external export remains open. |
| Q-0264 `t_gate_test.py` | `3bfc107c313fc5757b4b715a493713852681e4c1` | Original suite and executable counterparts verified. Full native adapter reproduction of the original three-gate Cirq diagram remains open. |

The original factory owns a single QBit `q` THRU port. Forward and adjoint
each emit one instruction, `t` and `tdg` respectively, and preserve the locked
diagonal phases exactly. The source models each as one atomic leaf and one T.
These paths have no measurements or auxiliary wires.

The native notebook retains the original 13 cell topics and adds two cells
for inverse restoration and separate count reporting. It prepares the magic
state by appending H and T, and verifies both complex amplitudes. The source
`TestTStateMaker` applies those same gates as a through operation. The added
reference test compares its full complex operator and the exact reversed
`tdg`, H operator.

## Original test coverage

| Test | Evidence |
|---|---|
| `test_t_gate`, `test_call_graph` | The exact original factory signature and terminal count match the native signature and call graph. Both directions cost one T. |
| `test_tensors` | The native matrices agree with the pinned tensors in both directions. The original suite verifies the Cirq matrices independently. |
| `test_to_cirq` | Forward T round-trips through native interop. Native H/T/T-adjoint restores the entire H matrix. The bridge reports `unsupported_operation` for `tdg` under the current contract. |
| `test_pl_interop` | Native T round-trips as the same PennyLane operation and matches its complete matrix. The adjoint export is outside the current contract. |
| `test_test_t_state`, `test_test_t_state_tensor_adjoint` | The native through operation and its adjoint match the original fixture and compose to the exact identity. |

Both notebooks ran in independent kernels. The native kernel rejected
Qualtran, Quimb and Cirq imports. The two SVGs and one score PNG were
inspected. The original suite recorded eight passes and three mode skips.
All 120 concentrated and 114 integration cases passed. Strict typing,
formatting, team scope, architecture, multilevel Program IR and public API
checks passed.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_t_gate.ipynb
```
