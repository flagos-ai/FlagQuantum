# T gate source family

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The original files retain the
Google LLC 2023 Apache-2.0 attribution.

| Inventory | Frozen Git blob | Disposition |
|---|---|---|
| Q-0262 `t_gate.ipynb` | `cbdf39584e2062b270487bdcee8e27465c44b108` | Closed by the complete native T gate tutorial. |
| Q-0263 `t_gate.py` | `77cc66bbed4f0fcf35d2565e338075890b0889f6` | Complete native source disposition, including approved Proposal 123 inverse conversion through both adapters. |
| Q-0264 `t_gate_test.py` | `3bfc107c313fc5757b4b715a493713852681e4c1` | Original tests and full native three-gate Cirq diagram/adapter assertions covered. |

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
| `test_to_cirq` | Full H/T/T-adjoint now round-trips as the exact ordered atoms and restores the entire H matrix. |
| `test_pl_interop` | Native T and T-adjoint round-trip as the corresponding built-in PennyLane operations and match their complete matrices. |
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

## Complete source capture and presentation follow-up

The actual locked atoms now pass through `import_register_emitter` in both
directions, followed by owned CircuitIR JSON round trips. The captured provider
and the existing native `phase_family_ir` provider supply the same single
instruction and operand to the unchanged append consumer. Full complex operators,
an entangled spectator state and inverse restoration agree with original tensors.
The named q port can move to physical wire one. There is no binding swap, borrowed
work, classical storage or measurement history in either direction.

The complete factory, DocSpec, adjoint toggle, tensor index order and tensor tags
are checked against the frozen source files. The Drawer-owned shared
`fixed_phase_wire_symbol` preserves the empty root connector and the directional
port label. Full musical-score geometry agrees with the original in both
directions. Arbitrary non-null registers and ignored indices retain the source
port-label convention.

The concentrated 136-case regression passes without skips or warnings, including
the installed optional PennyLane adapter checks and the existing control-branch
evidence. Both changed Python files pass strict typing, Black, Ruff, Compiler
scope, architecture, multilevel IR and protected API checks. The source files
and original test blobs remain unchanged.

Each direction emits exactly one phase-family primitive. Independently, the
source QEC model declares one Clifford for S-family or one T for T-family, and
one logical qubit. These physical and model accounts remain separate. The public
Cirq/PennyLane inverse-export boundary was the remaining gap at that follow-up;
Proposal 123 now closes it, completing Q-0263 and Q-0264.

## Approved whole-file closure

The user approved Proposal 123 on 2026-11-25. Static Cirq exports T inverse;
static PennyLane exports the built-in one-level adjoint T, with exact `tdg`
reimport. Full complex operators on both physical placements, spectator phases,
source three-atom circuits, exact inverse restoration and SDK-free native append
pass. No private work or measurement branch applies to these atomic adapters.
One emitted T-adjoint remains separate from the original one-T resource model.

The shared acceptance receipts and full two-version regression counts are in
the approved closure section of `S_GATE_FILE_AUDIT.md`. All original definitions,
tests and direct dependencies retain their earlier source/tensor/model/display,
control and tutorial evidence. Unknown adjoints and controlled wrappers do not
gain static admission. No original source, public entry or Core opcode changed.

The frozen `TGate` constructor has only `is_adjoint`, with no precision field.
Its atom costs one T; the existing controlled-owner resource comparisons also
pass. The early ledger note about a precision-sensitive terminal does not
describe a missing TGate parameter. Generic rotation precision remains owned
by the separate rotation resource inventory.
