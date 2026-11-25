# S and S-adjoint source family

The unchanged reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The two files retain the
Google LLC 2023 Apache-2.0 attribution.

| Inventory | Frozen Git blob | Disposition |
|---|---|---|
| Q-0253 `s_gate.py` | `6111677a1e55ccab022e988598366a0f1bdcfc94` | Complete native source disposition, including approved Proposal 123 inverse conversion through both adapters. |
| Q-0254 `s_gate_test.py` | `98ed87f00f6788bd20a54eeed8b3770e738f13a8` | Original tests and native diagram/adapter assertions covered in both directions. |

Both directions have the scalar `q` THRU signature, emit one `s` or `sdg`
instruction and preserve the source complex matrices. The adjoint toggles
the direction and the score helper renders the corresponding `S†` label.
Each version has one atomic graph leaf and one modeled Clifford. Neither
direction requires measurement, feedback or work wires. Existing control
tests compare full complex states, clean work and all tested measurement
histories for multi-control variants.

| Original test | Native evidence |
|---|---|
| `test_s_gate` | The locked factory binds to the native diagonal tutorial and executes as one atomic gate. The original harness skips decomposition, counts and decomposition-dependent dtype modes. |
| `test_call_graph` | The native graph and the locked graph each retain one leaf and zero edges for S and S-adjoint. |
| `test_to_cirq` | The full native H/S/S-adjoint circuit now matches the source operations after mapping its single allocated virtual wire to native LineQubit zero. The complete complex state and exact native instruction roundtrip agree. |
| `test_pl_interop` | Native S and S-adjoint produce the same PennyLane operations as the locked source, round-trip, and match the full complex matrices. |
| `test_tensors` | Full complex S and S-adjoint matrices are checked against the locked tensors. The unchanged source suite also compares both against Cirq. |

Proposal 123, explicitly approved on 2026-11-25, extends both static contracts
for `sdg` and `tdg`. The earlier inverse-export gap is closed below.

The unchanged source suite recorded six passes and three mode skips. The
98-case native regression included the Cirq and PennyLane conformance paths
and passed without skips. Strict typing, formatting, Compiler scope,
architecture, multilevel Program IR and public API checks also passed.

Use the existing tutorial to append and verify S, its inverse, its square,
and both controlled phases:

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_diagonal_gates.ipynb
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
the approved closure below completes the implementation and test inventories.

## Approved whole-file closure

Proposal 123 reuses existing Core phase atoms. Cirq exports canonical S inverse;
PennyLane exports the built-in adjoint S. Static reimport restores `sdg`, its
operand and exact absolute phase. No general phase substitution, extra gate,
private work, measurement or feedback is introduced. The one-level PennyLane
recognizer rejects arbitrary adjoints, powers, controlled wrappers and subclasses
with the same display name. Other adapter exclusions remain unchanged.

Both complete S/T families reuse the accepted source method, factory, DocSpec,
display, tensor, resource, control-history and notebook evidence. The new adapter
batch passes 153 cases on Cirq 1.7.0/PennyLane 0.44.1 and 153 on
Cirq 1.6.1/PennyLane 0.45.1, plus 145 local integration cases. Fourteen execution
and research-provider cases are intentionally outside this static batch, not
silent skips. The unchanged S/T suites pass fourteen cases with six expected
atomic-mode skips. The downloaded second PennyLane wheel identity is recorded.
Strict typing covers seven touched Python files; formatting, scopes, both
contracts, language, architecture, IR and public API checks pass. Final adapter
and contract checks pass 26 cases on each version combination.

Receipts: `g2-static-phase-1.7-0.44.1-20261125.log`,
`g2-static-phase-1.6-0.45.1-20261125.log`,
`g2-static-phase-original-20261125.log`,
`g2-static-phase-quality-20261125.json` and
`g2-static-phase-wheel-sha256-20261125.txt`. The source files are unchanged.
Q-0253 and Q-0254 have complete-file closure evidence.
