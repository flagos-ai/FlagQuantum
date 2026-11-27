# Pauli Y and controlled Y source file audit

Reference: Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The source files are under `qualtran/bloqs/basic_gates/` and are licensed under
Apache-2.0, copyright Google LLC.

| Item | Source file | Frozen Git blob |
| --- | --- | --- |
| Q-0270 | `y_gate.ipynb` | `1d9a6795a74eb594e8ff213984277109ad6cab1e` |
| Q-0271 | `y_gate.py` | `a3b305679692809a00407b630e2cc9f4227caebe` |
| Q-0272 | `y_gate_test.py` | `9c7dab6b15ec1055f3a1d90fa08bb5d3a211a1a6` |

## Register binding and direct dependencies

Y has the through port `q`. CY has the ordered through ports `ctrl, target`.
A positive scalar control on Y binds directly to CY. Both leaves are atomic and
self-adjoint. Their operand order is preserved through the emitted IR. The
control retains its value. The target flips with phase `+i` on zero and
`-i` on one in the active sector. The two atomic leaves need zero workspace
and emit zero measurements.

The Compiler owns fixed Pauli control dispatch in
`registers/primitives/pauli_gate.py`. Its shared predicate owns the temporary
work wires and classical slots for additional controls. Existing
`test_register_pauli_gate.py` and `test_register_controlled_pauli.py` compare the
source calls, bound operands, and conditional cleanup for each enumerated
history with the full complex output. The nested CY case erases the inner
predicate before the outer predicate. Simulation owns local numerical
execution and tensor-network basis projection. The Cirq and PennyLane adapters
own foreign gate types.

## Original test coverage

| Source test | Native verification |
| --- | --- |
| `test_y_gate`, `test_cy_gate` | Ordered signatures, declared self-adjoint action, atomic emission, full matrices and QEC models. |
| `test_to_cirq` | Plus and minus preparations followed by Y, identical exported leaf order and complete complex states. |
| `test_pl_interop` | Source gate objects and full complex matrices, plus native Y and CY PennyLane round trips. |
| `test_cy_vs_cirq` | Source CY specialization, canonical Cirq matrix, and native atomic binding. |
| `test_cirq_interop` | Unchanged source circuit shape, exact operand order and native resource identity. Native static round trip is now implemented under approved Proposal 127; original SDK-qualified inverse mapping diagnostics remain separate. |
| `test_active_cy_is_y` | Both target basis columns after one preparation and one tensor-network effect on the control. |
| `test_cy_adjoint` | Two consecutive CY operations give the full identity matrix. |
| `test_cy_t_complexity` | One modeled Clifford and one emitted CY. |

With Cirq 1.7.0 and PennyLane 0.44.1, the unchanged original suite reports
10 passed, 6 mode skips, and one failure. The positive-control constructor
returns canonical `cirq.CY`. Qualtran's inverse lookup recognizes
`cirq.ControlledGate(cirq.Y)` and wraps `cirq.CY` in `CirqGateAsBloq`. The
FlagQuantum resource adapter recognizes both forms. At the time of that run, the separate protected static Cirq contract excluded
`cy`. Proposal 127 was separately approved on 2026-11-27 and now closes that
conversion gap. Generic source wire-symbol dispatch and external object wrappers
are covered by the subsequent capture/presentation evidence below.

| Case | Emitted primitives | Independent source declaration |
| --- | --- | --- |
| Y | 1 Y, no work or feedback | 1 Clifford, 1 bound qubit |
| CY | 1 CY, no work or feedback | 1 Clifford, 2 bound qubits |
| CY with three extra bits | 1 CY, 12 T/Tdg, 3 measurements, 3 work qubits; the full stream also includes Clifford and feedback instructions | 3 temporary And calls, 3 And adjoints, 1 Clifford, 8 qubits including workspace |

## Tutorial

`examples/notebooks/register_y_gate.ipynb` preserves the two source instances,
signatures, and musical scores. Its 18 cells extend the original 16 with a
full complex action and resource example through the ordinary append path.
The unchanged original and native notebooks execute in independent kernels.
The native render produces two SVG signatures and two PNG scores.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_y_gate.ipynb
```

## Actual atomic source capture and complete presentation

The actual `_y_gate` and `_cy_gate` examples and DocSpecs now terminate at the
existing named register importer. Both independently captured directions restore
through native JSON, retain exact operands under reordered physical binding,
and emit one Y or CY. Full source matrices, entangled-spectator states and
adjoint recovery match through `circuit.append(op)`. The leaves have no work
or measurements. Their independent source QEC declarations remain one Clifford,
with logical peaks one and two, respectively.

The internal feedback importer now recognizes canonical `cirq.CY`, whose Cirq
class differs from a generic ControlledGate. Previously it expanded this atom
into SX, CZ and SX-adjoint. Exact CY recognition preserves the declared atomic
stream. Separate regression cases compare every unnormalized branch of
classically conditioned CY, in both directions, to Cirq's own gate matrices and
predicate resolution. Static public conversion continues to enforce its
existing admission contract.

Drawer reuses the shared Pauli callback in `basis_symbols.py`. Y and CY complete
score geometry matches source, including empty root, filled control circle and
Y target. Y accepts an arbitrary named register, as upstream. CY rejects an
unknown register, retaining the native Register representation in its message.

The source inverse-mapping failure remains directly reproduced. Canonical
`cirq.Y.controlled()` maps to CirqGateAsBloq, whereas explicit
`cirq.ControlledGate(cirq.Y)` maps to CYGate. This discrepancy remains an original Cirq 1.7 diagnostic. Cirq 1.6 canonical
Y.controlled() returns a generic ControlledGate and maps to CYGate directly.
The native static CY gap is now resolved by approved Proposal 127.
The generic concrete source wrapper and display gaps are now covered.

The batch passed 149 concentrated cases without warnings or skips, including
existing nested-control branch/work evidence and optional PennyLane boundaries.
Strict typing of the four changed Python files, formatting, lint, Compiler and
Ecosystem scopes, architecture, multilevel IR and protected API checks passed.
Source blobs remain unchanged. Retained logs use `ftqc-reference-env/g2-y-atom-*.log`.


## Complete file acceptance on 2026-11-27

Q-0271 and Q-0272 are accepted under the explicitly approved
[faithful-migration rule](G2_FAITHFUL_MIGRATION_CLOSURE_PROPOSAL.md).
Complete source definitions, original examples/DocSpecs, notebook assets,
register bindings, numerical action, shared controlled-predicate dependencies,
SDK hooks, full presentation and original test counterparts have the owners and
evidence above. Proposal 127 supplies the last protected static conversion gap.

Both supported SDK profiles pass 107 interop cases and 45 direct Y/CY cases.
Full complex operators retain absolute scalar phase under reversed wires and
spectators; normal append executes without source SDKs. Both atomic directions
are self-inverse and have no work or measurement. Existing conditional branches
retain complete complex amplitudes and zero private predicate work. Source
model one Clifford/two logical wires remains separate from one actual CY.

The untouched original suite's 1.7 inverse-lookup failure and mode skips remain
recorded. The native diagnostic now follows the actual source constructor and
converter in each SDK instead of assuming a 1.7 wrapper on 1.6. This is not a
repair to the source library or a claim that its failing test now passes.
No runtime kernel, PennyLane admission, public entry or Core opcode is added.


Final local checks pass strict typing for all seven changed Python files,
Black/Ruff, scope, language/links, architecture, IR/public API and both SDK
contract checks. Frozen Y implementation/test blobs are checked by the actual
capture fixture. Receipts are `ftqc-reference-env/g2-static-cy-*` outside the
repository. Source runtime, gate counts and old source failure logs are retained.
