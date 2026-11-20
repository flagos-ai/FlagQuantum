# Register gate source integration audit

## Frozen source and ownership

The reference is Qualtran v0.7.0 revision `bcb17bb25360ed68ce3bc084645db3e929799389`. The original files remain unchanged. This batch is classified `single_device_fast_path`. Ecosystem owns SDK translation, Compiler owns typed calls and binding, Core owns registers and signatures, and Simulation owns numerical execution. The reusable tutorial uses only native owners and supports `circuit.append(op)`.

| Parent | Complete file | Git blob | Disposition |
|---|---|---|---|
| Q-0162 | `qualtran/_infra/gate_with_registers.ipynb` | `108c73c15f07476d976e0777c4b916bb466670f7` | Accepted native replacement of all 17 cells, preserving cell types and source-cell metadata. |
| Q-0163 | `qualtran/_infra/gate_with_registers.py` | `df21c8cc00ae7cac7a8bfcd4d769f76daac5515f` | Zero-width ownership accepted. Open pending wider typed/control dependency closure. |
| Q-0164 | `qualtran/_infra/gate_with_registers_test.py` | `31fd6f8635cec0b18b1c1f3487d4db9c8fabacf3` | Original five tests pass, with native counterparts for every original scenario. |

## Complete definition mapping

The inventory counts 26 class/function/method occurrences, including the class and both overloads. It is navigation and ownership evidence. It does not independently close Q-0163.

| Source definition | Native owner | Evidence or remaining boundary |
|---|---|---|
| total_bits | Core register widths and adapter operand layout | SDK raw operand count includes LEFT and RIGHT independently. |
| split_qubits | Ecosystem register import / Core split bindings | Shaped r1/r2/r3 and tutorial words preserve operand order. |
| merge_qubits | Ecosystem _port_arrays / register export | Scalar Qid normalization now matches source; shape and alias validation retained. |
| get_named_qubits | Ecosystem _named_qubits | Source named operands matched exactly. |
| _qubit_array | Ecosystem _named_qubits | Shaped indices and per-bit suffixes owned at SDK boundary. |
| _qubits_for_reg | Ecosystem _named_qubits | Scalar bit and multibit names match source. |
| _get_all_and_output_quregs_from_input | Compiler bind_register_program / Ecosystem workspace replay | LEFT release, THRU retention and RIGHT allocation preserve output leases. |
| _get_cirq_cv | Ecosystem _control_values | Binary polarity evidence reused; wider typed admission remains Q-0158. |
| GateWithRegisters | Core Signature / Compiler Program / Ecosystem RegisterCirqGate | Responsibilities distributed across existing native owners. |
| decompose_bloq | Compiler ComponentProgramBuilder / Ecosystem register importer | Tutorial graph has 2 Split, 6 CNOT, 2 Join; zero-width source graph fails. |
| as_cirq_op | Ecosystem register export and workspace gate | Directional source diagram and lease context agree. |
| wire_symbol | Drawer register score / Ecosystem wire_symbol | Existing symbol callback remains authoritative; optional first label added. |
| _num_qubits_ | Ecosystem RegisterCirqGate operand layout | Directional SDK arity 3 differs from logical input/output arity 2. |
| decompose_from_registers | Ecosystem _register_operations | Original H5/X3 stream and directional X/Y/Z body compared. |
| _decompose_with_context_ | Ecosystem workspace replay | Clean external manager lifetimes and output lease mapping compared. |
| _decompose_ | Ecosystem RegisterCirqGate | Existing contextual decomposition reused. |
| on | Cirq Gate validation at Ecosystem boundary | Native gate uses ordered physical operands. |
| on_registers | Native named Program bindings / Ecosystem register importer | User append accepts reusable native operation; scalar r3 import fixed. |
| __pow__ | Compiler register_power_ir / Ecosystem gate power | Powers -10/-1/1/10 compared; title retained. |
| _get_ctrl_spec | Ecosystem typed control and _control_values | Native signature/value interface retained; source argument dialect stays external. |
| controlled overload 1 | Ecosystem control_register_operation | Binary controls compared against source Cirq-style control. |
| controlled overload 2 | Ecosystem control_register_operation | Typed CtrlSpec source comparison for both polarities. |
| controlled implementation | Compiler quantum_control_signature / Ecosystem control | Directional metadata tested; whole wider typed admission remains open. |
| _unitary_ | Simulation unitary / Ecosystem matrix import | Full complex operator and entangled spectator compare. |
| my_tensors | Simulation / existing tensor owners | Original atomic 4-bit tensor fallback compared as one stored matrix leaf. |
| _circuit_diagram_info_ | Ecosystem RegisterCirqGate / Drawer labels | Exact named tutorial and directional diagrams compared. |

## Original scenarios and native execution

`tests/unit/test_register_gate_tutorial.py` verifies all three frozen blobs and all 17 original cells. The native notebook runs with imports of Qualtran, Cirq and Kickmix forbidden. `examples/algorithms/register_gates.py` builds one shared typed CX definition, splits the two input words, emits the original ordered pair loop and rejoins their exact dtypes. Singleton words and nonconsecutive wire placement are covered.

The five original tests cover register decomposition and controls/powers, atomic unitary-to-tensor fallback, automatic directional decomposition, nonunitary controlled metadata, and notebook execution. Their native counterparts check the H5/X3 stream, both binary control polarities, integer powers, atomic matrix storage, directional X/Y/Z and leases, exact diagrams, and the complete native notebook. The original suite also runs its actual notebook kernel.

Full complex operators and an entangled spectator state agree at the asserted tolerances. The tutorial has no measurement or work allocation, so measurement histories are inapplicable. The directional case starts its consumed LEFT bit at one, applies X to return it to zero, releases it, and reuses that lease as the RIGHT output. The separate idle work wire remains zero. Control metadata for this sided operation is compared separately from the wider controlled physical implementation.

## Resource accounting

| Scenario | Emitted native instructions | Stored graph | Frozen source QEC model |
|---|---|---|---|
| Two-by-three tutorial | 6 CX | 2 split + 6 shared CX calls + 2 join | 6 Clifford, 0 T, 0 measurement |
| Original `_TestGate` | 5 H + 3 X | Imported concrete decomposition | Clifford stream compared directly |
| `_TestGateAtomic` | 1 `any` matrix leaf | Atomic matrix storage | No hardware decomposition inferred |
| `BloqWithDecompose` | X + Y + Z | LEFT free then RIGHT allocate | 3 Clifford, 0 measurement |

A stored matrix leaf is counted at its storage granularity. Physical primitive synthesis is outside that atomic result. Source `total_bits` sums raw SDK operands. The directional l/t/r signature therefore has three raw SDK operands and two logical bits on each side. `Signature.n_qubits()` describes the logical side width, so these counts are kept separate.

## Concrete fixes and unresolved boundaries

The original scalar `r3=Qid` placement exposed a native shape refusal before this batch. `_port_arrays` now wraps a scalar Qid using the same normalization as frozen `merge_qubits`. The failure and passing original scenario are retained in the reference receipt directory. Optional private `gate_label` preserves the original first-operand class title and survives integer powers and adjoint. Existing unlabelled behavior remains covered.

Zero-width words and empty register arrays succeed in source layout/unitary paths. Original `decompose_bloq()` raises `ValueError` through empty-tuple identity collisions for scalar and positive-shaped empty words, or zero iteration dimensions for zero-sized arrays. Native register import now retains independent zero-width identities as detailed below. Wider classical words and control admission remain under Q-0158 and the remaining generic control dependencies. Proposal 107 approval covers nonlinear parameters and scalar phases.

## Reproducible checks

Use the locked reference environment and `tests/unit/test_register_gate_tutorial.py`, together with existing register import/export/control and symbolic parameter suites. Evidence files under `../ftqc-reference-env/` include `g2-register-gate-focused.log`, `g2-register-gate-original-complete.log`, `g2-register-gate-regression.log`, `g2-register-gate-integration.log`, the strict/format/scope/architecture/IR logs, environment JSON, source inventory and zero-width boundary receipt. Default-suite baseline failures and optional dependency skips are recorded separately when collected.

Final batch results are 478 concentrated checks passed with two existing optional skips, 172 local integration checks passed, five files passed strict typing, and format, scope, architecture and IR checks passed. The additional Cirq 1.6 lane passed 70 cases with 669 NumPy deprecation warnings. The dependency scan reports zero findings on changed paths and 688 whole-repository findings.

The default tier selected 74,391 cases and reached its 480-second bound before completion. Its log contains failures in the existing example policy and circuit composition contract suites. The example policy failure was isolated and reproduced, with baseline source inspection recorded in `g2-register-gate-baseline-comparison.json`. The new example and notebook contain no matching forbidden runtime import. This default run is incomplete and failed. It supplies no all-repository passing claim.

## Zero-width SDK binding continuation

The importer's physical-qubit map continues to own nonempty words. Empty scalar
words and indexed zero-width elements use a separate `(register name, index)`
map at this external boundary. Each value is an existing RegisterBinding with
its own SSA identity. Empty arrays have no elements. Calls and outer ports match
empty words by the declared name and row-major index because a flattened SDK
operand carries no address for those words. Unmatched names create independent
clean logical values. A live empty output cannot overwrite another value.
This explicit naming convention is a native ownership decision. Source empty
qubit tuples do not provide a unique physical identity.

The existing builder owns consumption, split/join type rebinding, allocation and
freeing. The adapter adds no new Core record or physical primitive. Cleanup
consumes all remaining nonempty and empty handles independently. Scalar phase
carrier selection continues to see only physical words. Quantum zero-width
admission is already covered by Proposal 103. Wider classical-word admission
remains independent under pending Proposal 108.


The continuation passes 19 focused cases and all 484 concentrated register,
SDK, graph, binding and feedback cases. Both frozen source blobs are checked
inside the test fixture. Six attached/declared definition cases cover LEFT,
THRU and RIGHT ownership. Five width/shape layouts include zero-sized arrays.
Type rebinding, unmatched names, repeated calls, live-output collisions and
scalar phase carrier allocation are explicit cases. Four original gate fixtures
preserve their actual upstream decomposition failures while their SDK unitary
and primitive behavior remain comparable.

The And compute/erase consumer emits 3 H, 2 T, 2 T-dagger, 6 CX, 1 S,
1 measurement, 1 CZ and 1 reset with or without three empty logical values.
The entire ordered stream, operands and feedback are equal to the original
primitive decomposition. Both measurement histories match unnormalized complex
amplitudes with an entangled spectator and return the work bit to zero. The
separate source model reports 1 And, 1 Clifford and 1 measurement. Empty logical
values emit zero additional primitives. Scalar phase uses the existing clean
physical carrier only when the body requires it.

Strict typing for both changed Python files, Black, Ruff, team scope,
architecture and IR checks pass. Dependency inspection reports 688 repository
findings and zero findings on changed paths. The default suite stops after
434 passes and 51 skips on the existing public-docstring coverage failure.
Its nine reported modules and failing test are byte-identical to baseline
`0ea36354ed7f35b5d7fcb4917da084f8739cdba9`. The initial concentrated invocation
lacked the frozen source notebook marker. Reusing the existing reference-marker
runner resolved that harness issue, with all 484 cases passing.

Receipts are `g2-empty-register-evidence.json`, `g2-empty-register-dependency.json`,
`g2-empty-register-default-baseline.json` and the matching focused, regression,
default and quality logs in the reference directory. The complete Q-0163 parent
remains open solely for the wider typed/control dependency tracked by Q-0158
and the remaining generic control dependencies.

The 2026-11-20 review of all 26 class/function definitions closes the canonical
one-bit binary field/polynomial control dependency using the existing shared
control owner. The full dependency evidence and original wider-field refusal
are recorded in `CONTROLLED_VIA_AND_FILE_AUDIT.md`. Foreign SDK object contracts,
nonbinary/symbolic physical execution and previously recorded source defects
retain their existing scope.

## Nonbinary classical dependency continuation

Proposal 111 closes the concrete classical nonbinary field and polynomial
port/read/control dependency. The unchanged full source inventory is reused.
`NONBINARY_CLASSICAL_FIELD_FEEDBACK_AUDIT.md` records source callback/codec
comparison, explicit polynomial tensor failures, complete child streams and
measurement histories. No generic SDK dialect or nonbinary quantum admission
is inferred. Q-0163 still requires the remaining control/interface review.

## Concrete remaining SDK storage boundary

The post-Proposal-111 interface regression passes 599 cases, with two optional
skips for PennyLane and qiskit-aer. Existing primitive, callback, branch,
workspace, diagram, parameter and empty-register evidence is reused.

A separate source intake uses a register gate with a CBit or CGF(3) THRU port
and an X on its first raw SDK operand. Unchanged source `on_registers` and Cirq
decomposition accept one and two physical operands respectively. Explicit
native classical signatures are refused with `Cirq placement requires a
concrete quantum register`. The receipt is
`g2-register-gate-classical-sdk-intake.json`. Raw SDK projection and true native
classical storage have distinct execution contracts. Proposal 111 admits
classical Program ports and read-only feedback. It does not authorize assigning
quantum wire ownership to classical values or conditional classical writes.

This is a concrete remaining interface boundary of Q-0163 and its control
consumer, independent from the now accepted field word dependency. Symbolic
physical execution and source defects remain separately recorded. The complete
parent stays open. The next closure must determine the explicit native storage
semantics of these source consumers before changing protected behavior.
