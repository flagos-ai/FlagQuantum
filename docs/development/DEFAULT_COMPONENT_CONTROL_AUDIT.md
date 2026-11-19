# Default component control dispatch

The complete trailing helper `make_ctrl_system_with_correct_metabloq` in frozen
`qualtran/_infra/controlled.py` is owned by
`compiler/registers/control/default_component_control.py`. The source revision is
`bcb17bb25360ed68ce3bc084645db3e929799389`. Its source and test blobs remain
`6fabe760e793f1b39f825d76f80e92b6e5fa5c28` and
`0a909d79efa343dc78ae67ebab896e8dc1fd187c`. No reference source was changed.

Compiler implementation and callback consumers are committed as `137b6737f`.

## Interface and ownership

`default_component_control_system` receives existing native Register descriptors,
ControlRegister activation patterns and three existing ComponentControlSystem
factories. It returns the selected ProgramRecord and StructuredControlAdder
objects unchanged. The factory owns the child body, signature naming, typed
admission, workspace and execution. This reuses the existing callback boundary
and Program representation. Routing runs before physical lowering.

| Request | Selected owner |
| --- | --- |
| Exactly one scalar positive QBit | Total control |
| Other quantum-only metadata | Shared predicate |
| Classical-only metadata | Existing feedback |
| Mixed storage | Source NotImplementedError |
| Empty descriptors | ValueError |

The exact default comparison retains dtype and shape identity. QAny(1), QUInt(1)
and singleton QBit arrays therefore use the predicate path. Classical CBit
identity and the existing classical storage marker classify descriptors even at
zero width. Width truthiness does not determine storage. Symbolic metadata is
forwarded to the selected owner and does not gain physical execution admission.
Mismatched native descriptor/pattern counts fail before invoking a factory.
Source empty CtrlSpec construction itself raises AssertionError because its
converter makes one empty array, before the helper's empty-category ValueError
can run. The native interface has explicit separate descriptor and pattern tuples.

## Actual consumers

`test_default_component_control.py` calls the unchanged source helper and native
selector. It compares concrete routing, dtype/shape distinctions, zero-width
quantum/classical routing, mixed-storage failure, symbolic quantum metadata,
factory identity and failure propagation. A replacement witness swaps total
hierarchy construction for the equivalent positive predicate implementation
while keeping the attachment consumer unchanged.

The quantum consumer supplies the existing total hierarchy or predicate owner,
then invokes the returned adder through ComponentProgramBuilder. The source
consumer also invokes its actual returned attachment callback. The shared child
is a stored one-X call, matching both graphs. Direct source total control of an
atomic X refuses decomposition because X has no stored body. It is not used as
successful total-control decomposition evidence.

All five quantum cases compare emitted gates, operands and feedback against the
unchanged source graph. The complete complex tensor acts on an entangled
spectator. Both histories of every two-control case retain the exact
unnormalized amplitude and return predicate work to zero. No-work cases also
run through `circuit.append(op)`. The branch helper executes that same append
entry for each coherent segment in measured cases.

The classical consumer supplies classical_bit_feedback_function, invokes the
returned attachment, binds all scalar positive/negative and two-element array
inputs, and appends the operation to DynamicCircuit. Each result agrees with the
unchanged source callback's basis action on a complete complex two-qubit state.
Returned classical slots preserve the caller values. Existing measured-control
branch evidence in `test_classical_controlled_file_audit.py` is reused by the
concentrated regression. Generic classical child graph transformation and wide classical typed execution
remain open.

## Resource evidence

`ftqc-reference-env/g2-default-control-resources.json` records exact observed
primitive counters separately from the source QEC analytical model. Dispatch
adds zero primitives and owns zero work. Predicate work belongs to its existing
owner. Classical input constants produce classical_xor instructions during
binding, recorded separately from the conditionally executed child X. The
classical wrapper's generic decomposition and model refusals are retained in
`g2-default-control-boundaries.json`.

## Acceptance and remaining scope

The default routing helper and its admitted execution consumers are closed.
Q-0158/Q-0159 remain open for complete constructor/equality/hash/count metadata
coverage, symbolic typed execution, additional physical encodings and generic
classical stored-child graph controls. Previously recorded source cursor,
matrix-singleton and signed tutorial defects remain explicit in their audits.


| Quantum request | Observed primitives | Source analytical QEC model |
| --- | --- | --- |
| Scalar positive QBit | 1 CX | 1 Clifford |
| Scalar negative QBit | 2 X + 1 CX | 3 Clifford |
| Positive singleton QBit array | 1 CX | 1 Clifford |
| Two-element QBit array | 22, including 4 T/Tdg, 1 measurement and reset | 1 AND + 2 Clifford + 1 measurement |
| Two-bit unsigned word | 22, including 4 T/Tdg, 1 measurement and reset | 1 AND + 2 Clifford + 1 measurement |

The AND model implies four T-equivalent gates. Each two-control execution uses
one clean predicate qubit. The full primitive counters, including conditional
CZ and X conjugations, remain available in the receipt.

All 4068 concentrated cases and 96 SDK integration cases pass, with one and two
existing deprecation warnings respectively. Strict typing passes both changed
Python files. Black, Ruff, Compiler scope, architecture and multi-level IR checks
pass. The dependency baseline remains 687 whole-repository findings, with zero
batch-path findings after guarding the test's optional NumPy import. The source
hash and native Kickmix Simulator checks execute before writing the receipt.
