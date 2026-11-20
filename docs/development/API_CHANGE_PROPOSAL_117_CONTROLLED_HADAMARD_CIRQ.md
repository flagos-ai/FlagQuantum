# Proposal 117 — Static controlled-Hadamard Cirq conversion

## Review status

Pending explicit user approval. This proposal covers the remaining static Cirq
contract gap for complete Qualtran Hadamard source and test files Q-0240/Q-0241.
It is independent of pending Proposal 116 fixed-point storage admission.

## Requirement and reference evidence

The reference is unchanged Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, Google LLC, Apache-2.0,
`qualtran/bloqs/basic_gates/hadamard.py`. CHadamard is an atomic two-qubit
operation with ordered control and target, positive binary control and exact
Hadamard action on the active target. Its source Cirq conversion and reverse
conversion succeed. Native named-register capture, execution, adjoint, source
resource models, diagrams and complex spectator states already pass.

Static `to_cirq` currently refuses native `ch`; static `from_cirq` does not lower
Cirq's canonical controlled-Hadamard gate. `HADAMARD_FILE_AUDIT.md` records this
specific protected gap. The existing source-binding test explicitly asserts the
export refusal. The machine-readable static SDK contract and its verifier own
the admitted opcode set. Changing their behavior requires an approved contract.

## Proposed contract

Admit the existing native `ch` instruction to static Cirq import and export.
Export one instruction as `cirq.H.controlled()` on control then target. Import
its canonical single-positive-binary-control representation as one native `ch`
instruction. The existing conversion entry points, conversion report types,
wire mapping, declared extent, IR version and circuit.append(op) entry stay in
use. No new operator, parameter, serialized schema or public method is needed.

Exact matrix semantics include complex relative and global phase. Recognition
must use the actual controlled-Hadamard protocol, not class-name strings or a
matrix approximation. Existing normalization may be reused only when it
preserves the complete operation. Noncanonical powers, extra/negative controls,
qudit controls, tagged/classically controlled forms and other unsupported
operations retain their respective existing admission or diagnostic paths.
This proposal does not admit arbitrary controlled gates or custom matrices.

The independent source model remains four Cliffords plus two rotations.
Emission and static round-trip remain one `ch` primitive. The source operation
is self-adjoint and uses zero private workspace and zero measurements.

## Ownership and implementation sequence

The integration change updates `contracts/cirq-interop-contract.toml` and its
verifier with this exact admitted operation. Ecosystem implements the existing
Cirq conversion owner. Compiler, Core, Runtime and Simulation retain their
existing `ch` semantics. Architecture boundaries and optional-dependency loading
remain authoritative. Existing report vocabulary and stable exports are reused.

First commit the approved contract and source acceptance requirements. Then
commit the converter implementation independently. Finally commit complete
Hadamard source-file evidence and synchronize both parent checklist entries,
the stage overview, NEXT and PR evidence after all local gates pass.

## Acceptance

Compare both original factories and library round-trips. Export and import
canonical CH with ordinary and reversed wire layouts and retained spectators.
Compare the complete complex operator with the unchanged original tensor and
Cirq matrix, self-adjoint restoration, actual source scalar-control callback,
existing multi-control measured branches, clean work and native append entry.
Assert lossless reports and one emitted primitive. Retain unsupported-form
negative examples, original atomic decomposition refusals, independent resource
models and full diagram evidence.

Run the complete Hadamard source tests with existing optional PennyLane support,
Cirq 1.6.1 and 1.7.0 profiles, concentrated SDK/controlled-H regressions, local
integration, strict typing, format, owner scope, architecture, IR and public API
checks. Update only the specifically approved refusal expectation. No converter
implementation or changed contract is included with this pending proposal.

Root AGENTS.md and PUBLIC_API_PROTECTION.md require explicit approval before
changing documented accepted inputs and exception behavior. The Cirq contract
is a protected integration surface. Existing G2 authorization alone does not
substitute for this specific approval.
