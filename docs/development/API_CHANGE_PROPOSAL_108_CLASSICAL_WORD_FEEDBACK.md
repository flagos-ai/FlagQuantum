# Proposal 108 — Typed classical word reads and feedback

## Review status

Pending explicit user approval. Proposal 107 approves nonlinear scalar
expressions and real-angle global phase. This proposal requires a separate
decision because it broadens owned Program port admission and the result arity
of an existing serialized operation. Core AGENTS and PUBLIC_API_PROTECTION.md
require approval before changing that behavior. This document authorizes no
production implementation by itself.

## Verified requirement

The complete controlled.py and classically_controlled.py source chains accept
classical integer descriptors and compare their numerical values before
delegating to the child. Native Simulation already implements those callbacks.
Core Program currently refuses even an identity function carrying a CUInt(2)
port. Compiler entry initialization and feedback reads only admit CBit.

The unchanged reference is Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`. Source controlled.py has blob
`6fabe760e793f1b39f825d76f80e92b6e5fa5c28`. Source
classically_controlled.py has blob
`a4236872f616527a1dbc92b5453cb647b656650f`.
The reproducible intake is recorded in
`ftqc-reference-env/g2-classical-word-admission-intake.json`.

The source wrapper supplies numerical action, activation and tensor semantics.
Its physical decomposition and QEC model raise DecomposeNotImplementedError.
Acceptance must compare native wrapper execution with the source numerical
action or tensor and reuse unchanged child primitive comparisons. Record the
source wrapper model refusal alongside the emitted native operations.

## Admission and ownership

Core admits existing canonical RegisterType records for classical unsigned,
two's-complement, ones'-complement, bounded unsigned, Montgomery and fixed-point
integer storage. CBit keeps its existing identity. Classical storage has zero
quantum width. Concrete execution requires built-in integral widths and
parameters, with the existing boolean fixed-point sign. Preserve canonical
parameter order and existing constructor guards. Reuse current metadata owners.

Symbolic widths and parameters remain declarations until concretely resolved.
Unknown physical encodings fail before execution. Finite-field and polynomial
classical values retain their separately tracked codec and admission boundaries.

Core owns register type compatibility, linear consumption and read signatures.
Compiler owns physical classical slot binding, snapshots and control guards.
Simulation owns existing numerical codecs and value validation. Ecosystem owns
foreign descriptor conversion. Each implementation lands in its owning module
after the approved contract commit. Existing ProgramRecord, RegisterType,
Operation and ConditionRef remain the representations.

## Read and lifetime contract

Extend the existing read_classical_register operation to consume one owned
classical element of width w. Return one successor of the identical register
type followed by w read-only bool snapshots in MSB-first order. CBit therefore
retains exactly its existing one-successor/one-bool shape. A permitted zero-width
element returns its successor with zero snapshots. Row-major array ordering
remains with the existing signature flattener.

The register handle remains linear. Bool snapshots retain the existing read-only
fanout semantics and refer to the same stored bits. Reading emits zero quantum
gates and allocates zero new classical slots. Calls forward successor handles
through existing parameters and results. Primitive leaves receive existing DNF
conditions. Structural calls and clean-work lifetimes remain unconditional.
Parity, destructive measurement and classical writes retain their current
admission contracts and are independently verified.

## Entry values and encoding

The existing internal classical_inputs tuple remains aligned with flattened
entry elements. Quantum entries use None. A classical entry supplies one built-in
integer in its declared numerical domain. CBit retains built-in zero/one and
boolean inputs. Validate the full tuple before producing a bound program.
Use existing numerical encodings to map a word to MSB-first stored bits.
Return its slots through the existing per-element classical_outputs tuple.

Signed words retain their declared complement representation. Bounded values
must satisfy their exclusive bound. Montgomery values represent stored residues.
Fixed-point values are raw integers with fractional metadata preserved. Retain
source numerical equality semantics, including the two representations of zero
for ones'-complement whenever both can reach a read. Zero-width and empty-array
ownership require explicit source-backed cases before physical admission.

Typed classical control uses the existing graph transformation, shared signature
naming and primitive guard composition. Descriptor and activation values remain
owned native data. Existing CBit callers keep their current invocation and
emitted stream. The user continues to append the bound operation through
circuit.append(op).

## Serialization and compatibility

The read operation keeps its existing opcode and record fields. The accepted
operand descriptor and result count broaden. Old Program payloads, hashes,
CBit slot allocation and failure cases remain covered by exact fixtures.
Older consumers reject a wider word or unsupported read signature during
validation. No implicit narrowing or metadata erasure is permitted. Contract
tests must establish old payload stability before owner implementation commits.

## Acceptance

1. Reuse accepted controlled descriptor, callback, tensor, child graph and CBit
   execution evidence. Inventory the complete affected source files and direct
   dependencies before declaring their parent items accepted.
2. Verify typed identity and nested-call transport before feedback. Exercise
   scalar, shaped and empty ports, valid integer domains, both activation
   polarities, mixed register families and malformed descriptors or inputs.
3. Verify read successors, MSB ordering, snapshot fanout, slot preservation and
   rejection of duplicate owned-handle consumption. Cover zero-width behavior
   explicitly and retain source defects as separate observations.
4. Compare source child operations, operands and existing feedback with the
   native child at the same primitive boundary. Verify complete complex
   amplitudes, spectator entanglement, every applicable measurement history and
   clean-work zero through the unchanged append consumer.
5. Separate emitted quantum gates, classical initializers, measurements and
   workspace from the source analytical model. Preserve source wrapper refusal
   as evidence instead of inventing an estimate.
6. Run concentrated regression, local integration, strict typing, formatting,
   classified team scope, architecture and IR/API checks. Synchronize the ledger,
   phase overview, current status and PR evidence only after acceptance.
