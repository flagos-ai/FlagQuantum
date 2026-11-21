# Proposal 119 — Registerless scalar phase in Program

## Review status

Pending explicit user approval.

## Source requirement

Locked Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`, Google LLC, Apache-2.0,
`qualtran/bloqs/basic_gates/su2_rotation.py` constructs one registerless
GlobalPhase and three one-qubit rotations. The actual source catalog factories
`su2_rotation_gate`, `hadamard`, and `t_gate` all contain that scalar child.
The source declaration order gives GlobalPhase instance zero. Its graph
traversal visits the connected Rz/Rx/Rz chain before the disconnected scalar.

FlagQuantum already owns exact scalar emission through scalar_phase_ir,
full SU2 execution, symbolic scalar expressions, and typed nested calls.
The finite Program opcode vocabulary currently requires quantum operands for
its physical phase instruction. A registerless typed child therefore needs an
owned scalar effect to retain both the original call graph and full execution.

## Proposed contract

Admit the Program-only `scalar_phase` operation in the existing Operation model.
It has one scalar parameter `theta`, zero operands, and zero results. Its action
multiplies the entire state by exp(i * theta). Constants and owned Parameter /
ParameterExpression values use the existing scalar admission and lowering.

- Preserve each scalar effect through Program copy, nested call expansion,
  verification, and scheduling. Inverse construction negates theta.
- Registerless functions use the existing empty register signature and can be
  called through ComponentProgramBuilder with an empty binding mapping.
- Typed physical binding lowers the effect using existing scalar_phase_ir on
  an available physical quantum wire. The carrier retains its arbitrary state
  and ownership. A live input or allocated live work wire can carry the phase.
- A wholly registerless Program retains its logical graph and scalar tensor
  meaning. Physical binding requires an existing quantum carrier because the
  public CircuitIR contract requires a positive wire count. Diagnose absence
  of a carrier before producing an executable circuit.
- Conditional effects retain existing conjunction/disjunction feedback rules.
  Scalar placement preserves the selected call-local schedule and branch phase.
- Logical QEC cost and logical quantum width are zero for the scalar child.
  Actual emitted X/phase carrier primitives remain separately measurable.
- Preserve exact complex branch amplitudes, quantum work cleanup, parameters,
  and gradients. The operation acts on the full joint state, including any
  spectator entangled with its carrier.

Core owns the internal opcode signature, verification and owned scalar literals.
Compiler owns typed construction, exact adjoint, finite expansion, physical
carrier selection and existing primitive lowering. Existing Simulation owners
execute the resulting IR. Generic logical tensor and classical-action consumers
must either implement the scalar effect exactly or issue their existing
unsupported-operation diagnostic. Resource owners report zero logical cost.

Consumer entry remains circuit.append(op). The proposal extends internal
protected IR behavior and uses existing public Circuit and parameter types.
Proposal 107's symbolic phase approval supplies scalar values, while this
proposal supplies registerless Program effect semantics.

## Acceptance evidence

1. Execute the three unchanged original SU2 catalog factories and retain their
   full ordered dependencies, operand identities and registerless child.
2. Compare fresh source drawings and all complex matrix entries, including
   source global phase, for all three factories and exact inverse recovery.
3. Verify empty-signature nested calls, repeated phases, symbolic binding,
   lowered CircuitIR JSON round trip, schedule preservation, and negative malformed opcodes.
4. Exercise entangled carrier/spectator states and each applicable measurement
   branch. Verify clean work and old aliases retain their ownership.
5. Verify carrier absence diagnostics and unchanged positive-wire CircuitIR
   admission. Compare measured carrier primitives and zero-cost source model
   in separate receipts.
6. Run strict typing, formatting, team scope, architecture, multilevel IR,
   protected API and relevant shared regression before publication.
