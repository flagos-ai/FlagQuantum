# Stored classical component control

## Source and ownership

The reference is unchanged Qualtran v0.7.0, revision
`bcb17bb25360ed68ce3bc084645db3e929799389`.
`classically_controlled.py` has Git blob
`a4236872f616527a1dbc92b5453cb647b656650f` and its original test has blob
`e24f6b9cc8023feedc204c1128ec458dd8476d6e`.
The source wrapper validates classical descriptors and attaches the inherited
control factory. It supplies numerical classical and tensor semantics. Its
physical decomposition and QEC model both refuse this composite child.

Compiler owns `component_classical_control.py`, which propagates CBit control
through existing stored Program functions. `component_feedback.py` owns the
shared per-function signature, linear control forwarding and primitive guard
composition. Existing Core records, verification, conditions and register reads
remain authoritative. The existing binder and DynamicCircuit append consumer
execute the result. This is a single-device CPU path.

The stored graph lowering is a native implementation of the source numerical
semantics. Physical child operations have unchanged upstream decomposition
comparisons. A physical wrapper comparison cannot be supplied by the frozen
wrapper because its decomposition is absent.

## Interface and lifetime

`classical_control_component_program(record, controls, entry="main")` accepts
ordered `(shape, bool_values)` controls using the existing feedback convention.
Each shape is concrete and positive-sized, with scalar shape `()` containing
one value. Arrays retain row-major order. Inputs prepend owned CBits in control
declaration order and return their successors before original child outputs.
The entry has THRU registers and each reachable definition has one return block.

One generated function per reachable child retains shared subcalls. Generated
names avoid existing definitions. Original child and unrelated definitions stay
available. The transformed root retains its entry name. Input records remain
unchanged and their canonical payload is invalidated in the rebuilt record.

Snapshots guard primitive gates. New guards are conjoined with every existing
DNF clause, including disjunctive child conditions. Calls forward owned control
successors without adding conditional call semantics. Structural split, join,
allocation, release and classical bool computations remain unconditional. Clean
work must return to zero on both enabled and disabled branches before release.

Reachable measurement, reset, classical discard, parity extraction and channels
are rejected by the graph transform. The previous leaf constructor retains its
existing conditional Kraus admission, protected by a dedicated regression.
Proposal 108 was approved and implemented on 2026-11-20. Canonical integer word
ports, MSB read snapshots, empty shapes and numerical feedback now execute
through the existing binder. See `TYPED_CLASSICAL_WORD_FEEDBACK_AUDIT.md` for
the scope and source tensor defect. Finite-field, symbolic physical and generic
external-consumer boundaries retain Q-0547's open status. Q-0548 evidence is reused.

## Schedule repair

A call forwarding a CBit through a one-operation allocation or release function
has a real classical edge despite zero quantum width. `component_schedule.py`
now classifies allocation only when the original operation has no inputs, and
release only when it has no outputs. Classical pass-through helpers are ordinary
components. Pure workspace helpers keep their classification. The independent
schedule regression checks both physical and logical schedules, classical values,
slot preservation, work events and complete binding.

## Numerical and replacement evidence

The stored child repeats a shared nested body twice. Each body allocates one
clean work bit, applies CX, phase(pi/4), CX and H, then releases work. Every
scalar positive, scalar negative and two-element mixed control assignment is
compared with the original wrapper direct tensor on a complex state entangled
with a spectator. Child wire streams, parameters and guard slots are compared.
Every final clean-work amplitude outside zero is below `4e-14`.

A second case obtains two owned flags from actual X-basis measurements and
attaches the returned control callback. Both outer polarities and all four
histories compare complete unnormalized complex amplitudes with independent
measurement evolution and original nested classical action. A third case keeps
an existing OR condition and compares all eight assignments under each outer
polarity with the existing flat controlled-IR implementation through the same
append consumer. Malformed shapes, nonbool values, missing bodies, non-THRU
entry and unsupported reachable effects fail before producing a transformed
record. Existing leaf construction is replaced without changing consumers.

## Observed resources and source estimates

| Witness | Observed bound primitives | Workspace | Separate source estimate |
| --- | --- | --- | --- |
| Two shared bodies, scalar control | 4 CX, 2 phase, 2 H, 1 classical input initializer | One reusable clean bit, four allocate/free events, zero measurements | Child model 2 T, 6 Clifford, zero measurements |
| Two shared bodies, two-element control | 4 CX, 2 phase, 2 H, 2 classical input initializers | Same reusable clean bit and four events | Same child model |
| Nested measured flags | 2 H, 2 measurements, 1 guarded X | Zero workspace | Numerical classical wrapper comparison only |

These are emitted instruction counts. Disabled guards retain their emitted gates.
The classical input initializer uses existing `classical_xor` instructions.
Typed read snapshots add zero primitive gates. The wrapper model raises
`DecomposeNotImplementedError`, which is recorded separately from the child
estimate. No wrapper resource estimate is inferred from the child model.

## Verification

All 13 focused cases, 4099 concentrated cases and 96 SDK integration cases pass.
The concentrated suite emits one existing upstream deprecation warning and SDK
integration emits two existing warnings. Strict mypy passes all five touched
Python files. Black, Ruff, Compiler scope, architecture and multi-level IR checks
pass. Dependency policy reports 687 existing repository findings and zero batch
findings. Whole-repository default-tier status is recorded in the batch log.

Persistent receipts are `ftqc-reference-env/g2-classical-graph-*`, including the
source blobs, Qualtran 0.7.0, Kickmix 0.1 with Simulator present, observed resources,
source model refusal, regression, integration and check outputs. Existing source
defects and broader control execution admission remain tracked in
`CONTROLLED_INTERFACE_SOURCE_AUDIT.md`.

The CBit-only interface and positive-shape descriptions above document the
original accepted baseline. Proposal 108 extends that baseline with explicit
`control_types` and zero-sized shapes, while the exact CBit graph and bound
serialization fixture remain unchanged. The typed-word audit is authoritative
for the current executable domain.
