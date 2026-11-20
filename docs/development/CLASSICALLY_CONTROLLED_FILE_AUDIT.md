# Classically controlled component audit

The declaration and its original test are pinned to Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`. The native Compiler
register owner prepends classical control metadata, preserves declaration
order, and resolves name collisions with the same numbering. When a
child already owns `ctrl` and `ctrl2`, the two added control names
are `ctrl3` and `ctrl4`. Shapes and classical data types are retained.
Quantum control metadata is rejected by this entry.

The native typed feedback function applies the unchanged bit-gate
lowerer for a coherent child. Each CBit input is read through an existing
owned snapshot operation and its successor stays in the same physical
classical slot. Conditions are attached to primitive gates in their
original order. The attachment uses ComponentProgramBuilder and
returns control successors in control declaration order, followed by
the child's ordered outputs. The existing one-bit CZ consumer uses
the same constructor. Register shape, execution admission and
classical-slot validation remain with the existing owners.

## Test-file acceptance

Q-0548 closes. The pinned source test passes and all four of its
scalar truth-table inputs run through append with the source control
value forwarded. The shaped test builds the original attachment
callback and its native counterpart with two CBit array entries and
one scalar CBit. A three-measurement native program evaluates all
eight histories against the corresponding locked classical action
for every computational child input. Every unnormalized complex
amplitude is compared. The emitted child is one X with an ordered
three-slot guard, with no extra primitive for control reads and no
quantum scratch. Both child wires and each classical output slot
retain their bindings.

The prior source tensor, phase, and measurement-branch work is reused,
as are the measurement-uncomputation and field-erasure consumers.
The concentrated regression passes 529 tests.

## Remaining implementation boundary

Q-0547 remains open for typed Program execution with wider classical
data types. Wider-word metadata and existing flat IR guards remain
available. The current typed feedback read accepts owned CBit
elements. The leaf adapter retains its original admission. Stored CBit child graphs now
forward controls through calls and conjoin existing DNF guards. Complete
complex-state, measurement-branch and clean-work evidence is recorded in
`CLASSICAL_COMPONENT_CONTROL_AUDIT.md`. Wider-word typed reads remain open.
The separate flat IR path also composes existing guards. The flat IR rejects conditional
measurement, reset, and classical writes.

## Current dependency review on 2026-11-20

The wider-word read boundary above is historical. Proposals 108, 110 and 111
now cover admitted integer, binary-field and nonbinary-field word reads and
equality feedback, including scalar, shaped and empty bindings. Their accepted
audits retain nominal types, immutable snapshots and existing input range checks.
The complete register-gate interface is accepted at `b18de0d84`.

A separate remaining execution gap is conditional mutation of a child-owned
classical port. The unchanged `ClassicallyControlled` delegates to
`_ControlledBase.on_classical_vals`, which returns child updates on activation
and preserves the original control values. The CBit flip intake at
`ftqc-reference-env/g2-classical-write-intake.json` compares both polarities and
all eight control/data cases. The flat wrapper currently rejects the existing
classical XOR child. Existing guarded XOR execution is already available under
`CLASSICAL_GUARDED_DATAFLOW_CONTRACT.md`; the wrapper admission and typed owned
word successor are the missing integration boundaries.

Proposal 112 records those additional protected semantics and was explicitly
approved. The acceptance below supersedes the intake refusal for owned writes. Q-0547 and the
complete inherited control file remain open. The original wrapper still lacks
a physical decomposition and root resource model, and the nonbinary polynomial
tensor index discrepancies remain independently recorded in Proposal 111.

## Complete-file acceptance with Proposal 112

Proposal 112 was explicitly approved on 2026-11-20. The complete source file has
three definitions. Their native owners and direct dependency evidence are now
closed for its concrete execution scope.

| Source definition | Native owner and evidence |
| --- | --- |
| `ClassicallyControlled` | Existing immutable Program, Signature and classical control transform retain child declaration, ordered controls and port identity. Accepted CBit/integer/field audits cover scalar, shaped and empty arrays. |
| `ClassicallyControlled.__attrs_post_init__` | `classical_control_signature` and `executable_classical_register_width` validate classical neutral descriptors. Existing original invalid-QBit and dtype tests remain authoritative. |
| `ClassicallyControlled.make_ctrl_system` | `call_classical_feedback` and `ComponentProgramBuilder` compose the actual stored callback path. Existing collision/attachment/all-eight-measurement histories are reused. |

The inherited numerical callback and basis-phase behavior retain Simulation
ownership and existing complete source comparisons. The inherited tensor path
retains its accepted source implementation and documented ones'-complement and
nonbinary polynomial counterexamples. Generic `controlled.py` protocol ownership
and unresolved symbolic physical admission remain with Q-0158, outside these
three source definitions. The original wrapper supplies no decomposition or
root QEC cost model. That absence is retained explicitly.

### Owned write integration

Core verifies `write_classical_register` as one owned word and its concrete
width of MSB Boolean operands, returning one identical-type successor. Existing
linearity, dominance, metadata and canonical DNF checks apply. The approved
contract is `contracts/classical-word-write-contract.toml`.

Compiler binds an unconditional replacement to its immutable source slots.
A conditional replacement allocates distinct successor slots, copies the old
word, and assigns replacement snapshots through existing guarded XOR records.
Overlapping satisfied clauses assign the same stable snapshots. Physically
aliased contradictory conditions leave the copy unchanged. Call expansion
forwards only the old owned word when all remapped clauses are contradictory.
The stored feedback transform already conditions all nonstructural operations,
so it admits the new verified write without a second transform or interpreter.
Flat conditional XOR children retain source order and operands and reject
writes into wrapper control slots before execution.

Tests retain raw internal unused GF(3) and bounded-integer patterns, strict
external admission, and numerical field codec refusal. They also exercise every
admitted word family, both polarities, sequential/nested writes, row-major arrays,
empty arrays, zero width, old/new snapshot quantum feedback, overlapping DNF,
call-local aliases, complete complex spectator states and clean workspace.
The two-measurement composition exhausts all four unnormalized complex branches
and restores its reused work wire to zero. Replacing both the flat wrapper and
numerical callback with raising stubs preserves the stored append path.

### Resource scope and validation

For width w, an unconditional complement body emits w input initializers, one
Boolean-one seed and w Boolean XORs. Its owned write adds zero gates or slots.
A single-clause conditional write adds w copies and w guarded assignments and
w fresh successor slots. The tested complement therefore emits `1 + 2*w`
assignments unconditionally or `1 + 4*w` conditionally. A prepended CBit control
adds one initializer. Quantum gates, measurement and workspace are zero for
these write-only examples. The two-measurement example separately emits two
measurements, two resets and one conditional Z, with one reused clean work wire.
Original wrapper model/decomposition absence stays separate from those observed
primitive counts.

Evidence is recorded under `ftqc-reference-env/g2-classical-write-*`.
The concentrated suite, original source test, strict reference-aware typing,
formatting, Integration scope, architecture, IR and public API baseline checks
are required before publication. Existing exact CBit/integer/binary/nonbinary
artifacts are part of the regression. The known default docstring coverage
failure is checked against the pre-batch files and reported independently.
No new parent acceptance is inferred for `controlled.py`.

Final local acceptance passes 4651 concentrated checks with 30 existing paired
source/native overflow warnings, 17 integration checks and the unchanged original
source test. Eight files pass strict typing with the frozen source on MYPYPATH,
Black/Ruff, Integration scope, architecture, IR and API checks. The three source
blobs, Qualtran 0.7.0 and native Kickmix 0.1 Simulator are verified. Dependency
inspection has 687 pre-existing repository findings and zero batch findings.
Default smoke/unit stops at the known docstring coverage failure after 434 passes
and 51 skips. All nine reported modules and its test are byte-identical to the
pre-batch `b18de0d84` files. The batch accepts Q-0547. Q-0158 remains open.
