# Source graph register decomposition

The immutable reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Ecosystem owns optional source
objects and their conversion. Compiler continues owning typed records, physical
binding, feedback and output-order restoration.

## Interface and complete module responsibility

The new private source-graph owner accepts a Cirq operation and the existing
allocation manager. It returns operations from the actual source CompositeBloq for concrete THRU
ports. Flattening preserves concrete GateWithRegisters SDK bodies. Generic
controlled wrappers still expand through their typed graph so control synthesis
does not cross measured children. Atomic or unavailable decompositions, concrete
SDK gates and non-THRU components decline this path. Source errors propagate. CompositeBloq
objects enter directly rather than calling their prohibited decomposition method.
Register order, shaped ports and MSB bit order come from the original signature.

When the actual graph returns different physical aliases, the existing Compiler
permutation helper restores caller positions across both logical and temporary
wires. All returned aliases must be distinct and have the original total width.
Output alias swaps are physical binding overhead. Source logical wire changes
remain separate from those primitives. The existing recursive importer retains
its declared-definition precedence, depth/cycle guard, live-frame reservations,
clean-work restriction and unified measurement history.

## Evidence

Actual ModNeg graphs at widths 2 and 3, moduli 1, 2, 3 and 5 import through the
normal register optree entry and append on reversed physical wires. All valid
residues match the actual original classical method. The conjunction channel
certificate checks every erasure history, including relative complex amplitudes
and zero work. Three independently sampled sparse trajectories additionally
compare the complete support of phase-sensitive superpositions. Numerical
roundoff outside expected support is checked against the same 2e-12 tolerance.

The original QEC model predicts `3*(n-1)` measurements and the emitted T/T-adjoint
count. Actual native counts are checked independently. Model peak scheduling and
physical caller workspace retain their distinct ownership. The graph-only
permutation fixture returns exchanged output aliases and verifies the physical
SWAP unitary. Atomic SDK gates correctly decline graph expansion.

The full existing Cirq register import, control, graph, leaf and export suites
exercise replacement, explicit definitions, feedback and borrowed lifetime
behavior. Existing ModNeg source primitive comparisons remain authoritative for
exact gate operands and dependency order, independently of this new consumer.

## Actual ECAdd boundary

The unmodified direct graph serializer still rejects NumPy uint8 constructor
arguments in stages one, two and six. Stages three through five retain distinct
serialized-declaration collisions. These actual intakes are recorded separately.
The new graph-expansion path bypasses serialization through the source's live
CompositeBloq export, without altering its data or callback semantics.

Within the controlled window consumer, the original stage-one constructor child
now imports successfully. The enclosing controlled stage-one node still reaches
the original Controlled(And) gap and the SDK unitary fallback recurses. Complete
controlled ECAdd execution and parent acceptance remain open. This dependency
supplies no claimed primitive implementation for an unavailable original child.
Receipts include `g2-ec-stage-graph-intake-20261121.json` and
`g2-source-graph-flattened-followup-20261121.json` in the reference directory.

## Concrete SDK gate boundary closure, November 22

The complete private source-graph expansion owner now retains the locked
source's concrete GateWithRegisters bodies during recursive flattening. The
same predicate declines a concrete SDK gate at the importer boundary, allowing
its existing direct decomposition to execute. Generic _ControlledBase wrappers
remain graph-expanded. Their SDK control wrapper can otherwise attempt a
control over And erasure, whose source explicitly has no unitary. Existing
physical-alias restoration, live-frame reservation, declared native precedence,
clean workspace checks and source error propagation retain their owners.

This closes the three controlled HasDuplicates SDK cases at widths 1, 2 and 3
and the four PermutationCycle/Permutation SDK cases at widths 2 and 3. The seven SDK success
tests and reference source remain unchanged. Tests independently compare every
SDK leaf's operands, wire dependency order or primitive matrix, measurement
feedback, full small complex amplitudes, spectators and clean work. The existing
native component route continues preserving the source typed graph's measured
And decomposition and separate analytical models.

A temporary no-graph probe passed the seven SDK cases but was insufficient to
preserve graph-only consumers. A probe preserving all GateWithRegisters nodes
exposed four ModNeg failures at controlled And erasure. The accepted predicate
retains graph expansion for generic control wrappers and passes those existing
ModNeg complete-residue, relative-phase and work-cleanup checks. These failed
probes were temporary runtime substitutions. Production uses the predicate
validated against the full related import regression.

SDK primitive histograms and independent source model fields are recorded in
g2-sdk-gate-boundary-duplicates-resources-20261122.json. Permutation SDK receipts
retain atomic CCX/matrix accounting separately from the native typed-component
primitive/model resource evidence. This boundary correction introduces no
public operation, parameter, schema or backend selection.

The broader regression also exposed four stale negative-test expectations in
Negate and negative-control Subtract/SubtractFrom SDK import. Replaying the
published pre-change source-graph owner reproduces all four failures. The native
importer's existing explicit no-unitary guard raises ValueError, while original
Cirq unitary evaluation still raises RecursionError. The tests now check these
two distinct existing outcomes precisely. No production refusal changed. The
baseline receipt is g2-sdk-boundary-baseline-negative-20261122.log.

Final related importer regression passes 440 cases on Cirq 1.7, with 100 existing
source serialization warnings and zero skips. The independent Cirq 1.6.1 lane
passes 63 cases with 174 upstream warnings. Its initial startup encountered a
missing platformdirs pytest plugin. The rerun disables that unrelated plugin,
matching the existing reference-runner configuration. Strict typing of the three
changed Python files, Black, Ruff, Ecosystem scope, architecture, IR, public API
and whitespace checks pass.

Controlled width-one HasDuplicates SDK execution emits four T-family gates,
one measurement and one atomic CCX. Its separate source typed model declares
two And computations and two measurements. Permutation SDK instruction totals
are 15/28 at width two and 120/238 at width three for cycle/composition.
Zero-cycle identity emits zero instructions. These SDK totals retain atomic
CCX and matrix leaves and remain separate from primitive decompositions of
native typed components.
