# Source graph register decomposition

The immutable reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Ecosystem owns optional source
objects and their conversion. Compiler continues owning typed records, physical
binding, feedback and output-order restoration.

## Interface and complete module responsibility

The new private source-graph owner accepts a Cirq operation and the existing
allocation manager. It returns operations from the actual flattened source
CompositeBloq for concrete THRU ports. Atomic or unavailable decompositions and
non-THRU components decline this path. Source errors propagate. CompositeBloq
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
