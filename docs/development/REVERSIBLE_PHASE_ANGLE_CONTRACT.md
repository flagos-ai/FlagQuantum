# Exact conditional phase contract

This internal compiler contract extends the existing reversible-phase artifact,
its KMX codec and local execution binding. Stable Core instruction schemas,
public APIs, positive ambient circuit widths and existing KMX text stay intact.

`z_pow` owns one quantum operand, one `PhaseAngle128` angle and no classical
result. The angle is required on this opcode and forbidden on every other
opcode. Its optional classical predicate and enclosing saved scopes use the
existing definition/lifetime rules. A zero angle remains a source operation.

KMX spells the instruction `Z_POW qN DECIMAL` followed by an optional `if bN`.
The decimal is in half turns and follows the locked source parser/renderer.
Exact words survive artifact/codec round trips, including tiny negative wrapped
angles. Existing operations omit the new field from their unchanged text.

Binding converts the centered half-turn representative only at the existing
Core `phase(theta)` instruction boundary. This is binary64 complex simulation,
not 128-bit numerical execution. The artifact remains the exact source-count
view. A replacement binding must consume the same exact operation and guards
without changing the circuit consumer. General floating Core phases are not
silently reconstructed as exact source angles during reverse lowering.

Acceptance includes invalid angle placement, undefined guards, malformed KMX,
zero-angle retention, exact word round trips, native KMX phase behavior,
coherent and entangled conditional execution, and replacement of the phase
binding with an independently equivalent instruction. No extra quantum work
is allocated; existing classical scope snapshots remain separate resources.

The QFT consumer is accepted separately. An empty logical target produces an
empty operation sequence that can be embedded in an existing positive-width
circuit. It does not require changing the protected zero-width Core contract.

Local acceptance: 56 new tests and 1,132 concentrated regression tests pass;
four optional backend cases are skipped. Native Kickmix confirms the encoded
phase and guards, and a Z replacement confirms the unchanged consumer boundary.
Exact zero and smallest wrapped angles survive the codec. Source opcode counts
and numerical Core phase instructions are distinct; no T synthesis is selected.
