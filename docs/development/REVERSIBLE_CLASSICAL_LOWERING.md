# Atomic guarded parity and X-reset legalization

G2-KCFB02 extends the compiler-owned reversible-phase legalizer behind the
existing internal artifact boundary. It adds no Core signature, opcode or
schema. Core validation still establishes all source, predicate and conditional
destination definitions before transformation.

For one positive conjunction on classical_xor, push every guard before any
write. Native PUSH_CONDITION snapshots its active mask, so a destination that
aliases a guard cannot change the assignment's predicate during expansion.
If the old destination appears an odd number of times among sources, retain
it; otherwise conditionally clear it using BIT_INVERT target if target.
Never use scoped BIT_STORE0, which is still outside the admitted subset.
Then XOR each non-destination source and the literal in order, and pop all
guards. This has no new quantum work or declared native classical slots.
Unconditional assignment operation encoding remains byte-for-byte unchanged.
Unused classical holes are marked as unowned inputs in the existing optional
KMX tag so rebinding does not overwrite them; contiguous old declarations
retain the same header. Negative
or disjunctive predicates still fail explicitly in this reverse route.

Recognize only adjacent, matching H/measure/conditional-X and H/reset patterns
as source HMR and R. Preserve qubits, positive guards, result slots and marker
requirements. Mismatched predicates, malformed effects and unrelated H remain
unsupported. Existing CCZ conjugation and marked scalar-sign recognition keep
their source semantics. Float phases are not converted into invented exact
source angles.

The target code preserves assignment/channel semantics, not necessarily the
original source text after a Core binding: explicit initialization and saved
scope copies are real representation overhead. Report those separately from
direct source primitive counts. Acceptance includes exact native truth tables,
complex density/branch checks, invalid patterns and a full measured-unlookup
Core/artifact/KMX/native round trip. Typed ProgramIR conditional result merge
remains outside this compiler slice and explicitly refused.

## Local acceptance

The dedicated classical lowering module owns parity expansion and the shared
positive-predicate admission rule; the existing reversible legalizer composes
it with exact adjacent HMR/R recognition. Unused classical holes use the
existing input-slot declaration instead of being initialized on rebind.
Enclosing caller values and undefined unused local slots are preserved.

Native truth tables cover 90 assignment configurations, including 720 distinct
input/alias/predicate cases, with no quantum operands or added declared native
bits. Sixteen full measured-unlookup basis scenarios preserve outputs and
exact Fraction phases through the reverse route. Eight guarded channel cases
independently match the unchanged native executor's seeded quantum/classical
outputs and exact phases, including live-guard/result aliasing.

Every small guarded HMR/reset branch is compared as an unnormalized complex
vector in active and inactive cases, alongside density ensembles. Inactive
potential measurement outcomes use identity label zero and impossible label
one, so enumerating the complete instrument preserves total probability.
Full one/two-address-bit compute/unlookup consumers also match every complex
branch after Core/artifact/KMX/local rebinding. Quantum instruction names,
operands, measurement/reset counts, seeded final states and all original
classical output slots are unchanged. Saved scope copies and initialization
in the converted representation are reported as classical overhead.

The existing `python -m examples.algorithms.measured_unlookup` now verifies
that reverse route through the same `circuit.append(op)` consumer, and prints
direct source, converted target and emitted IR counts separately.

Acceptance: 140 new tests; 1,040 concentrated regressions plus 366 disjoint
finite-field regressions passed, for 1,406 passed and four optional Qiskit/Aer
skips. Strict mypy passes all three touched production/example files. Black,
Ruff, compiler scope, architecture, diff checks and the public API migration
baseline pass; no API snapshot was regenerated. Original source fixtures and
reference sources remain unchanged.

Negative/disjunctive reverse predicates, parameterized float phases and typed
ProgramIR conditional merges remain explicit boundaries. This is not a G2
completion claim and does not alter protected ProgramIR vocabulary.


G2-KCFB03 [negative predicate binding](REVERSIBLE_NEGATIVE_PREDICATES.md)
now admits mixed zero/one conjunctions through explicit owned classical
complements. Earlier negative-predicate refusals record the prior baseline.
Disjunctions, typed conditional result merges and general float phase recovery
remain explicit boundaries; positive encoding and public schemas are unchanged.
