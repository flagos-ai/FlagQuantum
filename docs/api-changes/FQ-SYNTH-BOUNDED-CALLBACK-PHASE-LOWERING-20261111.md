# Bounded callback phase lowering

The user authorized completing all 216 native quantum programs while retaining
the frozen source semantics and resource limits. This unreleased Compiler
extension reuses existing Python qfunc syntax, scalar controls, indexed phase
effects, exact word metadata and ordinary gate decomposition. No public symbol,
signature, serialized opcode/schema, execution backend or resource ceiling is
added. Release authorization remains false.

## Semantic contract

An initialized raw one-bit quantum Name compared with literal integer zero or
one using equality/inequality may control an existing mutable helper directly.
Reversed operands have the same meaning. The selector must have one actual bit
and raw integer metadata; Python bools do not become integer literals. A
negative writable selector may use checked X conjugation. Negative Const or
underlying-readonly selectors retain their existing computed path. The existing
condition/dependency borrowing rules still apply, including nested helpers.

Inside checked pure callback expansion, a bounded integer predicate-X can use
`H(target); predicate-phase(pi, target); H(target)`. This is XOR on an arbitrary
target, not preparation of a presumed-zero flag. It preserves absolute and
conditional scalar phase, entanglement and every nonselected basis value.
Conditions retain the existing eight-input-bit bound. Signed whole integer
words use their declared two's-complement interpretation; raw unsigned views
use their ordered actual bits. Numeric formats in aliases come from formal
bindings, not accidental matches with root register names.

Predicate dependencies must be live and initialized. The target must be
writable and disjoint from every dependency register before either H is
emitted. The extension does not relax conservative register-level alias
refusals, Const, ownership, Output escape or borrowing rules. Fixed-point,
dynamic, overly wide and unsupported numeric-bitwise conditions keep the
existing path or fail with their established checked boundary. No unsupported
condition may bypass validation merely because its phase cancels algebraically.

Raw modular increment/decrement ripple toggles inside callback expansion may
reuse the same H/controlled-phase/H identity instead of requiring a clean
multi-controlled-X ladder. Logical component ordering, modulo wraparound,
cross-component carry, zero-offset initialization witnesses and unchanged
surrounding bits retain their existing contracts. This uses existing-storage
unitaries and introduces no copied data register, fresh-zero assumption or
escaping scratch.

Adjoint, within/apply, nested coherent controls and pure QPE use the existing
effect verifier and controlled expansion. Existing source, call-depth, loop,
phase, operation, memory and incomplete-search limits remain authoritative.
Avoiding clean scratch can increase gate count and depth. This extension makes
no optimality or hand-tuned resource-parity claim.

## Acceptance

- Compare complete small basis domains, both nonzero/zero marker values and
  complex coherent states against independently constructed mathematical
  permutations and Circuit matrices. Include signed thresholds, reversed scalar
  comparisons, negative selectors, conditional global phase, adjoint,
  within/apply, nested controls and arbitrary numeric component order.
- Verify formal numeric metadata with differently named root registers and an
  otherwise partial borrowed helper. Check fallback boundaries for fixed-point
  and oversized parent views instead of changing their established semantics.
- Refuse Const writes, uninitialized reads, duplicated/overlapping storage,
  predicate/target dependencies, Output escape, dynamic views and actual
  operation, width and workspace budget violations. Preserve all existing caps.
- Run the complete native Vlasov-Ampere program with every original constant,
  signed velocity and reference register. Independently compare all 4194304
  logical complex amplitudes, including absolute phase, and measure aggregate
  dirty-work probability. Retain 24 physical bits, 2 clean work bits, 100000
  operations and 256 MiB planned-state allocation. Process RSS is measured
  separately and is not bounded by the planned-state limit.
- Deliver standalone public native qfunc and Circuit examples without importing
  Qmod sources, tests or synthesized gates into the user reference. Retain
  source identity/license/provenance in test fixtures and metadata. Report
  measured native/reference/original resource differences explicitly.
- Update the five-category catalog only after real production acceptance.
  Re-run all 216 original files and six approved revisions locally, preserve
  immutable original resource baselines, record changed outcomes honestly, and
  freeze the batch evidence and delivery ledger. Re-audit native216 acceptance
  before opening the single integration PR or running remote full CI.

The out-of-tree in-memory Vlasov prototype is exploratory evidence only. Its
24-qubit/2-work result and full-state comparison do not substitute for these
production guards, tests, user examples and final coverage audit.
