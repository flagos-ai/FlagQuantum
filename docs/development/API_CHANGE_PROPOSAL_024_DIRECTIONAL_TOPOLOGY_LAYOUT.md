# API Change Proposal 024: Directional topology and explicit layout evidence

## Status

**Approved on 2026-09-10; implementation and evidence handoff complete.**

Date: 2026-09-10

Approval token:
`approve API_CHANGE_PROPOSAL_024_DIRECTIONAL_TOPOLOGY_LAYOUT`

The API owner supplied the exact token on 2026-09-10. The first bounded
implementation adds the private directed topology model, explicit placement,
direction-correct CX legalization, final legality verification, physical-plan
2.0 lineage, artifact-to-artifact compilation, strict Core evidence 2.0,
explicit version dispatch, and Compiler construction/verification. The
Runtime now verifies version 2.0 against the actual source, snapshot, and
executable artifact, and Deployment carries only that verified evidence in its
side-effect-free dry run.

## Problem

The current vNext compiler can route a `CircuitIR` through an undirected
`CouplingMap`, restore the logical output order, and serialize the resulting
physical plan in `CompilationEvidenceBundle` 1.0. That contract deliberately
fixes the initial and final layouts to identity and explicitly excludes
directed-edge synthesis.

The removed private multi-level-IR compiler previously proved a narrower but
important capability: deterministic routing on directed CX graphs, explicit
non-identity logical-to-physical placement, direction-correct reverse-CX
synthesis, final restoration to identity, state equivalence, and preservation
of trainable gradients. Reintroducing those semantics by silently changing the
public Compiler `CouplingMap` would reinterpret existing callers and invalidate
the identities and claims of the approved evidence-bundle 1.0 schema.

## Decision

Migrate the proven semantics into the current single-`CircuitIR` compilation
chain through a new, initially private `DirectedCouplingMap` and a distinct
direction-legalization stage. Keep `CouplingMap` exactly undirected. Extend the
separate compilation-evidence schema as version 2.0; do not change
`ProgramArtifact` v1 or v2 and do not create a TargetIR.

The initial implementation supports only equal logical and physical capacity.
An explicit initial layout is a complete permutation of
`[0, ..., CircuitIR.n_wires - 1]`. Routing must restore the final logical layout
to identity before emission, so existing result-wire semantics remain valid.
Unused physical qubits, physical ancillas, partial layouts, observable
remapping, and calibration-weighted placement remain excluded.

## Compiler model

`DirectedCouplingMap` records a positive physical-wire count and canonical,
unique ordered edges `(control, target)`. A directed edge authorizes a native
CX only in that order. Weak connectivity may be used for deterministic path
selection, but it never authorizes operand reversal.

The ordered target path becomes:

```text
source CircuitIR
  -> explicit initial placement and topology routing
  -> native-gate legalization
  -> directed two-qubit legalization
  -> final native and directional legality check
  -> dependency schedule
  -> deterministic emission and conformance
```

Direction legalization may preserve a requested `cx(control, target)` over a
reverse-only physical edge using the exact identity
`H(control) H(target) CX(target, control) H(control) H(target)`. It may do so
only when the emitted `h` and `cx` gates are in the target native-gate set. If
neither the requested nor reverse edge exists, or if the required native basis
is absent, compilation fails closed. Other asymmetric two-qubit operations are
not repaired by exchanging operands and remain unsupported in version 2.0.

Every generated instruction retains the original source-instruction index,
the routed/topology instruction index, native replacement ordinal, and
direction replacement ordinal. The direction stage records its deterministic
identity and reversed-CX count. A final verifier proves that every emitted CX
uses an authorized ordered edge and every emitted operation belongs to the
target native set.

## Placement semantics

`initial_logical_to_physical[L] = P` means logical wire `L` is initially placed
on physical wire `P`. Version 2.0 requires a full permutation and equal logical
and physical counts. Circuits begin from the existing standard all-zero
initial state; arbitrary externally supplied physical input states are not
claimed.

Mapping transitions begin at the declared initial layout, record every routing
SWAP, and replay to the declared pre-restore and final layouts. The final layout
must be identity and `mapping_restored` must be true. A non-identity initial
layout is therefore a compiler placement choice, not a change to public
measurement or result ordering.

## Evidence-bundle 2.0

Version 2.0 keeps the 1.0 top-level envelope and source, target, and output
records. Its `physical_plan` adds:

```text
direction_legalization_identity
reversed_cx_count
```

Its coupling record is:

```text
n_wires
directed_edges
direction_semantics = "directed_cx"
```

Each final instruction record adds:

```text
native_instruction_index
direction_replacement_ordinal
direction_rewrite = "none" | "reverse_cx_h_conjugation"
```

The plan and bundle identities cover all added fields. Version 1.0 remains
strictly readable and retains its undirected, identity-layout meaning. A 1.0
object is never promoted to 2.0 by inference. Version dispatch is explicit and
unknown versions fail closed.

## Compatibility and ownership

- `CouplingMap` behavior, constructor, identities, and exports do not change.
- `CompilationEvidenceBundle` 1.0 bytes, reader behavior, and identity do not
  change.
- `ProgramArtifact` v1 and v2 do not change.
- Core owns strict 2.0 evidence values, canonical encoding, limits, and version
  dispatch.
- Compiler owns directed topology, placement, rewrites, final legality, and
  construction from actual compilation objects.
- Runtime and Deployment only verify and carry 2.0 evidence beside the actual
  artifacts; they do not repair layouts or reinterpret instructions.
- No public root export or default-path change is included.

## Reuse and non-reuse of the historical implementation

The old Batch C implementation is a semantic oracle, not a tree to restore.
The migration reuses its directed-graph invariants, lexicographically stable
paths, reverse-CX identity, final-layout restoration, and differential tests.
It does not restore the removed `_compiler` IR, SSA value model, pass manager,
exporter, or parallel runtime authority.

## Limits and excluded claims

Version 2.0 retains the 1.0 size, nesting, instruction, transition,
predecessor, string, duplicate-key, finite-number, and sensitive-content
limits. It does not claim:

- more physical than logical wires or physical ancilla allocation;
- partial placement, live-range allocation, or observable remapping;
- arbitrary directed two-qubit-gate synthesis;
- calibration/error-aware routing, gate duration, pulse scheduling, or
  crosstalk optimization;
- numerical correctness beyond the bounded differential test corpus;
- QPU correctness, fidelity, capacity, latency, or performance;
- fault-tolerant logical-qubit expansion or QEC scheduling.

## Implementation gates

1. **Complete:** record this proposal and the exact machine-readable candidate.
2. **Complete:** receive the exact approval token from the API owner.
3. **Complete:** add `DirectedCouplingMap` without changing `CouplingMap` behavior.
4. **Complete:** port placement, routing, direction legalization, lineage, and final legality
   into the current `CircuitIR` target pipeline.
5. **Complete:** extend physical-plan validation and deterministic identity without adding a
   second instruction authority.
6. **Complete:** implement strict Core evidence version 2.0 and retain pinned 1.0 fixtures.
7. **Complete:** add Runtime and Deployment verification in a separate bounded handoff.
8. Keep all new entry points non-root until naming and lifecycle review.

## Acceptance

- All permutations and ordered CX pairs on a three-wire asymmetric topology
  preserve state up to global phase and restore logical output order.
- Reverse-only CX is synthesized without silently swapping operands.
- Direct-only, reverse-only, disconnected, invalid-layout, capacity-mismatch,
  missing-native-basis, tamper, and version-dispatch cases are covered.
- Trainable parameter values and gradients match the source for the bounded
  statevector oracle corpus.
- Every final CX is legal on its directed edge and every final instruction is
  target-native.
- Mapping and instruction lineage replay from the actual source program.
- Identities and canonical JSON are deterministic across hash seeds.
- Evidence 1.0 fixtures and identities, ProgramArtifact v1/v2 fixtures, public
  exports, and the default execution path are unchanged.

## Approval

The API owner approved this proposal with the exact token on 2026-09-10. The
approval authorizes the gated implementation above; it does not authorize a
public root export, default-path change, physical-ancilla support, or any of the
excluded hardware claims.
