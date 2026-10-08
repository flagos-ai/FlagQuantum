# IR-007: Phase 4 hybrid and dynamic scope

Status: Approved
Date: 2026-10-06
Ratified: 2026-10-08 by the repository owner, together with the Phase 4 activation
in [`API_CHANGE_PROPOSAL_062`](../../development/API_CHANGE_PROPOSAL_062_MULTI_LEVEL_IR_PHASE_4_ACTIVATION.md).
That proposal admits the internal level structure and names this document as the
ratified scope, so the two carry one decision rather than two.
Applicable phase: Phase 4 functions, classical control, and dynamic migration.
Related: [`API_CHANGE_PROPOSAL_062`](../../development/API_CHANGE_PROPOSAL_062_DYNAMIC_CIRCUIT_PROMOTION.md),
[`ARCH-012`](ARCH_012_CUDAQ_PARITY_CONTROL_SEQUENCE.md).

## Background

`MULTI_LEVEL_IR_ARCHITECTURE.md` § 13 schedules Phase 4 as functions, scopes,
blocks, classical control, measurement def-use, reset, conditionals, bounded
loops, partial evaluation, `ProgramIR` lowering, QASM Static/Adaptive, and dynamic
migration, with the exit condition "shared static/dynamic semantics without
batched regression". § 5.2 and § 5.3 specify the intended levels, and § 19.13
fixes their internal layout. `IR-001` through `IR-006` are approved, but they
cover the importer, linear values, identity layers, custom matrices, parameter
binding, and the Phase 1 static profile only. Phase 4 has no ratified scope.

Without a ratified scope, three failure modes are open. First, "Phase 4" can be
read as license to build a general-purpose classical compiler, which
`MULTI_LEVEL_IR_ARCHITECTURE.md` § 4 and § 5.3.1 explicitly bound against —
"bounded SSA without a general-language compiler". Second, the boundary between
static structure fixed early and numbers bound late (§ 6.3) can be stretched case
by case until no pass can declare what it preserves. Third, dynamic semantics can
leak into the public `CircuitIR` schema, which § 5.1 and `IR_006` both forbid.

The repository has already recorded the underlying gap from the
interoperability direction. `docs/development/CUDAQ_EXPORT_CONTRACT.md` states
that arbitrary hybrid kernel import "would either lose semantics or require
FlagQuantum IR features that do not exist", which is why the CUDA-Q adapter is
one-way by design. Phase 4 is the work that would make those features exist.

## Decision

### 1. Scope admitted

Phase 4 admits the following into the internal levels. The admission is settled by
this ADR; the layout is `flagquantum/core/ir/`, the module that already owns the
public IR, for the reason recorded under Decision 11.

- `ProgramModule` with functions, typed arguments and results, scopes, and calls;
- blocks, branches, bounded loops, and function terminators;
- classical values, including values produced by measurement and consumed by
  later operations, with explicit def-use;
- reset and mid-circuit measurement as typed operations;
- `QuantumModule` with qubit/register/result/angle/parameter/observable types,
  linear values, and the ownership rules of § 5.3.1;
- `TargetModule` with layout, native gates, and no unresolved parameters;
- partial evaluation and `ProgramModule`-to-`QuantumModule` lowering;
- a verifier with typed diagnostics for each internal level.

### 2. Scope rejected

Phase 4 rejects:

- a general-purpose classical language: recursion, unbounded loops, dynamic
  memory, exceptions, closures, and higher-order values are out of scope;
- timing, pulses, calibration, and stretches, which remain Phase 5;
- provider-native instruction sets;
- `fq.AncillaQubit` or any compiler-managed allocation surface, which § 5.3.2
  defers until after at least two release cycles with interoperability evidence
  through a separate proposal;
- any change to `IR_VERSION` or to `CircuitIR` schema `1.0`;
- any root-level export of `ProgramModule`, `QuantumModule`, `TargetModule`,
  `PassManager`, `AnalysisManager`, or `ExecutableArtifact`;
- a text format for the internal levels as a compatibility surface. § 19.13
  marks textual internal IR as debug only, with no cross-version promise.

### 3. Fixed-early and bound-late boundary

Following § 6.3, Phase 4 evaluates early: constants, static loop bounds and
indices, dead branches, structural configuration, and target-independent duration
expressions. It binds late: gate angles left as parameters, shots, backend choice,
and target selection. A construct that cannot be assigned to one side is rejected
rather than being deferred to a later pass by convention.

### 4. Preservation is declared, not assumed

Every Phase 4 pass declares which of the following it preserves, following § 6.1:
state or density semantics, measurement distributions, expectations, gradients,
approximation budget, operation ordering, and feedback semantics. Silent
reordering across a measurement-dependent branch is a correctness defect, not an
optimization.

### 5. Failure taxonomy extends `IR-006`

The `IR-006` rule stands and is extended to the new levels: exactly one status per
input, drawn from `supported_exact`, `unsupported_with_diagnostics`, and
`invalid_input`. There is no `best_effort` mode, no implicit metadata omission, and
no default lossy conversion. A rich internal structure that cannot be represented
losslessly in `CircuitIR` schema `1.0` is rejected rather than flattened.

### 6. Contract first, then implementation

The internal level boundary, value identity and linearity rules, join rules,
failure categories, and round-trip guarantees land on the integration branch as a
versioned internal contract with a contract fake and a conformance test before
team implementation begins. This follows `docs/development/MULTI_TEAM_DEVELOPMENT.md`.

### 7. Public promotion is separate and narrow

The only public surface Phase 4 may affect is the candidate-stable
`flagquantum.dynamic` namespace, whose promotion is governed by
[`API_CHANGE_PROPOSAL_062`](../../development/API_CHANGE_PROPOSAL_062_DYNAMIC_CIRCUIT_PROMOTION.md)
rather than by this ADR. `run_dynamic`, providers, dialects, and native results
remain experimental.

### 8. Local fast paths are preserved

Phase 4 must not slow an ordinary static circuit or require an ordinary user to
understand levels, values, ownership, or verification. § 4 already requires local
execution to be able to emit at the quantum level and bypass the rest; Phase 4
keeps that bypass and treats a regression on the recorded local baseline as a
failed acceptance item.

## Rejected alternatives

- **Ratify Phase 4 implicitly by starting implementation.** Rejected: the scope
  question would be re-litigated per pass, and § 6.3's boundary would erode.
- **Adopt a general-purpose classical IR or compile from Python's AST directly.**
  Rejected: `MULTI_LEVEL_IR_ARCHITECTURE.md` § 5.3.1 admits bounded SSA and
  explicitly not a general-language compiler; the additional machinery buys no
  current capability.
- **Extend public `CircuitIR` schema to carry dynamic semantics.** Rejected by
  § 5.1, by `IR_006`, and by the compatibility promise; it would also break
  consumers for no gain.
- **Place the internal levels in a new internal home of their own.** Rejected, and
  this is the decision that the layout in Decision 11 records. The design sketch in
  `MULTI_LEVEL_IR_ARCHITECTURE.md` § 19.13 named `flagquantum/_compiler/**`, and
  that name cannot be created in this tree: measured, creating
  `flagquantum/_compiler/__init__.py` fails `tools/check_architecture.py` with
  `flagquantum/_compiler: unreviewed top-level package directory is forbidden`,
  and one `from .._compiler import ...` line under `flagquantum/compiler/` fails
  the same gate with `compiler imports forbidden layer _compiler`, because
  `boundaries.compiler_forbidden` lists that name. `team-ownership.toml` records
  it among five patterns removed for matching no tracked path. A third internal
  IR home beside the public one would also be the parallel scaffolding that
  `ARCH-012` clause 2 prohibits, so the levels go in the package that already
  owns the IR rather than in a second location.
- **Land the internal levels before the contract and conformance test.** Rejected:
  Phase 4 spans Core, Compiler, and Runtime, and `MULTI_TEAM_DEVELOPMENT.md`
  forbids teams writing against each other's unfinished surfaces.

## Compatibility and rollback

`IR_VERSION` remains `"1.0"`. `CircuitIR` schema `1.0` remains the only public
serialized IR. `docs/public_api_v1.json` and the public API contracts are
unaffected, and no root export is added.

Rollback is removal of the internal level modules and the Phase 4 passes. Because
no public schema changed, rollback requires no migration and no deprecation cycle.
A public promotion made under `API_CHANGE_PROPOSAL_062` follows the standard
deprecation process instead.

## Protected-path question, verified

An earlier draft of this ADR recorded a suspected defect: that
`team-ownership.toml` protected `flagquantum/core/ir/**`, a glob that cannot match
the sibling module `flagquantum/core/ir.py`, leaving the public IR module outside
the protected-path mechanism. That suspicion has been checked against the tree and
is **not** a defect for the tree as it stood on 2026-10-06. It is recorded here
because the earlier draft asked for the pattern to be verified, and the
verification is the finding.

Three checks were run.

1. The entry was `flagquantum/core/ir.py`, a literal repository-relative path, not
   `flagquantum/core/ir/**`. The `**` form appeared nowhere in `protected_paths`.
2. `tools/check_team_scope.py` is present in this checkout and matches protected
   paths with `fnmatch.fnmatchcase`. Under that matcher `flagquantum/core/ir.py`
   matched the literal entry and `flagquantum/core/ir/**` would not have, which is
   why the distinction matters. The literal form was the correct one for a flat
   module.
3. The mechanism is enforced, not merely declared. `check_team_scope.py --team core
   --files flagquantum/core/ir.py` exited non-zero with `protected integration
   surface; submit a contract/ADR change`.

`protected_path_errors()` additionally fails `--validate` when a protected entry
matches no file in the worktree, so the entry cannot silently become stale. The
sibling glob rule remains a real trap for any future entry of that shape; it is
recorded here as guidance rather than as an open defect.

**Correction, 2026-10-08.** The tree changed, so the conclusion did. The public IR
is now the package `flagquantum/core/ir/`, and the protected entry is the glob,
because a module and its same-named sibling directory cannot both be protected by
one spelling. This is the same matcher finding read in the other direction: the
glob matches every file of the package and no sibling module, and the literal path
now matches nothing at all, which `--validate` would reject. The readings above are
kept as the record of 2026-10-06 and are not restated.

## Layout decision

The internal levels live in `flagquantum/core/ir/**`, the package that already owns
the public IR, and not under `flagquantum/_compiler/**` as
`MULTI_LEVEL_IR_ARCHITECTURE.md` § 19.13 sketched. § 19.13's name cannot be created
in this tree: `flagquantum/_compiler/` fails `tools/check_architecture.py` as an
unreviewed top-level package directory, importing it from `flagquantum/compiler/`
fails the same gate as a forbidden layer, and `team-ownership.toml` records the
name among five patterns removed for matching no tracked path. Landing there would
mean amending two policy files to revive a name the repository deliberately
deleted, which is a separate change and not part of Phase 4. The consequence for
the levels is only their filesystem location: the levels, ownership rules,
verifier taxonomy, and pass contract in Decisions 1-8 are unchanged.

## Acceptance

- [ ] Internal level contract, contract fake, and conformance test merged on the
      integration branch before team implementation begins.
- [ ] Supported static circuits round-trip to canonical `CircuitIR` payloads
      exactly; dynamic, unknown, and lossy inputs fail closed with typed
      diagnostics under the `IR_006` status taxonomy.
- [ ] The verifier rejects stale values after gates, duplicate consumers, use after
      release, asymmetric allocation without valid joins, merging different
      resources, and ambiguous function return or capture ownership.
- [ ] Every Phase 4 pass declares its preserved properties from Decision 4.
- [ ] Seeded replay of a fixed dynamic circuit is reproducible, and conditional
      execution is def-use checked.
- [ ] A target lacking conditional execution is rejected at planning time with an
      actionable blocker; no silent flattening occurs.
- [ ] Sharded dynamic execution either preserves feedback semantics or fails closed
      and reports the blocker. Replicated per-rank execution is never described as
      distributed scalability.
- [ ] Static batched execution shows no regression against the recorded baseline.
- [ ] `IR_VERSION`, `CircuitIR` schema, `docs/public_api_v1.json`, and root exports
      are unchanged.
- [x] `flagquantum/core/ir/**` is the only internal home for the levels, and no
      second internal IR location is proposed or created. The public IR module
      became the package `flagquantum/core/ir/` in the same change, with
      `__init__.py` holding the former module's contents and the glob added to
      `team-ownership.toml` protected paths beside it.
- [x] Repository owner approves the Phase 4 scope. (2026-10-08, together with the
      Phase 4 activation in `API_CHANGE_PROPOSAL_062`.)
- [ ] Compiler and runtime owners confirm the pass-contract and defect-semantics
      details during review.
