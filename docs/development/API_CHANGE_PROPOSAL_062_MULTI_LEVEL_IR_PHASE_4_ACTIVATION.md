# API Change Proposal 062: Multi-Level IR Phase 4 activation for hybrid and dynamic execution

## Status

**Approved. Integration authorization granted; no Stable Core change requested.**

Approved by the repository owner on 2026-10-08, which is the acceptance item
"Repository owner authorizes Phase 4 activation" below. Approval authorizes the
internal module structure under `flagquantum/core/ir/`, not the public surface:
`IR_VERSION` remains `"1.0"`, `CircuitIR` schema `1.0` remains the only public
serialized IR, and no Stable Core export is added.

The decision also ratifies
[`IR-007`](../architecture/decisions/IR_007_PHASE_4_HYBRID_DYNAMIC_SCOPE.md),
whose own `Status` is updated to `Approved` in the same change. That is a
consequence rather than a second decision: `IR-007` fixes the Phase 4 scope this
proposal activates, both
[the `DynamicCircuit` promotion proposal](API_CHANGE_PROPOSAL_062_DYNAMIC_CIRCUIT_PROMOTION.md)
and [`API_CHANGE_PROPOSAL_066`](API_CHANGE_PROPOSAL_066_DYNAMIC_CIRCUIT_V1.md)
state that their scope "is ratified by `IR-007`", and `IR-007`'s own background
records an unratified scope as a failure mode. Activating the phase without its
scope would leave every downstream row citing a `Proposed` authority.

This proposal activates Phase 4 of
[`MULTI_LEVEL_IR_ARCHITECTURE.md`](../architecture/MULTI_LEVEL_IR_ARCHITECTURE.md)
as a funded development path and authorizes new internal module structure under
the protected path `flagquantum/core/ir/**`.

The public contract does not change. `IR_VERSION` remains `"1.0"`,
`CircuitIR` schema `1.0` remains the only public serialized IR, and the existing
Stable Core export set is unchanged.

## Problem

FlagQuantum's public `CircuitIR` is a flat instruction sequence. That shape is
correct and deliberately frozen for exchange, simulation, training, and the
REST/MCP/task boundaries. It is also, by design, unable to express the following,
which CUDA-Q expresses today and which several FlagQuantum roadmap capabilities
depend on:

- functions, kernels, typed arguments, and calls;
- classical values produced by measurement and consumed by later gates;
- branches, bounded loops, and repeat-until-success;
- reset and mid-circuit measurement with explicit def-use;
- resource allocation and release with lifetime semantics;
- static resource counting over control flow.

`MULTI_LEVEL_IR_ARCHITECTURE.md` § 2.2 records exactly these limitations and § 3
decides that they are addressed **behind** `CircuitIR` rather than by breaking it.
Phase 4 of § 13 is the scheduled work: functions, scopes, blocks, control,
measurement def-use, reset, conditionals, bounded loops, partial evaluation,
`ProgramIR` lowering, QASM Static/Adaptive, and dynamic migration, with the exit
condition "shared static/dynamic semantics without batched regression".

The repository has already recorded this gap from the outside. `CUDAQ_EXPORT_CONTRACT.md`
states that CUDA-Q kernels "can accept arguments, compose kernels, perform
classical computation and control flow, and include measurements", and therefore
that "arbitrary kernel import would either lose semantics or require FlagQuantum
IR features that do not exist". That is the same finding, reached from the
interoperability direction rather than the parity direction, and it is why the
CUDA-Q adapter is one-way by design.

Two things currently block that work.

**Authorization.** `flagquantum/core/ir/**` is listed as a protected path in
`team-ownership.toml`, so the new internal level packages fall under integration
control and cannot be created by the Core team alone. `IR-001` through `IR-006`
cover the importer, linear values, identity layers, custom matrices, parameter
binding, and the Phase 1 static scope — not the Phase 4 surface, which is what
this proposal authorizes.

A related defect was found while preparing this proposal, and it is settled here
because this activation is the change that has to settle it. The public IR was
the single module `flagquantum/core/ir.py`; there was no `flagquantum/core/ir/`
directory. A glob of the form `flagquantum/core/ir/**` matches paths *inside* a
directory and does not match a sibling module, so the two spellings protect
disjoint sets of files.

**Re-measured at approval, because the proposal could not inspect the matcher
when it was written.** `tools/check_team_scope.py` matches protected paths with
`fnmatch.fnmatchcase`, measured directly on the spellings and on files created for
the measurement. Before the split, the module path matched the literal entry and
not the glob. After the split, `flagquantum/core/ir/__init__.py` matches the glob
and not the literal path, and a scratch second file created under the package and
then removed matched the glob as well, which is the case the glob exists for -- a
package with any number of files, not one. The sibling glob
`flagquantum/core/ir/**/*` is strictly narrower than `flagquantum/core/ir/**` and
misses a package's own `__init__.py`. The gap the proposal suspected was therefore
real, and the coarse pattern is the correct one once the directory exists.

The correction could not be applied ahead of the split, and that is a property of
the validator rather than a preference: `protected_path_errors` rejects a
protected entry that matches no file in the worktree, precisely so that an
unprotected surface cannot look protected, so adding `flagquantum/core/ir/**`
while the directory did not exist would fail `--validate`. `team-ownership.toml`
recorded the module alone and carried a comment explaining why the package
pattern had been removed. The gate is therefore satisfied by **two coordinated
edits at the moment the directory appears**: the first file under
`flagquantum/core/ir/` and the `flagquantum/core/ir/**` entry land in the same
change.

**The split is the first implementation step of this activation**, because the
authorized home and the protected-path mechanism have to agree, and the two
candidate ways to make them agree failed measurement differently.

- **Renaming the module was not open.** The public IR has a name, and it is
  `flagquantum.core.ir`. Moving it to `flagquantum/core/ir_levels/` or any other
  sibling directory would leave the authorized path naming nothing and would put
  the single source of truth behind a new name for no capability gain.
- **`flagquantum/_compiler/`, which `MULTI_LEVEL_IR_ARCHITECTURE.md` § 19.13
  fixes as the internal layout, cannot be created in this tree.** Measured:
  creating `flagquantum/_compiler/__init__.py` makes `tools/check_architecture.py`
  fail with `flagquantum/_compiler: unreviewed top-level package directory is
  forbidden`, because the directory is absent from
  `architecture.toml` `package_layout.allowed_top_level_directories`; adding a
  single `from .._compiler import ...` line under `flagquantum/compiler/` fails
  the same gate with `compiler imports forbidden layer _compiler`, because
  `_compiler` is listed in `boundaries.compiler_forbidden`. `team-ownership.toml`
  independently records `flagquantum/_compiler` as one of five patterns removed
  for matching no tracked path. § 19.13's layout is therefore a design sketch
  written against a tree that never had it, and correcting it is a policy change
  rather than part of this activation.

So the module becomes the package `flagquantum/core/ir/` in the same change that
adds the entry. `flagquantum/core/ir/__init__.py` is the former module with three
relative imports deepened by one level and nothing else, so
`flagquantum.core.ir` keeps every name it had, `IR_VERSION` stays `"1.0"`, and
the split is invisible to the module's roughly forty importers.

**Contract-first ordering.** Phase 4 spans Core, Compiler, and Runtime. Under
`docs/development/MULTI_TEAM_DEVELOPMENT.md` these teams cannot write against each
other's unfinished surfaces. The internal IR level boundaries, value/ownership
rules, and verifier failure categories must land as a versioned internal contract
with a conformance test before team implementation begins.

A prior framing of this work as "IR v2" is withdrawn. There is no v2 public IR.
Renaming or re-versioning the public schema would break the compatibility promise
for no capability gain, and non-negotiable rule 9 disallows naming a stable API for
its implementation era.

## Decision

### 1. Activate Phase 4 without changing the public IR

`CircuitIR` schema `1.0` remains the public exchange contract and the only
serialized public IR. Phase 0/1 exit gates in
[`MULTI_LEVEL_IR_PHASE_0_1_EXECUTION_PLAN.md`](../architecture/MULTI_LEVEL_IR_PHASE_0_1_EXECUTION_PLAN.md)
are not reopened. Phase 4 work proceeds alongside any remaining Phase 1/2/3 debt,
subject to the control sequence in `ARCH-010`.

### 2. Authorize the internal module structure

`flagquantum/core/ir/` gains internal level packages for program, quantum, and
target levels, plus a verifier and a diagnostics module. These are internal: they
are not exported from the package root, are not in `docs/public_api_v1.json`, and
may change without a public API proposal as long as `CircuitIR` round-trips are
preserved.

Exactly one public surface is affected, and it is promotable rather than new:
`DynamicCircuit` and `fq.experimental.dynamic.run_dynamic`, already introduced by
[`API_CHANGE_PROPOSAL_010`](API_CHANGE_PROPOSAL_010_DYNAMIC_CIRCUIT.md). This
proposal defines their promotion criteria and authorizes promotion of
`DynamicCircuit` from candidate-stable to stable only when those criteria are met.
`run_dynamic` stays experimental.

### 3. Internal contract before team implementation

The internal IR level boundary lands on the integration branch first, together
with a contract fake and a conformance test, covering:

| Element | Contract |
| --- | --- |
| Level entry and exit | which levels a pass may consume and produce |
| Value identity | value IDs, block arguments, and def-use indexing |
| Linearity | one consumer per quantum value, no stale use, no use after release |
| Joins | exactly one live value per resource contributed by each predecessor |
| Failure categories | typed verifier diagnostics, fail-closed, no `best_effort` |
| Round trip | supported static circuits round-trip to canonical `CircuitIR` payloads exactly |
| Rejection | dynamic, lossy, or unmodelled structure is rejected with diagnostics |

The failure taxonomy of `IR-006` — exactly one of `supported_exact`,
`unsupported_with_diagnostics`, `invalid_input`, with no implicit lossy mode — is
extended to the new levels rather than replaced.

### 4. Do not create a second pass manager or a second artifact boundary

The pass contracts, pipeline, analysis invalidation, `ExecutableArtifact`, and
`RuntimeAdapter` of `MULTI_LEVEL_IR_ARCHITECTURE.md` § 6, § 7, and § 19.10 are the
pass and artifact infrastructure. This proposal adds levels to that pipeline; it
does not introduce a parallel mechanism.

### 5. Preserve the local fast path

Phase 4 must not make an ordinary static circuit slower or require an ordinary user
to understand levels, values, ownership, or verification. The CPU, single-GPU, and
single-device paths retain their existing behaviour, and `MULTI_LEVEL_IR_ARCHITECTURE.md`
§ 4 already requires local execution to be able to emit at the quantum level and
bypass the rest.

## Public API

No new public API in this proposal. The affected surface is the promotion of an
existing candidate:

```python
import flagquantum as fq

circuit = fq.DynamicCircuit(3)
circuit.h(0)
circuit.measure(0)
circuit.cx(0, 1)          # executed only when the measurement result is 1
```

The promotion criteria for `DynamicCircuit` are:

| Criterion | Requirement |
| --- | --- |
| Semantics | measurement, reset, and conditional execution have specified, tested semantics |
| Determinism | seeded replay of a fixed dynamic circuit is reproducible |
| Feedback | measurement-to-condition def-use is explicit and verifier-checked |
| Distribution | sharded execution either preserves feedback semantics or fails closed with a reported blocker |
| Degradation | a target without conditional execution rejects at planning time, never silently flattens |
| Serialization | round-trip through the internal program level preserves conditions and ordering |
| Non-regression | static batched execution is unaffected |

Until every criterion holds, `DynamicCircuit` remains candidate-stable and
`run_dynamic` remains experimental. No Stable Core export is added by this
proposal, and `docs/public_api_v1.json` is not modified.

## Compatibility

`IR_VERSION` stays `"1.0"`. `CircuitIR` schema `1.0` is unchanged and continues to
be the only public serialized IR; `contracts/public-api-v0.2-baseline.json` and
`docs/public_api_v1.json` are unaffected. Existing importers continue to accept
what they accept today.

Rich internal structures convert to `CircuitIR` losslessly or fail closed. There is
no silent dropping of dynamic semantics, matching `IR_006`'s rejection rule and
non-negotiable rule 9's fail-closed requirement in `AGENTS.md`.

Rollback is removal of the internal level packages and the Phase 4 passes. Because
no public schema changed, rollback requires no migration and no deprecation cycle.
`DynamicCircuit` promotion, once made, follows the standard deprecation process
instead.

## Acceptance

- [ ] Internal IR level contract, contract fake, and conformance test merged on the
      integration branch before team implementation begins.
- [ ] Supported static circuits round-trip to canonical `CircuitIR` payloads
      exactly; dynamic, unknown, and lossy inputs fail closed with typed
      diagnostics.
- [ ] Verifier rejects stale values after gates, duplicate consumers, use after
      release, asymmetric allocation, and ambiguous join ownership.
- [ ] Seeded replay of dynamic circuits is reproducible; conditional execution is
      def-use checked.
- [ ] A target lacking conditional execution is rejected at planning time with an
      actionable blocker.
- [ ] Sharded dynamic execution either preserves feedback semantics or fails closed
      and reports the blocker; no replicated execution is described as distributed.
- [ ] `python tools/ci_tier.py pr-runtime` and the relevant `distributed_cpu` tier
      pass.
- [ ] Static batched execution shows no regression against the recorded baseline.
- [ ] `IR_VERSION`, `CircuitIR` schema, and `docs/public_api_v1.json` are unchanged.
- [ ] `DynamicCircuit` promotion is assessed against the criteria table above and
      recorded, or explicitly deferred with reasons.
- [x] Repository owner authorizes Phase 4 activation. (2026-10-08; the same change
      lands the split described under "The split is the first implementation step
      of this activation", because the authorized home and the protected-path
      mechanism have to agree in one change.)

## Non-goals

- Changing `IR_VERSION`, the `CircuitIR` schema, or any Stable Core export.
- Selecting a native or MLIR compiler implementation. That remains the
  evidence-gated Phase 5 question in `MULTI_LEVEL_IR_ARCHITECTURE.md` § 13.
- Committing to timing, pulse, or calibration semantics; these are Phase 5 and each
  matures independently.
- Publishing `fq.AncillaQubit` or any compiler-managed allocation surface, which
  § 5.3.2 defers until after at least two release cycles.
- Promoting `run_dynamic` to stable.
- Any parity or scalability claim. Phase 4 completion is a correctness milestone,
  not performance or capacity evidence.
