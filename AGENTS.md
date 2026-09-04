# FlagQuantum Agent Operating Manual

This repository is being built into a production-grade quantum AI framework.
Every agent working here must optimize for a coherent FlagQuantum product, not
for isolated demos.

## Product Mission

FlagQuantum must become a flagship quantum AI framework with:

- one public identity: `import flagquantum as fq`
- one source of truth: FlagQuantum IR
- PyTorch as the primary training interface
- JAX as an optional quantum kernel accelerator
- DLPack and `torch.autograd.Function` for cross-framework gradients
- true distributed statevector, MPS, and tensor-network execution
- trainable sharded gradients, not forward-only showcases
- deployment of trained parameterized circuits to quantum cloud or hardware
- first-class CPU, single-GPU, and single-node developer experience
- production multi-GPU and multi-node scale-out without changing user code

## Non-Negotiable Rules

1. Do not describe replicated per-rank execution as distributed scalability.
   A scalability claim is allowed only when one logical workload is partitioned
   across ranks and reported as `distribution_semantics="sharded_across_ranks"`.

2. Do not let distributed orchestration slow down local users. CPU, single-GPU,
   and single-device JAX paths are first-class fast paths.

3. Do not silently fall back from MPS or tensor network to statevector for
   performance, capacity, or gradient claims. If a fallback is used for
   correctness inspection, report it explicitly.

4. Forward-only sharding is incomplete for quantum AI training. Production
   training claims require gradients and optimizer updates to preserve the same
   distribution semantics as forward execution.

5. Every distributed result or benchmark must expose:
   `world_size`, `local_world_size`, `node_count`, rank ownership, memory,
   communication, `distribution_semantics`, `scalability_claim_allowed`, and
   blockers.

6. Keep the user API simple. Advanced controls may exist, but normal usage must
   remain centered on `fq.Circuit`, `fq.Module`, `run`, `plan`, training loops,
   and deployment packages.

7. The repository remains FlagQuantum. Do not introduce public TensorCircuit-NG
   or `tc` branding in user-facing APIs, docs, examples, or benchmark claims.

8. Treat the Stable Core public API as protected. Do not add, remove, rename,
   reorder, or change stable exports, signatures, defaults, result fields,
   enum/Literal values, documented exception behavior, or serialized public
   schemas without explicit user authorization and an approved API change
   proposal. Never update an API contract or snapshot merely to make tests
   pass. Follow `docs/development/PUBLIC_API_PROTECTION.md`.

## Engineering Decision Principles

These principles govern architecture and implementation decisions. They are
subordinate to the repository's non-negotiable rules, Stable Core API
protection policy, capability maturity policy, and scientific evidence
requirements.

1. **Do not create permanent compatibility debt.** Experimental and
   unprotected implementations may be removed when obsolete. Protected public
   APIs, serialized schemas, deployment packages, checkpoints, and released
   provider contracts must follow the approved deprecation and migration
   process. Compatibility adapters must have a named owner, documented scope,
   removal condition, and target version.

2. **Choose the smallest implementation that satisfies the current verified
   requirement.** Do not add speculative configuration layers, generic
   managers, or extension points without a concrete use case. Prefer a narrow
   implementation that preserves established architectural boundaries.
   Simplicity does not justify bypassing contracts, evidence, safety, or
   capability checks.

3. **Deliver the smallest complete vertical slice first.** Establish an
   end-to-end path through input, validation, planning, execution, result,
   failure, and evidence before expanding breadth. Preserve working local CPU
   and single-device paths while distributed, accelerator, QPU, and service
   capabilities mature independently.

4. **Keep responsibilities and failure domains separate.** Each module has one
   authoritative responsibility. Compiler transforms programs; Runtime
   organizes execution; Simulation performs numerical computation; Providers
   adapt external systems; Ecosystem adapters translate external objects;
   Agent and MCP layers invoke deterministic services. Cross-layer shortcuts
   are prohibited.

5. **Keep core semantics infrastructure- and vendor-neutral.** Core domain
   models, IR, capability vocabulary, and validation rules must not depend on
   network frameworks, databases, MCP SDKs, vendor runtimes, QPU SDKs, or
   accelerator-specific types. Infrastructure adapters should use mature,
   maintained libraries when justified, but external library objects must stop
   at their owning boundary.

6. **Inspect existing capabilities before adding dependencies or
   abstractions.** Before adding a dependency, package, contract, registry, or
   compatibility layer, inspect the existing repository and current
   dependencies for an authoritative implementation. Do not create a second
   source of truth. A new production dependency requires a documented need,
   ownership boundary, license and supply-chain review, replacement interface,
   and exit plan.

7. **Design stable boundaries for long-term evolution.** Program artifacts,
   capability models, execution contracts, result evidence, and module
   responsibilities must be designed for versioned evolution. Avoid temporary
   cross-layer designs described as "replace later." Stable contracts may
   change only through an explicit proposal, compatibility analysis, migration
   path, conformance tests, and approval.

8. **Reuse proven patterns without surrendering architectural ownership.**
   Review mature frameworks, standards, and production systems before
   inventing a new mechanism. Adopt validated principles and isolated,
   replaceable components where appropriate, while preserving FlagQuantum-owned
   semantics, contracts, evidence rules, product identity, and dependency
   direction.

9. **Fail closed and make degradation observable.** Unsupported capabilities
   must fail at the earliest knowable stage. Approximation, precision downgrade,
   backend substitution, and CPU fallback are permitted only when explicitly
   authorized by policy and must be recorded in the plan, result, and evidence.
   Silent fallback is forbidden.

10. **Prove architecture through replacement and conformance.** A boundary is
    not considered complete merely because an interface exists. It must be
    demonstrated by replacing at least one implementation without modifying its
    consumers and by passing the corresponding contract and conformance tests.

11. **Prefer deliberate code over generated volume.** FlagQuantum must remain
    simple without becoming simplistic. Code is not valuable because it is
    longer, more generic, or more heavily layered. Before review, remove
    speculative scaffolding, pass-through wrappers, duplicated validation,
    repeated representations, commentary that merely restates code, and tests
    that only enumerate implementation details. A new manager, registry,
    factory, protocol, helper layer, or intermediate object requires a distinct
    current responsibility or a second concrete use, not a hypothetical future
    need. Preserve necessary scientific rigor, failure semantics, and evidence;
    reduce cognitive load rather than correctness.

## Human Maintainability Guardrails

Architecture must remain approachable to contributors who do not understand
every internal contract. Machine-verifiable rigor must not force ordinary
feature developers to manipulate orchestration, identity, provenance, or
evidence internals.

1. **Keep ordinary changes within one primary domain.** An ordinary feature or
   fix should normally modify the internal implementation of only one primary
   domain. Cross-domain delivery must be split into an approved contract change
   followed by domain-owned implementation changes. If one ordinary
   implementation repeatedly requires changes across four or more domains,
   treat it as an architecture defect and stop for boundary review.

2. **Maintain one ten-minute golden path per domain.** Each migrated domain must
   provide a small, executable example that a new contributor can understand,
   run, modify, and test within approximately ten minutes.

3. **Keep internal machinery out of public APIs.** Public interfaces must not
   require users to construct or understand internal fingerprints, provenance
   records, legality attestations, capability snapshot identities, scheduler
   objects, or evidence internals.

4. **Require justification for every new contract type.** Do not introduce a
   new contract, identity, verdict, registry, or intermediate representation
   unless the proposal demonstrates why an existing authoritative type cannot
   express the verified requirement.

5. **Split large modules without multiplying public concepts.** Once behavior
   and boundaries are stable, divide oversized internal modules by
   responsibility. Internal refactoring must not create additional public
   abstractions merely to reduce file size.

6. **Preserve scenario-oriented tests.** Contract, determinism, and tamper
   tests must be accompanied by readable tests that demonstrate how a user or
   provider completes a real workflow.

7. **Document every migrated domain locally.** Each target domain directory
   must contain a concise README describing what the domain owns, what it must
   not own, its allowed dependencies, its public entry points, and the shortest
   path for making and testing a typical change.

8. **Include contributor usability in architecture acceptance.** A migrated
   boundary is not complete until a contributor unfamiliar with its internal
   implementation can independently make, test, and explain one representative
   small change using only the domain README and referenced golden path.

9. **Review for subtraction before addition.** Every non-trivial change must be
   reviewed for code and concepts that can be removed or reused. Passing tests
   is necessary but does not justify redundant abstractions or machine-like
   boilerplate. Reviewers must reject code whose structure cannot be explained
   in terms of current product behavior and domain ownership.

## Source Documents

Before changing distributed runtime, planners, benchmark claims, or quantum AI
training paths, read:

- `docs/concepts/DISTRIBUTED_QUANTUM_AI_PRINCIPLES.md`
- `docs/concepts/DISTRIBUTED_SCALABILITY_PRINCIPLES.md`
- `docs/roadmap/CAPABILITY_MATURITY.md`
- `docs/reference/KNOWN_LIMITATIONS.md`
- `docs/development/PUBLIC_API_PROTECTION.md`

These documents are binding design standards.

## Multi-Team Worktrees

Concurrent development sessions must use the branch and linked worktree assigned
in `team-ownership.toml`. Never run two writing sessions in the same worktree,
switch another team's worktree to a different branch, or commit unrelated
changes left by another session.

Before editing, identify the responsible team and run
`python tools/check_team_scope.py --team <team> --files <paths...>`. A team may
change its most-specific owned paths and shared test/documentation paths.
Protected integration surfaces, including Stable Core contracts, public API,
architecture policy, CI, dependency manifests, and ADRs, require a separate
integration change before team implementations proceed. Do not create a private
duplicate contract to bypass this rule.

Cross-team work follows contract first, then implementation: the integration
branch lands the versioned contract, contract fake, and conformance test; team
branches synchronize that baseline and implement independently; the integration
branch then runs replacement and cross-implementation tests. Follow
`docs/development/MULTI_TEAM_DEVELOPMENT.md`.

`codex/flagquantum-vnext-architecture` is the only authoritative integration
branch. Team branches must deliver committed, clean work with a handoff record,
must obtain other teams' changes only by merging the integration branch, and
must never merge each other directly. The integration worktree merges one team
at a time with a non-fast-forward merge and runs required checks before the next
merge. Abort conflicted merges and return them to the owning team; revert a
shared failed merge instead of rewriting integration history. Releases,
benchmarks, and capability claims must name a verified integration commit or
tag, never an unintegrated team branch.

## Architecture North Star

The intended product architecture is:

```text
fq.Circuit / fq.Module
        |
        v
FlagQuantum IR
        |
        +-- compiler / drawer / cloud deployment
        +-- local CPU/GPU PyTorch native runtimes
        +-- JAX quantum kernels exposed through PyTorch autograd
        +-- distributed statevector amplitude sharding
        +-- distributed MPS site/bond sharding
        +-- distributed TN slice/reduction and reverse contraction
```

The same circuit and training code should run in local development and
production. Backend choice is controlled by environment and planner policy, not
by rewriting user code.

## Required Development Workflow

1. Inspect the existing implementation with `rg` before editing.
2. Classify the path being touched:
   `single_device_fast_path`, `sharded_across_ranks`,
   `rank_local_replicated_kernel`, `manual_sliced_tensor_contraction`,
   `observable_term_parallel`, `data_parallel_replicated`, or
   `replicated_per_rank`.
3. Implement or update runtime metadata before writing benchmark language.
4. Add focused tests that fail if semantics are misreported.
5. Preserve existing public APIs unless a migration is explicitly planned.
6. Run the smallest meaningful tests first, then broader tests when the blast
   radius is larger.
7. Update the capability matrix and known limitations when a support boundary
   changes.

## Verification Standards

Use the tiered pytest policy in `docs/development/TESTING.md`: run the smallest meaningful
layer first, then expand by blast radius. Do not start with the largest suite
when a smaller seeded tier proves the touched surface.

Default verification for every change:

```bash
python tools/ci_tier.py pr-default
# equivalent direct command:
python -m pytest -m "smoke or unit" -q
```

Expand by blast radius:

| Change Surface | Primary Tier | Command |
| --- | --- | --- |
| local API, runtime, planner, compiler, or single-device behavior | `integration` | `python tools/ci_tier.py pr-runtime` |
| distributed planning, runtime metadata, audit logic, benchmark JSON, or release-gate code | `distributed_cpu`, `benchmark_contract`, `release_gate` | `python tools/ci_tier.py pr-distributed` |
| accelerator-backed distributed behavior | `distributed_accel`, `gpu` | `python tools/ci_tier.py gpu-scheduled` |
| multi-node transport or rank-placement behavior | `distributed_multinode` | `python tools/ci_tier.py multinode-scheduled` |
| release candidate or promoted benchmark claim | full non-scalability pytest plus benchmark audit | `python tools/ci_tier.py release` |

Only run marker-selection commands that have seeded tests; an empty selection is not verification evidence.

Direct marker commands are also valid when a focused run is clearer:

```bash
python -m pytest -m "integration" -q
python -m pytest -m "distributed_cpu" -q
python -m pytest -m "distributed_cpu or release_gate or benchmark_contract" -q
python -m pytest -m "distributed_accel and gpu" -q
python -m pytest -m "distributed_multinode" -q
python -m pytest -m "benchmark_contract or release_gate" -q
```

Path classification maps to test tiers as follows:

| Path Classification | Required Verification Tier |
| --- | --- |
| `single_device_fast_path` | `smoke`, `unit`, and local `integration` |
| `sharded_across_ranks` planner, preflight, audit, or metadata | `distributed_cpu` plus `release_gate` when claim rejection or promotion is touched |
| `rank_local_replicated_kernel` or `replicated_per_rank` | `distributed_cpu` or `benchmark_contract` proving non-release classification |
| `manual_sliced_tensor_contraction` or `observable_term_parallel` | relevant local `integration` plus benchmark/audit contract tests when claims are emitted |
| real accelerator distributed execution | `distributed_accel` and `gpu`, plus benchmark audit before any public claim |
| real multi-node production transport | `distributed_multinode`, plus benchmark audit and release payload validation |

Current focused file-list commands remain useful for review-sized checks and
legacy bisects:

```bash
python -m pytest tests/test_native_circuit.py tests/test_backends.py -q
python -m pytest tests/test_distributed_statevector.py tests/test_jax_distributed_plan.py tests/test_distributed_scalability_audit.py -q
python -m pytest tests/benchmark_contract -q
```

CPU distributed tests prove semantics and fail-closed behavior only. They may
also check metadata shape and development parity. They do not prove real
multi-GPU or multi-node capacity expansion and must never be cited as
scalability release evidence.
Never use CPU distributed tests alone as scalability release evidence.

Promoted benchmark JSON still requires fail-closed audit validation:

```bash
python benchmarks/audit_results.py --input benchmarks/results
python benchmarks/audit_results.py --input benchmarks/results/scalability --require-scalability
```

Only promote a benchmark as scalability evidence if the audit passes and the
payload demonstrates one logical workload sharded across ranks with release-gate
claim evidence. Real multi-GPU or multi-node production claims require
`distributed_accel` or `distributed_multinode` coverage plus benchmark audit and
release payload validation.

## Benchmark Honesty

Benchmark conclusions must separate:

- local peak performance
- JAX kernel acceleration
- data-parallel throughput
- rank-local replicated execution
- observable-term parallelism
- true capacity expansion through sharding

If PennyLane, TensorCircuit-NG, CUDA-Q, cuQuantum, Qiskit, or another framework
is used as a comparison, the comparison must state backend, device, interface,
gradient method, dtype, warmup, iterations, timeout behavior, and whether the
workload is exact, approximate, sharded, sliced, or replicated.

## Coding Bias

- Prefer reusable runtime, planner, and IR features over benchmark-specific
  shortcuts.
- Prefer fail-closed behavior over optimistic claims.
- Prefer precise blockers over vague TODOs.
- Prefer one coherent FlagQuantum abstraction over parallel project silos.
- Preserve CPU and single-GPU usability even while developing distributed code.

## Current Strategic Priority

The immediate vNext control sequence is binding:

1. freeze new horizontal abstractions;
2. finish and simplify the current Compiler boundary;
3. deliver the smallest complete CPU vertical path;
4. move the code on that proven path into its authoritative target domains;
5. delete or explicitly freeze legacy and transitional code in every round.

A new cross-domain contract or abstraction is allowed during this sequence only
when the current vertical path cannot be completed with an existing
authoritative type, and the change passes the proposal and subtraction review.
Do not open additional horizontal architecture tracks while the CPU path is
incomplete.

Longer-term statevector, MPS, tensor-network, accelerator, QPU, and distributed
training goals remain product outcomes, but they must grow by extending proven
vertical paths rather than by accumulating parallel scaffolding. The target is
a repository whose structure is obvious, whose code is restrained, and whose
hard scientific and systems problems are handled deeply.

## Execution Environment Safety

- All external commands must be bounded by an explicit timeout when hanging is plausible.
- Do not retry the same failed operation more than twice.
- Do not use privilege escalation for ordinary repository operations.
- On timeout, terminate stale child processes before continuing.
- Check repository locks, test-runner locks, temporary files, and sandbox permissions.
- Long-running distributed commands must emit phase heartbeat, last operation,
  memory trend, collective state, and rank activity. On genuine no-progress,
  classify the cause, retain diagnostics, terminate stale participants, and
  verify cleanup. Bounded compile/checkpoint phases use explicit larger budgets.
- Application restart is a last resort, not the default recovery action.
- A session must never claim completion when repository writes or required tests were unavailable.
