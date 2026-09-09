# Private hybrid compilation contract

Status: approved for conditional Phase 1 entry by the repository owner's
2026-09-09 direction to record and execute the Python-first hybrid compilation
plan.

## Purpose

Authorize the smallest Compiler-owned semantic core needed to represent the
first hybrid quantum-classical vertical slice without changing a stable public
API or creating a second execution contract.

This contract is deliberately narrower than the long-horizon multi-level IR
architecture. It authorizes only the semantic objects that are exercised by the
first nested-loop, data-dependent-gate, scalar-expectation scenario.

## Existing contracts reused

- Core-owned `CircuitIR` remains the quantum-region handoff.
- Existing compiler entry points continue to consume and return `CircuitIR`.
- Existing Runtime entry points continue to consume `CircuitIR` or their
  existing execution-plan type.
- Existing local statevector execution remains the first numerical target.
- Existing tensor and result types remain authoritative at the execution edge.
- Existing extension discovery and lifecycle remain the only plugin authority.

No new public artifact, execution result, backend, plugin registry, or root
export is authorized.

## Newly authorized private responsibility

`flagquantum.compiler.hybrid` may own a private structured program
representation containing:

- typed SSA-like values;
- immutable operations, blocks, and regions;
- function arguments and one return;
- scalar/tensor constants, indexing, comparison, addition, and remainder;
- structured `if` and `for` regions with explicit yields;
- ordered quantum effects;
- parameterized RX, RY, CX, embedding, and scalar expectation operations;
- deterministic semantic identity and source-located diagnostics;
- a fail-closed verifier.

These types are compiler implementation details. They must not be re-exported
from `flagquantum`, added to the Stable Core manifest, serialized as a public
schema, or accepted by `fq.run`/`fq.plan` during Phase 1.

## Dependency direction

The Phase 1 package may depend on Python standard-library modules and stable
Core semantic types needed for later `CircuitIR` lowering. It must not import
Runtime, Simulation, Compute, Remote, Ecosystem, provider SDKs, or optional
frameworks.

PyTorch is not required by the Phase 1 semantic core. Framework tensor values
enter only in later capture and execution phases behind separately tested
adapters.

## Required invariants

The Phase 1 implementation must prove:

1. immutable deterministic representation;
2. definitions dominate uses within the accepted single-function profile;
3. values have one definition and type-consistent uses;
4. branch result arity and types match;
5. loop-carried value arity and types match initialization and yield;
6. ordered quantum effects cannot be duplicated, reused, or dropped;
7. unsupported operation names and attributes fail closed;
8. source locations do not affect semantic identity;
9. no current stable import, signature, default, schema, or behavior changes;
10. the package remains unused by the default compiler and runtime until the
    later vertical-slice gates authorize integration.

## First-profile exclusions

Phase 1 does not authorize:

- Python AST capture;
- `while`, recursion, exceptions, mutation, or general Python semantics;
- measurement-produced control flow;
- stateful simulator sessions;
- lowering or execution;
- differentiation implementation;
- public API or default-path integration;
- MLIR, LLVM, QIR, timing, pulse, provider, or deployment behavior;
- Catalyst or FlagQuantum-authored C++.

## Entry gates

Implementation begins only when:

- the Integration worktree contains no unrelated uncommitted changes;
- the assigned Compiler worktree is on the branch declared by
  `team-ownership.toml` and synchronized with the approved Integration commit;
- team-scope preflight passes for all proposed implementation and test files.

The current Integration worktree satisfies the unrelated-change gate after the
compiler-extension work was incorporated upstream. The assigned Compiler
worktree branch reconciliation remains outstanding.

## Phase 1 acceptance

- focused construction and negative-verifier tests pass;
- semantic identity is deterministic under repeated construction;
- source relocation does not change identity;
- all declared invalid value/effect/control-flow cases are rejected;
- compiler package boundaries pass the architecture checker;
- default smoke/unit tests pass;
- protected public API comparison reports no hybrid additions;
- the Compiler README contains a ten-minute internal contributor path.

Phase 1 completion authorizes Phase 2 capture design review; it does not
automatically authorize capture, execution, gradients, or public claims.
