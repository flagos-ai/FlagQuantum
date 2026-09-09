# Phase 24 pass audit and differential verification evidence

Date: 2026-09-10

## Accepted profile

The private optimizer is divided into verified analysis, SSA rewrite helpers,
concrete transformations, and pipeline/audit orchestration. `PassOutcome`
separates a transformed `HybridProgram` from immutable statistics and remarks;
audit content is excluded from Program IR semantic identity.

Constant folding, structured-control simplification, bounded loop unrolling,
and dead-constant elimination report actual transformation counts. The unroll
pass reports preserved-loop counts and stable loop-identity remarks for dynamic
bounds, empty-loop delegation, iteration-budget overflow, and operation-budget
overflow.

## Differential evidence

Twelve deterministic random seeds generate different constant branches,
initial and selected wires, RX/RY choices, loop lengths, and wire strides. For
every seed, the optimized and unoptimized paths agree on:

- ordered instruction names and wires;
- bound parameter values;
- local statevector expectation;
- statevector adjoint value and gradients;
- fixed-point semantic identity after a second optimization run.

A runtime negative loop step is exercised separately. Both paths reject it with
the same fail-closed specialization boundary; Phase 24 does not silently treat
the index type as a signed loop integer.

## Verification summary

| Gate | Result |
| --- | --- |
| Four optimizer responsibilities are split into local modules | pass |
| All four default passes expose immutable statistics | pass |
| Iteration and operation budget skips expose deterministic remarks | pass |
| Twelve seeded circuit and parameter comparisons agree | pass |
| Twelve seeded statevector expectations agree | pass |
| Twelve seeded adjoint values and gradients agree | pass |
| Twelve seeded optimized programs reach a fixed point | pass |
| Negative-step rejection agrees with optimization on and off | pass |
| Hybrid compiler tests | 134 passed |
| Full unit suite under the repository fallback-git environment | 1,257 passed, 14 skipped |

## Claim boundary

This phase strengthens maintainability, observability, and evidence. It adds no
new optimizer semantics, signed loop support, target dialect, Target IR,
general pass plugin API, MLIR/LLVM/QIR integration, performance result,
accelerator, provider, distributed, capacity, or fault-tolerance claim.
