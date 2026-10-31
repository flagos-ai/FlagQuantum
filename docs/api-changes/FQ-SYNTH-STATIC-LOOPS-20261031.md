# Static bounded range loops in the Python construction profile

Date: 2026-10-31. The human instructed merging #978 and continuing the sixth
plan. #978 merged as 8aec335f477272d8e5f33a75af6f56b9e0b1a502 after its exact
head 2a60f76f806014115b7d9065ff3669bcdf2b422e passed 57 checks with three
configured skips. Stage 2 includes bounded loops after predicate composition.
This slice adds compile-time unrolling behind the existing entry; no runtime
loop, public keyword, contract schema or executor is introduced.

## Accepted computation and scope

A for statement uses one fresh index name and builtin range with one to three
positional nonnegative classical integer arguments. The step must be positive.
Bounds can use existing pure arithmetic, local constants and explicit constants
parameters; they must resolve to INDEX, never quantum values or sampled input
states. Globals, callable range substitutes, keywords, starred arguments and
shadowing range/helpers are refused.

One active loop is admitted, including helper expansion. Nested loop bodies,
while, break/continue, loop else, branches, return inside a loop and side effects
remain unsupported. All loop bodies contain ordinary single-name assignments.
At most 256 iterations are expanded across the entire capture and all helper
calls; the existing 4096-source-node/operation, width and helper-depth limits
still apply. Large ranges are checked arithmetically before iteration.

Parameters remain read-only. Locals initialized before a loop may be updated
once per iteration, each update producing a fresh SSA value. The index is
read-only; newly defined body locals are single-assignment within an iteration
and cannot escape it. Prior aliases retain their original SSA value. Sequential
loops may reuse a scoped index name. Outside loops, the existing single-binding
assignment contract remains unchanged.

Empty ranges are no-ops for carried values. Their bodies are still validated
once in an isolated environment with a classical index; provisional operations
and value IDs are discarded. This refuses unsupported hidden calls/types and
nested helpers without emitting quantum work or leaking a loop-local value.
The normal source-safety budgets also bound that validation.

## Reuse and acceptance

Unroll source expressions into the existing flat HybridProgram/SSA. Existing
width inference, constants/helpers, verifier/ranges/lifetimes, arithmetic/Boolean
kernels, both schedules, value XOR, exact phase prefix-adjoint cleanup, finite
candidate/target checks and Program/run/read remain authoritative. No Core,
Runtime, Services, source file format, signature/export/report or serialization
changes; native Qmod grammar is unchanged.

Compare looped and explicitly expanded computations on complete basis domains,
nonzero XOR outputs, phase-bearing and external-reference-entangled states,
with fresh workspace restored to zero under both schedules. Check public value
and phase, explicit constants, optional qfunc/helpers, snapshots of carried
values, range forms, empty ranges, scope/readonly/step/nesting refusals, exact
iteration/expansion budgets and no planning on source errors. Run the existing
arithmetic, comparison, Boolean, controlled phase and original-Qmod acceptance.
Independent human trials remain 0/2 and release permission remains false.
