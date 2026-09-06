# API Change Proposal 014: Canonical Compiler entry point

## Status

**Approved for pre-public convergence; implementation migration in progress.**

The repository owner required stable API names to express their domain meaning
directly. This proposal applies that rule before the first public release, so no
versioned compatibility window is created.

## Decision

The stable expert Compiler surface has two primary operations:

```python
flagquantum.compiler.optimize(program)
flagquantum.compiler.compile(program, coupling_map=...)
```

`optimize` performs target-independent canonical optimization. `compile`
performs optimization and optional topology-aware lowering. The historical name
`compile_for_backend` is not retained in the first-public API: it mentions a
backend even though its contract accepts no backend object and makes no Runtime
backend selection.

## Migration order

1. record `compiler.compile` as the sole stable compilation entry and remove the
   old name from the root compatibility destination;
2. migrate Compiler and Runtime callers;
3. migrate Circuit and Deployment callers;
4. remove `compile_for_backend` from the Compiler implementation and compatibility
   API after repository imports reach zero;
5. run API, CPU vertical-slice, deployment, and architecture conformance tests.

The function signature and behavior do not change during this naming migration.
Each implementation step remains independently testable; no forwarding wrapper
survives the final step.
