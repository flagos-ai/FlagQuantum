# API Change Proposal 013: Compiler optimization naming

## Status

**Approved and implemented before the first public release.**

The repository owner explicitly authorized the direct API convergence on
2026-09-06. No released package promised the removed name, so this change does
not create a compatibility adapter or a versioned deprecation window.

## Decision

Replace:

```python
flagquantum.compiler.simple_compile(program)
```

with:

```python
flagquantum.compiler.optimize(program)
```

`optimize` retains the same accepted inputs, `CircuitIR` result, fixed-point
behavior, parameter and gradient semantics, metadata behavior, and failure
behavior. `simple_compile` is removed from the compiler namespace rather than
retained as a forwarding wrapper.

## Rationale

The operation performs target-independent canonical optimization: it removes
identity operations, cancels adjacent self-inverse operations, and merges
adjacent rotations. It does not perform target legalization, gate-set lowering,
placement, routing, or code generation. `optimize` therefore describes the
actual responsibility more precisely than `simple_compile` and remains distinct
from the target-aware `compiler.compile` entry point.

## Verification

- all production, test, benchmark, and current documentation references use
  `optimize`;
- `flagquantum.compiler` exports `optimize` and no longer exposes
  `simple_compile`;
- the existing fixed-point, trainable-parameter, metadata, differential, and CPU
  execution tests continue to verify unchanged behavior;
- the repository contains no compatibility wrapper or second implementation.
