# Native borrowed-helper expression XOR contract

Status: approved for implementation in the native synthesis frontend.

## User-visible behavior

A Python `@qfunc` may compose the existing reversible expression-XOR statement
inside a statically inlined helper:

```python
@qfunc
def add_mask(source: QNum[2], target: QNum[2]) -> None:
    target ^= source
```

The call applies `target <- target XOR source`. It preserves a nonzero target,
is its own inverse, and reads live quantum basis values coherently. No helper
result is returned and no fresh register is inferred.

## Accepted boundary

- The destination is one complete unsigned scalar `QNum` or `QBit` with an
  `inout` helper role. It cannot be read by the XOR expression.
- Every quantum expression input is a complete raw scalar register, borrowed in
  ascending logical-bit order. Partial, permuted, array and formatted numeric
  views are rejected.
- Inputs use the existing `inout` helper ports. Actual physical locations for
  all helper arguments must be pairwise disjoint; root `Const` storage remains
  read-only.
- The expression uses the existing bounded unsigned arithmetic/Boolean profile.
  Its inferred domain must fit the full destination width without truncation.
- Calls remain statically inlined and subject to existing depth, call-count,
  expression, operation, width, workspace and memory limits.

## Explicit refusals

This change does not admit writable arrays, indexed targets, target/input
aliasing, overlapping helper arguments, partial/permuted numeric views,
formatted signed/fractional values, implicit resizing, fresh assignment,
mutable-helper adjoint or control, dynamic loops, measurement, allocation,
publication, drop or escaping locals. Those effects require separate contracts.

## Semantic and resource evidence

Acceptance compares the complete complex state against an independent
`flagquantum.Circuit` for every source/target basis value in a bounded domain,
nonzero target values, coherent superpositions, repeated use, helper inlining
and nested static call composition. Workspace amplitudes must return to zero,
and width, workspace, operations and memory refusals must be based on actual
lowering. This contract does not promise an optimal gate decomposition or
resource parity with Qmod.
