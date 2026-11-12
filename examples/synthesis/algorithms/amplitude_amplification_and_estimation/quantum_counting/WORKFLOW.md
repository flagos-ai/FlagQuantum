# Native amplitude-amplification workflow

The original predicate is a+b<=2 on two two-bit unsigned numbers. The IQAE companion has the same quantum program and reuses these preparation/reflection helpers.

The qfunc prepares the data and reversible indicator, then composes phase,
within/apply and the inverse preparation for k bounded iterations. Read the
named indicator probabilities directly; dropped data remains in the full state.
No hand-designed controlled gates, physical wire numbers or bitstring decoding
are needed.

At k=1 the native form uses 5 physical bits, zero work bits, 998 operations
and depth 863. The source path uses 5 bits, 781 operations and depth 665.
These are distinct decompositions; no resource improvement is claimed.

See the [shared semantics and acceptance guide](../../../amplification_guide.md) for k=0/1/2
full-state and global-phase checks, nonzero indicator XOR, real budgets and
the separate boundary for complete Notebook workflows.
