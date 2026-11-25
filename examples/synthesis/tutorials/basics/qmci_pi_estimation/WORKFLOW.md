# Native amplitude-amplification workflow

The original uses one-bit unsigned x/y and x*x+y*y<4, so all four inputs are marked. This example preserves that setup and the iteration-dependent global sign; it does not claim a numerical estimate of pi.

The qfunc prepares the data and reversible indicator, then composes phase,
within/apply and the inverse preparation for k bounded iterations. Read the
named indicator probabilities directly; dropped data remains in the full state.
No hand-designed controlled gates, physical wire numbers or bitstring decoding
are needed.

At k=1 the native form uses 3 physical bits, zero work bits, 17 operations
and depth 13. The source path uses 3 bits, 15 operations and depth 11.
These are distinct decompositions; no resource improvement is claimed.

See the [shared semantics and acceptance guide](../../../amplification_guide.md) for k=0/1/2
full-state and global-phase checks, nonzero indicator XOR, real budgets and
the separate boundary for complete Notebook workflows.
