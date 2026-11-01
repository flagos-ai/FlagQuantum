# Bounded native positive literal powers

Follow #995 at 7c5d6e4b896957df4c820782717990c5c01f99d4. Extend native
arithmetic from literal square to positive literal powers 1 through 8. Exponents
must be literal leaf nodes. Zero, negative, fractional, symbolic and larger
exponents remain refused. Preserve right associativity and require parentheses
when a nonliteral power would otherwise occur. Infer the full unsigned domain,
not a basis-specific value; preserve the 256-bit domain guard.

Ecosystem expands a power to at most seven existing qint.multiply operations,
left to right with exact intermediate widths. Power one returns its existing
operand; purely classical powers fold as existing constants do. Compiler remains
authoritative for multiplication, aliasing, scheduling, cleanup, targets and
complete resources. No new IR opcode, API, dependency, executor or power kernel.
This bounded implementation does not claim optimal multiplication chains.

Acceptance covers all admitted exponents, classical and quantum-dependent
constants, cubes with nontrivial multi-bit inputs, explicit-product replacement,
both objectives, coherent phases/references and zero work, initialization, source
locations, associativity and invalid forms, width/expansion/memory budgets.
Original source/license bytes, four-original cohort, trials 0/2 and unreleased
status remain unchanged. Broader Classiq exponent support is not implied.
