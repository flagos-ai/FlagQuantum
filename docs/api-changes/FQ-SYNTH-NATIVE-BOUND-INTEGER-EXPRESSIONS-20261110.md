# Bound integers in native unsigned expressions

The authorized source-driven native migration admits synthesis-time integer
parameters and active static range indices as constants inside the existing
unsigned arithmetic/predicate expression profile. Each loaded name bound to a
plain integer is replaced by an AST integer literal before existing grammar,
range, alias, model and lowering checks. Quantum operands remain borrowed
registers; at least one quantum operand is still required. No Python body is
executed and no general expression evaluator or dynamic loop is added.

Bound real values, arrays, unknown names and host objects do not become unsigned
constants. Existing binding limits, negative-literal rejection, full-domain
result bounds, static loop expansion limits and resource budgets remain intact.
Source locations and original source bytes are preserved. Lexical loop indices
do not escape their existing scope or override quantum names. Target/input alias
checks remain independent of algebraic cancellation.

This completes combinations such as `control(2*j+s%2 == index, ...)` under the
already bounded static loop. It introduces no exports, signatures, physical IR
opcode or numerical kernel. Acceptance compares two nonzero classical bindings,
static loop predicates, coherent arithmetic XOR and budget/error boundaries with
independent Circuit states. The concrete tridiagonal source retains all32 table
entries and exact preparation/inverse semantics. Fresh216-original replay and
immutable resource baseline preservation remain required. No release promotion.
