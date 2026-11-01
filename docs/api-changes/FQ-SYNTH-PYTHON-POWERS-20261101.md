# Positive literal powers in the Python synthesis entry

Follow #997 at d78eb7f2a2ca890ecd9263a368c7fa35bf8fb3ed. Extend the existing
Python capture from **2 to positive literal exponents 1..8, matching the native
bounded range. Literal AST integers only: bool, zero, negative, fractional,
symbolic/configured/computed and larger exponents remain refused. Parenthesized
bases and composed functions retain existing capture rules. Source bodies are
never executed; widths use complete declared domains with the 256-bit guard.

Capture the base once. Exponent one returns it; higher powers emit at most seven
existing qint.multiply operations with exact intermediate bounds. Reuse one
private arithmetic fold/width/emission routine for existing +/* and powers,
preserving the **2 graph. No power kernel, canonical IR, public type/API,
dependency, executor or backend selector. Existing Compiler budgets, schedules,
aliasing, cleanup, checked Program/run and named results remain authoritative.

Acceptance covers all exponents, constants and quantum-dependent constants,
nontrivial coherent cubes, native/product producer replacement, both objectives,
value/phase outputs, reference coherence, zero work, base evaluation once,
source diagnostics, rejection before planning and resource/model limits.
No notebook acceptance, global optimum or release claim follows. Preserve
original sources and evidence counts; trials stay 0/2 and unreleased.
