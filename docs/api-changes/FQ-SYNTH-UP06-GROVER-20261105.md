# UP-06 checked static phase oracles and uniform Grover composition

This private Qmod boundary profile changes no public export or serialized IR.
One-register phase oracles borrow a fixed or caller-sized Const qbit array.
They may apply constant phases under bounded predicates or call another checked
phase oracle with exactly the same borrowed register. They cannot allocate,
write a basis value, transfer ownership, declare outputs, capture a different
quantum register or escape a callback. Validation checks callback bodies even
when a surrounding static power count is zero. Names, recursion, expansion,
source and final resource budgets remain checked.

grover_operator accepts one such function reference or a one-argument lambda,
the built-in hadamard_transform preparation, and one initialized whole writable
array. State preparation remains explicit in the enclosing program. One round
applies the oracle, then the reflection 2|s><s|-I. Absolute phase is retained:
the negative zero-state reflection must include its leading minus sign.
Custom state-preparation callbacks, higher-order callback ports and captured
quantum data remain rejected. Top-level int constants are exact bounded signed
integer literals; they share the existing 256-bit integer admission contract.

Normalize these adapter-private constructs into the existing owned conditional
phase and gate representation. Compiler remains authoritative for ownership,
initialization, scheduling, actual costs and budget rejection. No truth-table
runtime callback, foreign SDK execution or alternative physical template is added.

Accept both unchanged pinned originals, 3_sat_grover.qmod and
grover_max_cut.qmod, at r=0/1/2 against independent public Circuits and analytic
state references. Compare full complex128 vectors without phase alignment,
including clean workspace embedding and explicit bit order. Reject callback
captures, alias substitutions, allocation/Output escape, cycles, unsupported
preparation, malformed constants and resource excess. Recheck all 216 originals
without changing previous passing rows or resource baselines. Preserve source
and license hashes. Update the ledger; no integration PR or full CI before the
agreed all-original acceptance gate.
