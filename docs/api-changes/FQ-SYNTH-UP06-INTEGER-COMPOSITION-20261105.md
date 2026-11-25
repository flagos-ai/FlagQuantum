# UP-06 bounded classical composition

This extends the private Qmod adapter profile without changing public exports,
Python qfunc parameter contracts, Program results or Stable Core serialization.

Classical int parameters are exact Python integers, excluding bool and float,
with magnitude at most 256 bits. int arrays have 1..256 elements; main array
ports declare a fixed length, and helper array ports may infer a bounded length.
Static integer arithmetic and indices never inspect a quantum state or round a
real number. Required main bindings are checked by their declared classical kind.

power(count) repeats a checked static body 0..256 times, without an exposed loop
index. repeat retains its named index. Static integer comparisons may select
gate-only if bodies inside these loops. Quantum conditions are rejected. Existing
scope, recursion, helper-expansion, owned-node and final resource caps remain.
No allocation, consuming ownership or runtime loop is added to static bodies.

Single-argument phase(expression) means phase(expression, 1.0), including
classical global phases and the same bounded real quantum-polynomial profile.
Existing two-argument phase behavior is preserved. No host eval is used.

Pauli evolution may target a static ordered view contained within one writable
storage owner. Local Pauli indices are checked against view width and translated
to the owner's numeric bit indices. Duplicate locations, mixed owners, width
mismatches and aliases are rejected. Compiler remains authoritative for the
existing compact rotation effects, ownership and physical costs.

multi_suzuki_trotter retains the existing singleton profile. Lists of 2..16
operators require equal-length finite real time arrays, first order and one
repetition. Each operator's first-order product is emitted in source list order;
the shared Pauli lowering is reused. Higher-order multi-operator formulas remain
explicitly unsupported. Real helper array literals may contain bounded classical
arithmetic; this does not enable quantum-dependent classical values.

Accept the three unchanged pinned originals: Fermi with iterations 1 and 2,
credit-card with two distinct nonzero pairs of data vectors, and fixed original
BB84 data. Compare full complex128 vectors against independently constructed
Circuits, retaining absolute phase, numeric bit order and zero workspace. Test
wrong classical kinds, array lengths/indices, loop caps, quantum if rejection,
noncontiguous views, invalid aliases, unsupported multi-operator orders and
actual resource-budget refusals. Recheck all 216 originals, retain original
source/license hashes and resource baselines, and update the delivery ledger.
No integration PR or full CI is opened before the agreed complete corpus gate.
