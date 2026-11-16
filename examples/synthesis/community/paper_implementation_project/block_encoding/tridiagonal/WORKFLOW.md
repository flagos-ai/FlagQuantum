# Tridiagonal block encoding

The native Python entrypoint retains all 32 table coefficients, including the
final zero, and the source's two selector bits, four position bits and data bit.
A static loop applies a named controlled-phase helper with table index
`2*j + (s & 1)`. Hadamards on the data bit turn its centered phase
`angle*(data-0.5)` into the required RX, including absolute phase.
The single comparison `(table_index ^ index) < 1` means equality and retains
a quantum predicate type in standard Python static checking. QFT/phase/inverse-QFT
implements the two conditional modular shifts. The final adjoint reverses the
exact preparation unitary on every input column, not just its zero-state action.

The original selector amplitudes were rounded to four decimal places and fail
the existing strict normalization check. `RAW` preserves these values; ordinary
Python explicitly normalizes them into `AMPLITUDES`, matching the previously
approved source revision. Tests retain the original refusal and compare the
approved revision, native program and independent Circuit. No original source
or recorded baseline is modified.

The independent Circuit constructs analytical preparation matrices, modular
permutations and all 32 RX blocks. Full complex state comparison checks absolute
phase, register order and zero hidden workspace. Two further nonzero angle and
signed preparation bindings cover all selector branches. Primitive acceptance
checks all preparation-unitary columns, coherent and entangled inputs, inverse
composition, invalid storage, aliases and budgets.

Current native costs are 7 physical qubits, 0 workspace qubits, 23604 operations
and depth 19568. The approved source baseline is 10 physical qubits, 17250
operations and depth 13899. The native version saves three physical qubits while increasing gate count
and depth. A prototyped root controlled-RX path measured 11 physical/4 work bits,
449 operations and depth 381, but its loop callback captures are not yet a
complete statically checked native user path. It is not the accepted example.
Efficient typed predicate capture in mutable helpers remains an optimizer/frontend
gap; this delivery makes no optimality claim.

Only the quantum program is accepted here. Complete Notebook workflows, quantum
controlled preparations, arbitrary register views and hidden-storage/drop
semantics remain separate work.
