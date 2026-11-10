# Private-state overlap measurement

Run the main module to prepare two three-qubit states and report the overlap
measurement. The Python function declares local quantum handles, prepares the
states, invokes a composable overlap helper and explicitly drops the private
names. Users see only `test` in `program.registers` and need no physical wires
or bitstring decoder. The helper uses existing gates and phase synthesis.

The original rounded vectors are not normalized. `build_program` explicitly
normalizes both; this matches the approved diagnostic revision and preserves
their relative amplitudes. The untouched original continues to refuse.

Drop is not free: the final seven-qubit state retains both private states and
their entanglement with `test`. Marginal probabilities sum over that environment.
The environment is not zero and is not counted as clean workspace. No storage
is reset or recycled. Both native and approved programs use seven physical bits
and zero work bits. The native phase-based controlled swap uses 67 operations
and depth 39, versus 37 operations and depth 16 for the approved source
primitive. This accepts semantics, not resource parity; the lower-level
controlled-swap optimization remains separate work. Tests independently compare
full complex vectors using a
Householder preparation and a direct Fredkin permutation, then check the
analytic vectors `(a⊗b ± b⊗a)/2` and overlap probability. Two further nonzero
input pairs, equal states and all actual resource caps are covered.

This accepts the quantum program only. It does not certify a complete Notebook
workflow or the pending generic callback/storage features in other examples.
