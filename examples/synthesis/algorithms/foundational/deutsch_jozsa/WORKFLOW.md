# Constant or balanced predicates

Run `python -m examples.synthesis.algorithms.foundational.deutsch_jozsa.main`.
The native workflow builds both examples from the reviewed Notebook, draws each
circuit, executes it, reads the named register `x`, classifies the function and
draws 1000 seeded samples. Both predicates are balanced: the first returns 8
with certainty and the second returns 4 with certainty.

The native oracles use algebraically reduced versions of the original predicates:

| Width | Original predicate | Equivalent phase expression |
| --- | --- | --- |
| 4 | `x > 7` | `pi * most_significant_bit` |
| 3 | `max(x ^ 4, x & 3) >= 4` | `pi * (1 - most_significant_bit)` |

For three-bit x, `x & 3` is always below four, so the second comparison depends
only on the complemented high bit of `x ^ 4`. The constant pi phase in `main`
is essential: removing it changes every complex amplitude's sign, even though
measurement probabilities remain the same. These are mathematical reductions
used in these native examples, not a claim that the native compiler accepts
arbitrary `max`, bitwise predicates or quantum callable parameters.

The independent Circuit constructs the phase matrix from the original Python
truth tables, without using these reductions. Complete complex amplitudes,
absolute phase, register order and clean hidden workspace agree. The unchanged
three-bit original is checked separately in test fixtures. The optimized native
three-bit program has three allocated qubits, one operation, depth one and zero
workspace, matching the frozen original resource row.

The local text drawer replaces the upstream cloud viewer. Standard-library
sampling from exact named probabilities replaces the remote sampling session;
this is local simulation, not hardware measurement. The decision uses zero-outcome
probability under the constant-or-balanced promise, rather than the Notebook's
row-count heuristic. Acceptance checks both constant and balanced decisions and
rejects intermediate zero mass. General oracle callbacks remain outside this
example's native scope.

Notebook identity and all reviewed code-cell hashes appear in
[provenance](provenance.json). The shared
[third-party license](../../../CLASSIQ_LICENSE.txt) applies.
