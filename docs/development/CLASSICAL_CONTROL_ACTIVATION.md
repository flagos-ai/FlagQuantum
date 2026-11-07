# Classical control activation

Simulation owns `classical_control_activation.py`, ported from the unchanged
Qualtran v0.7.0 `CtrlSpec.is_active` in `_infra/controlled.py`, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

The numerical boundary accepts already-bound activation arrays plus native
symbolic width/shape parameters. It does not copy foreign dtype objects or the
CtrlSpec constructor/converter. Metadata binding retains those responsibilities.
The existing shared `is_symbolic_value` classifier performs the symbolic guard
before checking the number of values. Integer inputs use `np.array`; other inputs
use `np.asarray`. Each value must have exactly the pattern shape. A first value
mismatch returns false before touching later clauses, including their shape or
array conversion. There is no broadcasting or implicit bit-domain validation.

Reference tests cover scalar, vector, matrix, empty and multiple clauses, value
types, malformed counts/shapes and symbolic integer literals. A later array
conversion that would fail is deliberately skipped when an earlier clause is
inactive. Replacement tests verify use of the shared symbolic classifier.
Constructor conversion and generic symbolic register binding remain separate
acceptance scopes.

Run `python -m examples.algorithms.classical_control_activation` for
`circuit.append(op)` with a two-bit word activated at value two. The existing
mixed-polarity Pauli emitter supplies the measured And predicate. Tests compare
its original primitive stream, full complex amplitudes over every input word,
target and spectator, both unnormalized measurement branches, and clean work.
Both source QEC profiles are checked. The example emits four T gates, one
measurement and one work bit. Its five declared wires include a spectator; the
source model has four T equivalents, one measurement and four peak logical wires.
Repeating the operation restores the complete input.
