# Bound classical action dispatch

Simulation owns `classical_action.evaluate_classical_action`, ported from the
callback-dispatch portion of unmodified Qualtran v0.7.0 `ClassicalSimState.step`
at `bcb17bb25360ed68ce3bc084645db3e929799389`. It consumes already-bound named
inputs, numerical and phase callbacks, and a caller-owned decomposition evaluator.
There is no new graph, signature, operation registry or backend abstraction.

The execution order is deliberate:

1. Call the numerical action with keyword inputs.
2. Call the phase action with the same original input values, before checking
   the numerical result's type.
3. For the `NotImplemented` singleton, reject any non-None phase. Otherwise call
   the decomposition evaluator with the original input dictionary.
4. For a direct result, require an actual dictionary, including subclasses,
   and return that dictionary and phase object unchanged.

The two keyword-call dictionaries are fresh, as in the source. Mutable field
objects remain shared. An empty dictionary is a direct result, not a request
to recurse. None, lists and other non-dictionary results are not reinterpreted.
Callback failures retain their order; no recursion occurs after a value or phase
failure. Decomposition-evaluator exceptions are not translated here: graph-owned
decomposition construction and its exception policy remain outside this helper.
Output binding, dtype checks and phase validation/application also remain outside.

`ClassicalValueCallback` is the single named callback interface for this dispatcher
and the existing controlled-value/phase consumers. Their former private protocol
has been removed. No protected Core exports or contracts changed.

Run the complete user workflow without Qualtran:

```sh
python -m examples.algorithms.classical_action_dispatch
```

The numerical root requests recursion and its evaluator composes existing
word-controlled X/Z callbacks through the same dispatcher. The quantum operation
is the existing `WordControlledPaulis`, still appended with `circuit.append(op)`.
The example checks forward and exact inverse complex states, work clearing and
the separate emitted/model counts: 8 T and 2 measurements, 1 reused work wire,
5 declared wires including a spectator, and 4 source peak logical wires.

Reference acceptance uses the actual inherited upstream `.step` on a local
contract leaf, recording the update boundary without replacing the source
dispatch. A second real source root has no numerical override and therefore
recurses through its controlled-X/Z decomposition. All eight computational inputs
in both directions match full source graph values/phases. Original primitive
streams, the complete source tensor, all four forward and sixteen round-trip
measurement histories, work zero and both QEC profiles are checked. A replacement
test confirms the example reaches the dispatcher for the root and both leaves.

This slice does not implement topological graph traversal, typed wire propagation,
graph-owned output collection or a general classical graph executor. It provides
the shared action-selection step those consumers need, without claiming their
remaining responsibilities are complete.
