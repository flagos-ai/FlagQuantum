# Controlled numerical values and phases

Simulation owns `classical_controlled_values.py`, ported from the unchanged
Qualtran v0.7.0 `_ControlledBase.on_classical_vals` and `basis_state_phase` in
`_infra/controlled.py`, commit `bcb17bb25360ed68ce3bc084645db3e929799389`.
The native boundary consumes bound control/child names, activation arrays,
native symbolic parameters and plain named callbacks. It adds no foreign dtype,
graph hierarchy or execution backend.

Value evaluation copies the named mapping as the original `**kwargs` call does,
then rejects non-THRU metadata before reading required fields. Controls and child
fields are extracted before activation, including when inactive. An active child
is called once; `NotImplemented` is returned unchanged. Mutable child output
dictionaries are updated in place with the original control objects. Inactive
values retain extra fields and object aliases in the fresh keyword dictionary.

Phase evaluation shares field extraction and the accepted activation helper.
It deliberately has no THRU-only guard, passes active child phases through
unchanged, and returns None when inactive. None and opaque child phase results
are not interpreted or converted by this layer. Generic stochastic measurement
phase evaluation remains a separate responsibility.

Reference tests compare unmodified direct `Controlled` callback behavior for X,
Z and Hadamard, including arbitrary diagnostic values, extras and unsupported
classical action. A local contract child verifies the unmodified wrapper's
in-place update and opaque phase identity without patching any reference class.
Replacement tests prove that both consumers use the shared activation helper.

Quantum parity follows the public `.controlled(...)` factory. For a two-bit
word, the original factory selects `ControlledViaAnd`; directly constructing
`Controlled` bypasses that selection and is not used as the gate-flow reference.
The factory-created X-then-Z composite is compared by original primitive stream,
complete source tensor, full complex input/spectator amplitudes, all four forward
measurement histories and all sixteen forward/inverse histories. Work is checked
clean, and both QEC profiles match existing native compound-control cost keys.

Run `python -m examples.algorithms.classical_controlled_values` for
`circuit.append(op)`. The two measured controls emit eight T gates, two
measurements and reuse one work bit. Five declared wires include a spectator;
the original model has eight T equivalents, two measurements and four peak
logical wires. Exact inverse restoration preserves complete phases and controls.
