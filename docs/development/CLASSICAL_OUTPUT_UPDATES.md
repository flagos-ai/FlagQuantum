# Ordered numerical output updates

Simulation owns `classical_updates.py`, adapted from the unchanged Qualtran
v0.7.0 `ClassicalSimState._update` and `PhasedClassicalSimState._update` in
`simulation/classical_sim.py`, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

The module owns one lifecycle decision: output assignment completes before phase
policy runs. It consumes the original named output mapping and an owner-provided
assignment callback. It creates no state class, signature or graph hierarchy.
The callback owns register order, native binding, dtype validation and distribution
resolution, using the already accepted helpers.

`commit_classical_outputs` assigns first, then rejects every non-None phase.
This includes `0` and the numerical identity `1`; the source does not regard them
as absent. A missing field, invalid value or sampling error wins over a subsequent
phase rejection. Already assigned fields remain visible on error.

`commit_phased_classical_outputs` assigns first, then calls the shared
`apply_classical_phase` using the owner's updated `(register, index)` mapping.
It returns the new accumulator only on success. The caller's old phase therefore
remains unchanged if assignment or phase resolution raises, while prior output
writes remain visible. The incoming accumulator is not revalidated during update;
initial phase validation belongs to state construction. Source NaN diagnostics
and exact outcome-equals-one feedback remain unchanged.

The same callbacks feed two existing user workflows:

```sh
python -m examples.algorithms.classical_action_dispatch
python -m examples.algorithms.classical_measurement_values
```

The recursive controlled-X/Z example now binds and collects both output registers
before phase accumulation. It still uses `circuit.append(op)`, emits 8 T and two
measurements, and cleans one reused work wire; five declared wires include a
spectator, versus the source model's four peak logical wires.

The measurement example assigns the sampled scalar output before resolving its
measured phase. It also demonstrates that pure-mode phase rejection leaves the
written register visible. Its existing quantum copy/clear/feedback recipe still
emits zero T and one measurement, cleans one work wire and declares three wires.
Source measurement-leaf estimates and the full emitted recipe remain separate.
No compiler emitter or bit-order rule changes in this slice.

Reference tests run the actual original update methods against a complete source
leaf graph, then compare exceptions, causes, ordered partial assignments, value
types and phase after success or failure. Cases cover scalar/array binding,
missing fields, symbolic values, invalid phases, indexed measurement feedback,
exact random-generator state, fixed outcomes and sampling failures. Replacement
tests ensure assignment failure prevents phase evaluation and successful phase
evaluation sees the fully assigned mapping. The source full-graph measurement
path is compared to the native user example. Existing source streams, full
complex tensors, every small forward/inverse measurement history, clean work and
both QEC profiles remain in the concentrated regression.

Graph topology and output-wire discovery are not implemented here. The graph
owner must provide the bound mapping; the source `_get_soquet` discovery errors
are not equated with dictionary-key errors. Recursive graph execution and its
decomposition-exception policy remain separate open capabilities.
