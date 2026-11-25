# Native entanglement tutorial

The reviewed Notebook contains three complete programs: a two-qubit Bell pair
built through a helper, a blank six-qubit exercise, and its GHZ solution using
an adjacent controlled-X chain. `main.py` retains all three. The empty exercise
intentionally produces the zero state; it is a starting point for the reader.

The user allocates named quantum arrays, composes a helper, and uses a bounded
Python loop. Synthesis handles physical wire assignment. The runnable workflow
draws all three circuits with the local text drawer, replacing the cloud viewer,
then executes them and reads named register values without bitstring decoding.

The Bell distribution is `{(0,): 0.5, (3,): 0.5}`; the blank exercise gives
`{(0,): 1.0}`; the GHZ solution gives `{(0,): 0.5, (63,): 0.5}`. The independent
Circuit comparisons check complete complex amplitudes, absolute phase, array bit
order and zero hidden workspace for every program. No remote execution or
hardware measurement is claimed. No additional compiler feature is introduced.

The [provenance record](provenance.json) binds every reviewed Notebook code cell
and retains the original quantum-source identity. See [README](README.md) for
the native workflow and independent comparison commands.
