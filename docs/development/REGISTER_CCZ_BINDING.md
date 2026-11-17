# Register CCZ binding

K-0194 through K-0196 cover the locked CCZ binding, header and original
tests. Their blobs are `387c1d7a4d1e6dacae9d940898740e83dd138fd8`,
`a4a3c4e191a5866231e8fc49255fbb72b26bbb57` and
`c3be451b2e6d45ec676585fd69aed0e2dd5a5c69`.

The native binding uses the shared three-column resolver and ordered
phase emission. For the source's homogeneous dispatch, sorting by
operand kind preserves the relative order of quantum controls. Removing
literal controls and using the classical conjunction therefore produces
the same effective stream. The heterogeneous dispatch retains lane and
quantum operand order directly. Nonempty X-basis controls are rejected,
and empty layouts emit no gates.

All 64 original scalar cases and 729 original broadcast shapes match
the source effective streams. Guard order is normalized as conjunctions
while phase operations, multiplicity and quantum operand order are
preserved. A separate append test compares the full complex state for
both settings of its shared classical control while preserving the
classical slot.

The example emits one CCZ and one guarded Z in KMX, with no measurements,
scratch wires or result slots. Native IR realizes CCZ by conjugating
CCX with H on its last wire. Neither the source binding nor the native
wrapper declares an analytical cost model.

Run `python -m pytest -q tests/unit/test_kickmix_ccz_binding_file.py`.
