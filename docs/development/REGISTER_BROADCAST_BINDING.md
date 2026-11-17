# Scalar and register broadcast binding

K-0260 covers the independent source header `broadcast_resolver.pybind.h`
at Git blob `452c10cc8b2432a0d3b369fcb585b7f9d48b4ebc`.

The shared compiler resolver admits all operand ports before choosing a
width. The first explicit sequence determines that width. Every explicit
sequence must match it. Scalar ports repeat across the selected width.
All-scalar inputs use width one. An explicit one-element array remains a
width-one sequence. An explicit empty array selects width zero and
suppresses scalar lanes. Empty and nonempty sequences cannot mix.
Native retains the ordered values of sliced input views. The source
expresses scalar repetition with a zero-stride span. Native materializes
the resolved tuples consumed by existing ordered gate emitters.

The binding supplies the existing CX and CZ emitters. Gate-specific
basis and target rules stay with those emitters. The source probe covers
five valid CX shapes and two size refusals through the unchanged Python
binding. The native append path matches each admitted ordered CX stream,
including repeated scalar operands and reversed or stepped arrays.
Full complex amplitudes cover all 256 basis positions in each layout.
A second source fixture combines a classical enable, an X-classical output,
and both X eigenstates. Its four Z records and their classical guards match,
and both classical input values preserve the expected full amplitudes.

The resolver itself emits no primitives and declares no resource model.
The valid CX fixture shapes emit 1, 4, 4, 0 and 2 CX gates in fixture order.
The mixed fixture emits three guarded Z and one unguarded Z. These schedules
use no scratch wires or measurements. C++ operation-specific broadcast
wrappers retain separate source entries.

Run `python -m pytest -q tests/unit/test_kickmix_broadcast_binding.py`.
