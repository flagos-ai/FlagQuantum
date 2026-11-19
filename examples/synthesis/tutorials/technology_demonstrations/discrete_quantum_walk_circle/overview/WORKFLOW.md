# Circle-walk resource comparison

Each case keeps the original number of position bits and one coin. A Hadamard
coin controls the periodic forward and inverse multi-controlled-X cascades.
The width-first Circuit implements each multi-controlled operation using an
exact parity expansion and a Gray-code traversal. It uses exactly the required
position and coin wires. The CX-first search starts from that same circuit
and visits native qfunc candidates under the increasing workspace caps, up
to twice the source position width. Each candidate is converted to the same
CX/U3 basis and measured. The fewest CX gates win, with depth and physical
width as tie-breakers. The selected circuit is the best of those visited.

Separate reference Circuits verify the complete complex state for both methods
at the two endpoints. The timing includes local construction and named-basis
measurement, or the bounded candidate search. Reference execution and plotting
are outside the measured interval. Historical system times are reproduced
for context and cannot establish a current relative speedup.

The large-circle curves replay the source's thirteen selected rows per method
from its published CSV. The bundled data preserve every row and field value,
with the repository's standard LF line endings. The smaller Qiskit values
come from the Notebook. Its commented rerun remains disabled.
