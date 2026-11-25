# Quantum program and workflow boundary

This counterpart covers the complete pinned molecular quantum program. The
surrounding Notebook workflow remains pending in the separate five-category
migration inventory.

The model uses four selector bits, four private data bits and five signed phase
bits with five fractional places. Data starts in logical bits zero and two.
Selector leaf fifteen retains the source unary tree's duplicate leaf fourteen.
Coefficient signs, preparation's action away from zero, conditional scalar
phases and the precise conjugated power sequence are all observable semantics.

`physics.py` records source coefficients and Pauli words. Its classical grouping
collects selectors carrying the same axis on each data bit. The native selector
keeps those groups coherent and preserves the scalar factor needed to express
Pauli Y with a rotation. Group padding uses selector sixteen, outside the entire
four-bit domain, and fails if the pinned group capacities are violated.

`operations.py` composes the selector and reflections into the source walk.
`main.py` prepares data, declares outputs and drops the private data name after
the complete sequence. Drop does not reset or reuse entangled data. Readout uses
named registers and exact signed fractional values.

The independent Circuit uses the same scientific constants, but builds matrices
from Pauli tensor products, probability partition sums, a dense reflection and
Fourier phases. It does not use native construction instructions or the source
importer. Tests also execute the unchanged original fixture, compare the complete
joint state and public density, verify signed marginals and reject insufficient
actual qubit, workspace, operation and memory budgets. Frozen original resource
measurements remain distinct from native measurements.
