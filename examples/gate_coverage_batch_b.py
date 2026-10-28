"""Use the complete API Change 098 fixed two-qubit gate batch."""

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

static = fq.Circuit(2).ch(0, 1).csx(0, 1).csxdg(0, 1).dcx(0, 1).ecr(0, 1)
dynamic = DynamicCircuit(2).ch(0, 1).ecr(0, 1).measure(1, classical_bit=0)

print([instruction.name for instruction in static.to_ir().instructions])
print([instruction.name for instruction in dynamic.to_ir().instructions])
