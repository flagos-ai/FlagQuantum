# Compute Team Boundary

Compute adapters own resources controlled directly by the current process:
device lifecycle, kernels, precision, memory, collectives, topology, and
execution-path evidence. Vendor runtime objects stop here. Do not add remote
task submission, simulation semantics, user-facing policy, or new public contracts.
