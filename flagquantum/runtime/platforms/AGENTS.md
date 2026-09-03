# Platform Provider Team Boundary

Platform adapters own device lifecycle, kernels, precision, memory, collectives,
topology, and execution-path evidence. Vendor objects stop here. Do not add
simulation semantics, user-facing execution policy, or new public contracts.
