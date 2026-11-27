# Compute Team Boundary

Compute groups direct resources in `local/`, external task adapters in `remote/`,
and classical wire protocols in `realtime/`. Follow each nested boundary.

Local adapters own directly controlled device lifecycle, precision, memory,
collectives, topology, launch mechanisms and execution-path evidence. Vendor
runtime objects stop at their owning adapter. Remote owns external submission,
status, cancellation and result decoding behind FlagQuantum contracts. Realtime
keeps its independent standard-library protocol boundary.

Numerical kernel algorithms belong to Simulation. Do not add simulation
semantics, Compiler passes, Runtime scheduling or new public contracts here.
