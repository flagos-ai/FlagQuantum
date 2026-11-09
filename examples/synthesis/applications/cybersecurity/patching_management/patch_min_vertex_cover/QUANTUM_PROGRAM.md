# Patch minimum vertex cover quantum program

The native program retains all eighteen edges, penalty ten, eight storage bits,
three cost/mixer layers and all six real parameters. Gamma values precede beta
values. Binary complement squares simplify exactly because each bit is zero or
one. The source field `v.x` becomes an explicit native `Output[QArray[QBit]]`
port `v`, preserving logical bit order without claiming general qstruct support.

Two nonzero parameter vectors and both objectives match independent graph
predicates and Circuit parity/RX rotations, including absolute complex amplitudes
and zero workspace. Native and source both measure 8 qubits, 103 static IR
instructions, depth 31 and maximum instruction width 2. These counts do not
certify hardware costs or optimality. No source importer is used by the example.

Run the commands in [README](README.md) to execute the native program and independent
Circuit comparison. Full Notebook training and optimizer workflow remain pending.
