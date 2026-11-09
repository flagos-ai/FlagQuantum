# Electric grid optimization quantum program

Complete retained eighteen-bit, four-layer electric-grid quantum program. Two nonzero parameter vectors; raw full-state comparison with independent local supply, balance and slack cost factors plus Circuit parities. Notebook optimizer workflow remains pending.

Three explicit four-bit decision ports and three two-bit slack arrays replace source fields. Retain all generation costs, four one-hot slot residuals, three source supply residuals, source coefficients and four layers. Native storage is explicit and does not claim general qstruct or multidimensional-array support. Mixer parameters remain separated from cost parameters as in the source; no layer reduction. Independent Circuit truth tables enumerate at most six bits per factor and use bounded parity operations, not synthesis Compiler algorithms.

Native resources are 18 qubits, zero workspace, 377 static instructions and depth
46. The separately frozen source uses 18 qubits, 373 instructions and depth 54;
maximum instruction width is two in each. These counts are current static IR
statistics, not hardware basis-gate costs or an optimality claim. Full 18-bit
statevector tests run locally with complex128 amplitudes.

Run the commands in [README](README.md) to compare the full native state with the
independent Circuit reference. Tests preserve absolute global phase, declared
field order and zero workspace. Width and depth objectives emit the same complete
instruction sequence. The full Notebook optimizer workflow remains pending.
