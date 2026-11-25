# Resiliency Planning quantum program

Complete retained 24-bit 20-layer quantum program. Two nonzero parameter vectors and both width/depth objectives; raw full-state comparison with independent factored classical costs and Circuit parities. Complete Notebook optimizer workflow remains pending.

Parameters retain interleaved gamma/beta order. Native static loops replace repeat/apply_to_all. Source coefficients, widths and layers are unchanged. No source importer in user examples. Each reference factor enumerates at most six logical bits, with independent Walsh spin-parity Circuit construction; no Compiler algorithms reused. The AMD-named original is byte-identical to the hardware-neutral source; both original paths are verified, and native helpers and reference are shared. Four migrated artifact paths represent three distinct quantum programs.

Run the commands in [README](README.md) for the complete native program and
independent Circuit comparison. Twenty-four-bit examples require a local
complex128 statevector; they print resources and norm instead of constructing
an enormous Python probability dictionary. No width, layer, coefficient or
parameter reduction is used to claim coverage.

Acceptance checks absolute complex128 amplitudes without phase alignment,
logical bit order and zero workspace. The reference only enumerates small
classical factor tables and applies bounded one/two-qubit Circuit operations;
it does not allocate a dense matrix over the entire problem. Static IR
instruction counts and scheduling are reported separately from the frozen
source baseline and do not certify hardware efficiency or optimality.

This completes the retained quantum program only. Optimizer training, data
preparation and complete related Notebook workflows remain separately pending.
