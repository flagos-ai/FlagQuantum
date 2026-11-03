# UP-02 native phase capture

Revision 9 UP-02 starts by bringing the existing one-bit pi phase contract to
the native program capture path. A private Python marker lets typed qfunc
bodies borrow initialized registers and capture an unsigned one-bit arithmetic
expression. The existing register verifier, expansion, phase lowering and
exact target checks handle the resulting qreg.phase operation.

The first slice accepts the exact Python float value of pi from a literal or
a static binding. It shares the current unsigned expression capture with
fresh arithmetic assignment. Static nested and repeated calls are covered
through the same register call graph as Qmod.

The test compares every complex amplitude against an independent sign oracle
for a two-qubit nested and repeated program. Zero amplitudes outside the
logical support also check workspace cleanup. Both native capture and Qmod
execute through the existing public synthesis and Runtime workflow. The
marker is internal while the complete phase contract is being developed.

UP-02 remains open for finite real coefficients, multi-bit phase results,
exact constant global phase, and the public native export. The full worked
example without control is the final acceptance case for this milestone.
