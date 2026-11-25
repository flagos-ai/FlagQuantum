# Native tutorial semantic boundaries

This is a local `single_device_fast_path` tutorial. `ansatz.py` owns the new native
reference solutions. `workflow.py` composes retained teaching functions, independent
Circuit references, decoded sampling and numeric metadata. Bell/GHZ and Fourier
square helpers reuse existing accepted native examples. `main.py` and the Notebook
execute the full exercise/solution sequence without modifying those counterparts.

The source Bell pair uses two bits. GHZ uses three, including signed three-bit
integer readout0/-1. Integer arithmetic uses two-bit a/b and infers four-bit c for
3*a+b. Fractional a/b have one/two fractional bits and infer unsigned c with five
bits and two fractional digits. Signed x has three bits and two fractional digits;
its strict x<1/2 condition excludes1/2 and includes all negative values.

Grover retains three two-bit fields and two rounds. Mark3*a+b+2*c==9 and reflect
about the joint zero state. Since the fields are unsigned, their sum is zero if
and only if all three fields are zero. Native `phase(1, pi)` expresses the source
constant phase. The independent reflection has the source sign, retaining global
phase rather than aligning it away. Product phase retains two-bit x/y and pi/8;
show complex amplitudes because probabilities hide the phase. Fourier square keeps
three-bit x/four-bit y and modular wraparound modulo16.

References enumerate arithmetic controlled-X truth tables, signed-condition
permutations, phase diagonals and the Grover reflection; they do not call native
lowering to obtain an oracle. Full states are ordered by declared registers before
separating hidden compiler wires. Native readout uses public `numeric_probabilities`
for signed/fractional words. Actual draws use explicit local seeds; known equations
or optima are never injected into sample counts.

Numeric introspection occurs after synthesis through `program.registers` and
`program.report`; no new public signature or qfunc syntax is introduced. Resource
reports replace the source hosted viewer explicitly. All source code-cell hashes
and the exact source revision are recorded without shipping source code.
