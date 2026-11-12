# Coherent lattice streaming and boundary reflection

Run the native program from the repository root:

```bash
python -m examples.synthesis.applications.cfd.qlbm.main
python -m examples.synthesis.applications.cfd.qlbm.comparisons.circuit
```

`main.py` prepares two three-bit position registers, two direction bits and two
two-bit velocity registers. It executes all thirteen scheduled streaming/reflection
steps. `operations.py` expresses periodic shifts and geometry predicates using
native `qfunc` helpers and coherent control. Users specify logical registers and
boundary rules; synthesis allocates and cleans the predicate work bits.

The initial positions are x=1 and y=2. Direction x is one and direction y is a
uniform superposition. Each velocity register has four amplitudes of 0.5.
These are amplitudes, not probabilities: four amplitudes of 0.25 have norm 0.5
and must be rejected. The test fixture retains that upstream normalization error
and separately records the approved amplitude correction.

The logical state-vector axes are x, y, vx, vy, ux, uy, followed by clean
workspace. Comparisons use declared register mappings when another program
allocates these fields in a different physical order.

The exact active-velocity schedule is `(3, 2, 3, 1, 2, 3, 3, 2, 3, 1, 2, 3, 3)`.
Streaming wraps modulo eight. Reflection flips the y direction at y=1 or y=5
when 2<=x<=5, and the x direction at x=2 or x=5 when 1<=y<=5, conditional on
the active velocity. Reflection reads the positions after streaming.

The independent Circuit constructs classical permutation matrices directly
from these geometric rules. It uses neither synthesis helpers nor predicate
lowering. Comparison checks absolute complex amplitudes, including global
phase, and every synthesized work amplitude. Dense matrices here are bounded
reference tools, not the native implementation or a scalability claim.

The default native program uses 19 physical qubits, including seven clean work
bits, 3,342 operations and depth 2,770. It requests at most eight work bits and
64 MiB of execution memory. These are local CPU correctness measurements;
there is no claim of optimality or hardware resource parity.

This example covers the quantum program. A complete application workflow,
including classical analysis and Notebook reproduction, is tracked separately.
