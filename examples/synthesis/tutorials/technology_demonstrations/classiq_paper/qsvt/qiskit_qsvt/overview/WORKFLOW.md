# Literal source workflow and semantic boundaries

| Source code cell (all commented) | Native counterpart |
| --- | --- |
| 3: imports, CX/U basis, optimization level 3, Draper QFT adder | Public native qfunc and exact CX/U3 conversion; all 128 columns independently verify each of the two source Fourier adders |
| 4: reflection, block encoding, projector and QSVT | Negative-zero reflection, controlled modular shifts, exact projector phase polynomial and checked adjoint composition |
| 6: size 6, degree 3, four phases, compilation and resource readout | Same logical workload and phases; actual width/depth/CX/U3 counts and monotonic local durations |

The source is entirely commented. This reconstructs its literal circuit semantics
using FlagQuantum; no Qiskit execution, optimizer-level equivalence, private
compiler pass, foreign performance reproduction or speedup is claimed.

The reflection is **-1 on data zero and +1 elsewhere**, opposite to the Classiq
peer. This sign is retained under coherent block control. The zero-select branch
adds two to the seven-bit word, then both branches subtract one. The source
no-swaps QFT plus phase gates implements ordinary modular addition with wire
list `[data[0], ..., data[5], deletion]` least-significant-first. Each independent
Fourier matrix is checked against all 128 modular permutation columns for
constants 2 and 127. Phase gates are not replaced by RZ without accounting for
their scalar phase. Projector RZ angles are radians, preserving absolute phase.

QNum bit indexing is least-significant-first; named output wire order is
most-significant-first. The independent Circuit uses named order block, data,
aux; select is wire 2, deletion wire 1, projection wire 0, data wire 8 is the low
bit, and aux is wire 9. Its largest dense complex128 operator is 256 by 256
(1 MiB), for correctness inspection only. Synthesis does not use dense matrices.

Native and converted complete complex states are checked for zero, nonzero
boundary and complex coherent inputs, and a second four-angle binding. Hidden
compiler workspace must return to zero; logical block and auxiliary registers
remain outputs. Acceptance checks absolute phase, named auxiliary probabilities,
actual target costs, invalid angles and native/target budget refusals. This is
not an exhaustive proof of all 1024 full-program columns.

The source provides no width limit. The native local policy adds an explicit
ceiling of eleven physical bits and one work bit, a 100,000 native-instruction
budget, a separate 200,000-added-operation conversion budget, and a 64 MiB
execution-memory budget. The latter is not a process-RSS bound. Costs and times
are actual local observations, without optimality, hardware or release claims.
This workflow is `single_device_fast_path`. Source Notebook outputs stay empty;
actual Jupyter execution evidence is stored outside the example tree.
