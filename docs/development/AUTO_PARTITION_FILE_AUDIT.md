# AutoPartition source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Frozen bookkeeping files are
Apache-2.0, copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0318 | `auto_partition.py` | `93cddd9cbb1ab309f713316f066f16d93e5fd8b0` | Open for generic external descriptors and symbolic typed child composition. |
| Q-0319 | `auto_partition_test.py` | `adb86ad3fbfabf4f2aab180d3eecc6ffb9daebf5` | Closed for the original runnable scenarios. |
| Q-0328 / Q-0329 | `partition.py` / `partition_test.py` | `8cebbb58095d172f3db564cdfc231537e6214957` / `c6c3236332577115ff6f09e17f340fe79825e1f4` | Reused for packed binding. Whole-file Partition acceptance continues separately. |

The Compiler's layout binds outer groups in declaration order and the
segments within a group from most to least significant. Typed Program
composition reuses Partition and its adjoint, calls the child once, and
keeps unused segments live and unchanged. For left-only binding it omits
repacking and returns the child right ports in declaration order. The
original left-only unused-segment and missing-input validation is retained.
Pure output-only child declarations are outside the admitted source
constructor as well. A partition adds no physical gates or work pool.
Numerical codecs and classical child callbacks remain in Simulation.

The original input-only and ordinary fixtures, both unused segment
placements, and the fully unused `z` word use the same native layout.
The original controlled three-bit exchange binds `a` as `y, ctrl` and
`b` as `x`, with activation `0110`. The tests traverse the unchanged
source Partition and child calls before lowering literal gates. This
retains the source's generic controlled-swap decomposition before the
Cirq export can replace it with another controlled-unitary synthesis.

| Fixture | Physical stream | Independent locked QEC model |
| --- | --- | --- |
| Original one-bit exchanges | Seven T/T-dagger, no measurements or work in either direction | One CSWAP, logical peak three or six, including unused segments |
| Original 10-port exchange, forward | 57 T/T-dagger, nine measurements, three clean work bits | Three CSWAP, nine And, 33 Clifford, nine measurements, peak 13 |
| Original 10-port exchange, adjoint | 33 T/T-dagger, three measurements, same three work bits | Three CSWAP, three And, 11 Clifford, three measurements, peak 13 |

The forward generic Controlled(Swap) invokes one predicate per bit pair.
The locked adjoint declares ControlledViaAnd around the register swap and
computes its predicate once. The compiler supports both emission paths
by reusing the same control and swap primitives. The complete coherent
state agrees with the original classical permutation and exact source
phase in every forward or reverse history. All work bits are zero at the
boundary. Prior typed Program cases independently establish equivalent
direct calls, reordered multi-dimensional ports, workspace ownership and
measured Add feedback.

The original AutoPartition and Partition non-notebook suite has 20
passes, ten mode skips and one expected failure. The Partition notebooks
listed in its test file belong to the next dependency audit. AutoPartition
has no source notebook. Physical execution admits concrete typed Program
ports. The source's generic external object construction and broader
symbolic typed decomposition remain outside the verified native interface.
