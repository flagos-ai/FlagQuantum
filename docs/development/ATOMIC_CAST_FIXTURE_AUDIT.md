# Atomic and cast register fixtures

The five audited Qualtran v0.7.0 files match their inventory blobs at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The native equivalents live
under tests and use the same Compiler, Simulation and Circuit interfaces
as existing fixture consumers.

| Inventory | Evidence |
| --- | --- |
| Q-0486 fixture package | The atom and cast exports are mapped. The many-register, declared-graph, parallel and serial exports await their direct fixture audits. |
| Q-0487 atom | The X, CX and self-adjoint identity fixtures preserve port order, tensor order including the CX input/output interleave, labels, and the identity fixture's adjoint marker. |
| Q-0488 atom tests | The original three tests ran unchanged. The native fixtures apply each complete tensor and append the corresponding instruction stream on coherent input. |
| Q-0489 cast | Closed. Native typed calls preserve the original QFxp signature and Cast/Add/inverse-Cast sequence through approved Proposal 115 admission. |
| Q-0490 cast test | The original structural test passed. The native fixture checks both one- and four-bit call chains and physical binding. Existing `test_register_cast.py` compares all three measurement bits, arbitrary complex states and zero work in the four-bit case. |

## Resource and representation boundaries

The atomic X and identity source fixtures each declare a synthetic
100 T cost. The native model query carries that declaration. The
physical X fixture emits one X, the CX fixture one CX and the identity
fixture emits no instruction. All three emit zero T-family gates and
use no auxiliary wires.

Both casts are metadata operations that preserve the MSB-first wire
order. The four-bit bound addition emits 12 T-family instructions
and three measurements. Its three carry wires return to zero in each
measurement history. The one-bit path emits one CX with no work.

The source fixture declares the second target as unsigned QFxp with fractional
width equal to its word width. Approved Proposal 115 now admits this canonical
storage. `cast_add_fixture` uses the exact typed descriptor and both Cast
functions retain it, removing the earlier opaque substitute. The complete
native signature equals the original serialized signature. Both historical
rejection assertions were reproduced on the batch base before correction.
One- and four-bit source streams match exactly; every small measurement
history agrees on the full unnormalized complex action with clean work zero.
The original structural test remains unchanged and passes.

Q-0489 is accepted as a complete source file, independently of the broader
Cast source's pending signed/empty storage and classical-view contracts.
Zero and negative original widths still construct the three-call root, but
tensor execution fails with IndexError and ValueError respectively. Their source
models contain negative category counts (zero width has -1 And, -10 Clifford
and -1 measurement). Native physical construction refuses nonpositive Add
widths earlier. These source invalid-width diagnostics are retained separately
in `g2-cast-fixture-invalid-widths.log`, alongside the unchanged structural test
in `g2-cast-fixture-original.log`. The accepted raw-value and
7-plus-2 fixed-point numerical evidence remains valid.
