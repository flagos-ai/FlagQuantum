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
| Q-0489 cast | Native typed calls preserve the sequence Cast, Add, inverse Cast and the stored bits. The fixed-point executable port remains an admission boundary. |
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

The source fixture declares the second target as unsigned QFxp with
fractional width equal to its word width. FlagQuantum can declare the
corresponding metadata and interpret the raw result. The typed Program
binder currently declines fixed-point descriptors. The test exercises
that rejection and checks the same raw word through the admitted
opaque port. A seven-plus-two word example recovers a fixed-point
result of 9/256. The test does not imply fixed-point Program
admission for other consumers.
