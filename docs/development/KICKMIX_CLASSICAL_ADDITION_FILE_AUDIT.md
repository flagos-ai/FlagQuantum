# Mixed-offset addition, subtraction and carry XOR

K-0053 through K-0056 track the locked `gen_iadd_classical.cc`,
`gen_iadd_classical.h`, `gen_iadd_classical.perf.cc` and
`gen_iadd_classical.test.cc` (blobs
`0729f224be462768b49023bab9214637b40f8e96`,
`c4aca3b2b6d77e947ba746abfc40e45a87388073`,
`6d75c228d7bb4a4c3c0a3dbae5e391711ac15bdf` and
`061d25f9cbc02526b4b69b3adeba6f4aa90f8e0f`).

The physical exporters use the existing big-endian carry XOR,
two-clean vented carry, mixed addition and simple measured-carry
schedules. The physical subtraction wrapper retains the nested
addition call between target inversions. Quantum carry, classical carry
and literal carry share the same port boundary. Standalone vented carry
preserves nested marks for high-bit promotion or offset padding.
Short vented erasure preserves the source's distinct guard ordering for
parity feedback and direct phase gates. A nested carry allocator
clears a previously measured scratch bit when the source reuses it.

The full physical comparison covers five source generator paths over
short and padded ports, three offset kinds, three carry kinds, control
and different work sizes. It matches 3,780 streams and 612 mutual
rejections. Original append consumers retain their pre-cleared work
contract. All five unchanged source fuzz tests passed. Native tests
check every projected reset and HMR branch for short mixed addition
and subtraction with a classical borrow or carry. Previously accepted
tests cover simple addition, standalone copy and vented feedback.

The earlier extra-clean prefix witness still demonstrates the locked
source's arithmetic and relative-phase defect. It is separate from
source stream parity and remains open under K-0056.04. The port does
not replace that schedule with an unverified corrected one.

## Generation and serialization benchmarks

Run `python -m benchmarks.benchmark_physical_classical_addition`.
The six locked workloads use 255 target lanes with a 256-bit offset.

| Native workload | Physical records | Source timing boundary |
| --- | ---: | --- |
| Carry XOR with text KMB conversion | 1,762 | Generate and validate, then write KMB |
| Carry XOR with direct column KMB conversion | 1,762 | Generate and write KMB with validation skipped |
| Carry XOR with classical offset bits | 1,792 | Generate raw records |
| Carry XOR with constant offset | 1,762 | Generate raw records |
| Mixed addition with two clean wires | 5,792 | Generate raw records |
| Mixed addition with 100 clean wires | 4,747 | Generate raw records |

A local native run measured median 12.89, 12.59, 12.08, 7.09, 33.29
and 42.97 ms in table order after one warmup and five repetitions.
The native timing includes physical lowering and materialization of
the report input. Both native KMB
routes validate, with the direct-column route skipping a textual
serializing pass over the parsed columns. The locked validation and
skip-validation cases produce identical bytes. Both native routes
produce those same bytes using the shared packet writer.

The classical-offset benchmark creates raw BitIds without reserving
them. Its scratch HMR and feedback CZ use b0, which aliases the lowest
offset bit. The native workload reserves the input width and uses b256.
The two affected records are isolated in the source fixture. The locked
full digest and the digest after only this operand change are both
retained. The other benchmark streams match exactly. For example, the
constant carry stream has 510 CCX, 1,245 CX, 2 X, 1 R, 1 HMR and
1 CZ plus two marks. These are physical counts. The estimator remains
separate.
