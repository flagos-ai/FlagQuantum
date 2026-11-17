# Quantum increment generation

K-0049 through K-0052 track the locked `gen_iadd1.cc`, `gen_iadd1.h`,
`gen_iadd1.perf.cc` and `gen_iadd1.test.cc` (blobs
`890b484fb6043d45d8e60a91dd241ab8092f7b21`,
`cfa4ce940d855d8706e32ce55eb9885b602108dc`,
`45581bf3792f350234c0613708117b9be43cf5bd` and
`cce0aee709048b56188ea0a0f3a2de4dc3c4db4e`).

The big-endian increment port walks low lanes first. Without an
external control, the low bit controls the upper-word carry ladder
and is then flipped. The controlled form keeps the enable untouched.
The measured ladder takes max(0, n−2) clean carries with an external
control and max(0, n−3) without it. It reuses one result slot and clears
the carries in reverse order with phase feedback.

The native physical boundary emits the allocation resets from the
same schedule, matching `iadd1` and `ciadd1` marks. A zero-width input
has no mark. Algorithm append continues to use its pre-cleared workspace
contract. Extra clean wires remain untouched in both paths.

## Verification

Both unchanged source tests pass, covering the controlled and
uncontrolled fuzz paths. The 72 captured full stream fingerprints and
opcode multisets cover widths 0–16 and 256, with and without control and
with extra clean wires. They preserve operands, feedback, reset placement
and marks. Only the raw register declarations are removed from the
comparison. The earlier 68 snapshots remain as the oracle for the
pre-cleared append path. Native execution tests project all reset and
measurement branches for representative four-bit physical circuits and
compare unnormalized complex amplitudes, retained enable, and cleared work.

## Width-256 generation benchmark

Run `python -m benchmarks.benchmark_physical_increment`. Both workloads
reserve 256 clean wires and produce the source record counts, including
two block marks each.

| Workload | Records | CCX | CX | CZ | R | HMR | X |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unconditional | 1,270 | 254 | 254 | 253 | 253 | 253 | 1 |
| Controlled | 1,274 | 255 | 255 | 254 | 254 | 254 | 0 |

The locked C++ benchmark on the reference host measured 750 ns and
740 ns respectively, with targets of 580 ns and 610 ns. A local native
run measured median 7.88 ms and 8.43 ms across 25 repetitions after
three warmups. The C++ timer measures direct mutable record generation.
The native timer covers instruction generation and lowering to the
validated physical program. Serialization and result verification
are outside that timer. The measured runtimes describe these distinct
boundaries. The table reports physical records and primitive operations.
The resource model remains separate.
