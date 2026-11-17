# General comparator file audit

Kickmix reference commit `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.

| Ledger | Source | Frozen blob | Native owner |
| --- | --- | --- | --- |
| K-0063 | `gen_cmp.cc` | `fd73b653bd9ef09000e56021aab804a0b2733cf3` | Compiler comparison |
| K-0064 | `gen_cmp.h` | `af8de480dcd72335eb88c4fddcf548846259e01f` | Compiler comparison |
| K-0065 | `gen_cmp.perf.cc` | `cbe7de42422259b6236960957c5964b4605483eb` | `benchmarks/benchmark_physical_comparison.py` |
| K-0066 | `gen_cmp.test.cc` | `66b33fcaa2726e9aef3568969de15acf77b8d140` | `tests/unit/test_kickmix_comparison*.py` |

The five relations consume equal-width, big-endian words. The right operand can
contain constants, external classical bits and quantum wires. Output is XORed
into a target qubit or applied as a sign on the whole state. Explicitly bound
clean wires are leased by the selected comparator. The physical adapter puts
result slots after external classical inputs and preserves the input and
unused quantum bindings. It keeps the ordered operands, nested call ranges,
actual resets, HMR operations and per-carry feedback conditions. The existing
append schedules continue to use pre-cleared work.

The full comparison ladder, dedicated quantum dispatch and three-clean-wire
classical route retain their original selection rules. Finite tolerance can
produce an empty comparison or a recursive comparison over the high prefix.
Greater-than temporarily inverts the equality input, compares and complements
the output, then restores the input. The source leaves mutable equality aliases
unspecified. The native greater-than entry rejects such aliases. Direct
standalone entry and diagram coverage of the quantum and three-wire generators
remains tracked by K-0067 through K-0072.

The complete record fixture covers all relations, both output modes, four
right-word layouts, constant or quantum control, external equality, finite
precision and two exact 256-bit workloads. Source register declarations and the
native transport header are excluded. Ordered physical record digests retain
marks, classical scopes, resets, measurements and operands. Regenerate with
`python tests/unit/fixtures/generate_kickmix_comparison_file.py --source-root <locked-kickmix-root> --output tests/unit/fixtures/kickmix_comparison_file.json`.

All nine unchanged `icmp` tests passed, including two source CCZ-state-weight
checks and seven fuzz tests. Earlier append validation supplies small exhaustive
basis, exact complex branch, and cleanup evidence. New physical tests check
every reset/HMR branch of small coherent cases, full complex amplitudes,
phase sign, output XOR, and zero work.

| Workload | Physical records | CCX | CCZ | R | HMR | Call marks | Source benchmark | Native construction and lowering |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical RHS, 256 clean wires | 3,834 | 255 | 0 | 255 | 255 | 2 | 1.3 μs | 54.90 ms |
| Classical RHS, three clean wires | 3,140 | 516 | 2 | 17 | 17 | 6 | 3.3 μs | 40.47 ms |

The records include condition scopes. Source timing covers mutable records.
Native timing is the median of 25 iterations after three warmups and includes
instruction construction, validation and lowering. Resource figures in the
table are emitted counts. The source CCZ-state-weight checks use a separate
estimator. Both native workloads run with
`python -m benchmarks.benchmark_physical_comparison`.
