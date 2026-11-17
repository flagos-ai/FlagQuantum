# Binary-field addition file audit

Kickmix reference commit `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.

| Ledger | Source | Frozen blob |
| --- | --- | --- |
| K-0077 | `gen_gf_iadd.cc` | `4fec5c24910e0cb9f605d2572dc8e334f1e0e59b` |
| K-0078 | `gen_gf_iadd.h` | `f6b85df67b279281efd3a3b670e4c3d17d0a0739` |
| K-0079 | `gen_gf_iadd.perf.cc` | `f74d94024f8664f291245a1055f88dcc48d54e81` |
| K-0080 | `gen_gf_iadd.test.cc` | `d0c6fdf144eb1dcdc5c18814093c9a0f00d83729` |

The existing coefficient generator accepts equal-width big-endian words.
It walks them least significant first and XORs the offset into the target.
Operands may be reversed or strided views of physical wires. A quantum
offset or enable is disjoint from the target. The source header describes
overlap as a precondition and leaves the scan to callers. FlagQuantum
validates target and quantum control disjointness before emission. Classical
slots are read-only and may be reused, including when the outer guard also
serves as a coefficient.

The physical entry in
`compiler/registers/finite_field/physical_field_addition.py` reuses that
schedule and retains its call marks. A classical control wraps the whole
source call, including the marks, with a condition scope. An identical
coefficient predicate inside that scope is still emitted. On empty words,
the generator emits no marks and the classical binding still emits its
outer scope. A false control emits no records. Neither path leases work.

The immutable source fixture covers 111 complete streams, including the
two width-512 layouts. It removes the source register declarations and
native transport header. It compares all remaining gate operands, scopes
and marks. The locked Python binding is also compared directly for three
outer guard choices. Six unchanged C++ tests passed, including the
strided/reversed view case. The append tests and new physical branch
checks cover mixed coefficients on coherent inputs with all small
classical measurement histories. The addition itself emits no measurement.

| Width-512 workload | Emitted gates | Mark records | Source mutable generation | Native construction and lowering |
| --- | ---: | ---: | ---: | ---: |
| Uncontrolled | 512 CX | 2 | 170 ns | 2.97 ms |
| Quantum controlled | 512 CCX | 2 | 710 ns | 3.84 ms |

Native timings are medians over 25 iterations after three warmups with
concurrent local checks. Gate counts are emitted records. Resource model
conversions are outside these measurements.

Run with `python -m benchmarks.benchmark_physical_field_addition`.
Regenerate with
`python tests/unit/fixtures/generate_kickmix_field_addition_file.py --source-root <locked-kickmix-root> --output tests/unit/fixtures/kickmix_field_addition_file.json`.
