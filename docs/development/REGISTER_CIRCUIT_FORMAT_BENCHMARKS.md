# Physical circuit format benchmarks

The native runner maps every workload in the locked
`circuit.perf.cc` (K-0016, blob `0836ad8059a722e7be33c135c1ae938f003dd7f0`)
and `mutable_circuit.perf.cc` (K-0025, blob
`e6f8ded90d5ffae590d3bdc0fa972d6eb575bf79`). It uses the compiler's
physical circuit values and mutable builder. The binary path reads
packets directly into the immutable circuit without formatting or
reparsing KMX. The two binary entry points share packet, opcode and width validation.

Run from the repository root with the locked `testdata` directory.

```console
python -m benchmarks.benchmark_raw_circuit_formats /path/to/testdata --warmup 1 --repeats 3
```

Setup verifies that text append, stream append and direct binary read
produce equal frozen circuits. Re-encoding verifies the complete KMB
payload. The full fixture contains 263,405 operation records in
3,379,783 KMX bytes or 2,207,593 KMB bytes. The condition workload
contains 4,004 records with source scope depth 1,001. The runner
checks the result of every measured repetition and reports record and
byte rates where applicable. These workloads measure parsing and
structure. They do not execute a circuit.

## Measurements

Local arm64 macOS 27.0 run with CPython 3.12.14, one warmup, and the
median of three native repetitions. The unchanged pinned C++ workload
values below are its reported local harness measurements with a
0.01-second run target. The target values are the source benchmark's
built-in reference thresholds.

| Workload | Timed native boundary | Locked C++ | C++ target | FlagQuantum |
| --- | --- | ---: | ---: | ---: |
| Mutable KMX text append | Clear builder and append predecoded text | 7.9 ms | 6.0 ms | 363 ms |
| Mutable KMX file append | Rewind, clear and append binary text stream | 55 ms | 6.0 ms | 487 ms |
| Mutable-source KMB file load | Rewind, read packets, validate and materialize immutable records | 500 µs | 350 µs | 289 ms |
| Frozen condition depth | Scan existing records with no parse or freeze | 4.0 µs | 1.7 µs | 139 µs |
| Frozen KMX load | Rewind, clear, append and freeze | 46 ms | 16 ms | 737 ms |
| Frozen KMB load | Rewind, read packets, validate and materialize immutable records | 380 µs | 640 µs | 290 ms |

Python measurement includes native object creation or iteration where
specified. The C++ source uses aligned arrays and its own timing
harness. The six native paths are slower in this run. The evidence
covers workload and result parity, with separate runtime measurements
for the two representations.
