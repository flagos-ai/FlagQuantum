# Physical operation counts

The command implements locked K-0030/K-0031,
`main_count_operations.cc` (blob
`3571d2ff92e27bed11f627a86f320701e2335e30`) and its original test
(blob `98b1896e4cca070500d0b1b22e9f0e2db278ae4d`).

```console
python -m flagquantum.compiler.formats.kmx_operation_counts --in circuit.kmb --groups
```

Input defaults to stdin and results default to stdout. `--in` and
`--out` accept file paths. The command detects KMX or KMB input,
preserves file-style metadata syntax, and uses the direct immutable
binary reader. It constructs the result before opening the output
and checks write completion. Argument and input errors return
status 1 with a diagnostic on stderr. Native argument diagnostics
follow the local parser.

Every physical record contributes one count. Inline-guarded and
unguarded variants share rows. The three debug classes share the
source's `DEBUG_PRINT_EMPTY` row. They contribute to the overall
total and are omitted from the six category subtotal rows.
`--groups` appends wire widths, category subtotals and the physical
record total. `--magic` takes precedence and prints the number of
CCX, CCZ, CSWAP and Z_POW records, regardless of angle or guard.
This command reports physical records. The CCZ-state estimate and
branch-dependent execution statistics are separate.

The unchanged all-operation C++ test passed. The frozen source
captures preserve every row and its padding for empty, all-operation,
guarded and large binary inputs. The native tests replay all four
flag combinations on small text and binary fixtures, the original
two-path CLI invocation, a binary stdin case and error handling.
The full locked binary fixture produced the same grouped report
byte for byte through the native file CLI, with 263,405 total and
16,756 physical magic records.
