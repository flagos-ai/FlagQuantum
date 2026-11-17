# Shared physical command utilities

K-0044 and K-0045 track the locked `main_util.cc` and `main_util.h`
(blobs `4f4fed8517ab18a479f15f9d79a545134104e94e` and
`63adf218f3dc6a3bb8da90d942ceb564e0c5e3a2`).

## Formatting and initialization

Compiler Formats owns the unsigned literal and lane-word codec consumed
by the fixed-modulus generator and both register sampling boundaries.
It preserves leading binary digits and the high hexadecimal nibble's
width, decimal significant width, underscores, uppercase report
hexadecimal, and empty-word formatting. Sample report flags retain hex
precedence over decimal. Very wide decimal input and output use bounded
conversion chunks when the host's decimal digit guard would reject the
whole conversion.

Simulation owns the 64-shot sample batch and the independent
initialization of the packed quantum/classical state on every batch.
The command layer resolves register declarations before handing off.
Lane zero is least significant. Fixed values are truncated to the
declared register lanes, and later assignments overwrite earlier
assignments to shared physical wires. The native sampling commands
process the final requested lanes directly. The locked executable
fills all 64 lanes and discards unrequested report rows. Each native
random initialization uses its existing independent random stream,
and the sampled values satisfy the same scope and modulus constraints.
Spin keeps its separate persistent benchmark state and mode-specific
widths.

The locked formatter oracle covers 25 small literals and a 4,500-digit
decimal value. Width and binary, decimal and hexadecimal output match.
The source command matrix covers 33 combinations of unsigned formats,
overlapping mixed registers, initialization and a 65-shot partial batch.
Eighteen deterministic accepted reports match byte for byte. Twelve
randomized combinations match the locked value domain and row-wise
feedback. The three source aborts for out-of-width random moduli return
a native validation error. Invalid input is rejected before replacing
the output.

## Command declaration mapping

| Locked declaration | Native module and arguments |
| --- | --- |
| `main_generate_circuit` | `flagquantum.drawer.kmx_generate_command` |
| `main_count_operations` | `flagquantum.compiler.formats.kmx_operation_counts` |
| `main_count_operations_sampled` | `flagquantum.ecosystem.kmx_sampled_counts` |
| `main_sample` | `flagquantum.ecosystem.kmx_sample` |
| `main_spin` | `flagquantum.ecosystem.kmx_spin` |
| `main_diagram` | `flagquantum.drawer.kmx_diagram_command` |
| `main_kmb2kmx` | `flagquantum.compiler.formats.kmb_convert kmb2kmx` |
| `main_kmx2kmb` | `flagquantum.compiler.formats.kmb_convert kmx2kmb` |

Each native module can be invoked with `python -m`. The two conversion
directions share one module. The existing command regressions cover
physical streams, file and pipe boundaries, and output errors.
The helper layer emits no operations. The packed executor's earlier
locked-history test covers complex amplitudes and work-bit cleanup
through append. Sample counts and timings use their existing reporting
contracts.
