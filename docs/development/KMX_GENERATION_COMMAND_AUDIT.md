# Physical circuit generation

K-0036 tracks the complete `main_generate_circuit.cc` at locked Kickmix blob
`8f77fa1c5512db186ab2c75873bf96f8f752339c`. The native entry point is
`python -m flagquantum.drawer.kmx_generate_command`.

- The command owns option parsing and output. The compiler binds register
  IDs and selects the native emitters. The add and table emitters retain
  the existing register and measurement contracts used by append callers.
- The source declaration order is least-significant lane first. The native
  emitters consume big-endian ports. The binder reverses each word and
  table row independently. Reserved clean and dirty wires stay outside the
  logical declarations. Physical width is inferred from the declarations
  and actual operation operands.
- Physical in-place addition emits the source's explicit resets on leased
  clean wires. Ordinary arithmetic callers still supply pre-cleared work.
  Forward lookup records the actual tree of recursive calls. Unlookup
  records its unary conversion, phase lookup and inverse conversion.
  These marks use lowered operation offsets and add no executable gates.
- Both formats preserve measurement slot identity and the ordering of
  condition scopes and feedback gates. KMB output uses the shared v2
  serializer. Unicode diagrams take precedence over KMB. The analysis SVG
  is independent of normal output. Native output validates before writing;
  `--validate` is accepted and native validation also runs by default.

## Verification

The checked-in CLI capture was generated with the unchanged source command.
The initial thirteen cases cover both serializations, zero-width and width-one table
paths, controlled lookup, measured unlookup, and controlled and uncontrolled
addition with borrowed and measured carry. The source's diagram and analysis
outputs are also captured. A separate grid sweeps widths 0–4, admissible
clean counts, both controls where supported, both space modes, and output
widths 0–2. The 462 admitted cases match both KMX and KMB byte for byte with two
reserved dirty qubits. With no dirty reservation, all 462 admitted KMX
streams also match.

The existing `test_kickmix_addition.py`, `test_recursive_lookup.py` and
`test_measured_unlookup.py` exercise coherent inputs, carry and erasure
feedback, and scratch restoration on the same operation emitters. The
generation tests compare the complete raw streams, including layouts,
resets and marks. In the three-bit example the uncontrolled physical stream
has 15 CX and 3 CCX, and the controlled stream has 10 CX and 11 CCX. In the
two-address-bit, one-output-bit case, lookup has 25 physical operations and
unlookup has 30, excluding call marks and register declarations. The analysis
SVG reports the source's CCZ-state estimate separately.

## Quantum less-than generation

The fourth mode uses the shared comparator schedules. The locked source calls
`gen_cmp.cc` (blob `fd73b653bd9ef09000e56021aab804a0b2733cf3`) and dispatches
to `gen_cmp_qq.cc` (blob `7c618915a2b61ea26472652cf88bc79fe17c4096`) for
infinite tolerance with fewer clean wires than input lanes. Finite tolerances
below the word width either skip the operation or recursively compare the
most significant `ceil(btol) - 1` lanes. The control, output and workspace keep
their original wire IDs. The physical exporter retains allocations as R
records and encloses the default ladder's feedback in one condition scope per
erased carry; the narrow route retains inline guards. Reserved dirty wires
are unused in these quantum routes.

Twelve captured comparator cases cover both serializations, nested calls,
empty widths, work allocation, and exact and reduced comparisons. An
independent 240-case sweep matches every one of 216 admitted KMX streams
and all 80 captured KMB streams. The native path rejects the same twenty
insufficient-clean inputs that abort the source, plus four finite-tolerance,
two-bit, zero-clean inputs where the source indexes nonexistent carry wires
and segfaults. The native validation error preserves the output file.

The execution test projects all reset and erasure branches on complex
superpositions of each logical register, output, and optional control.
For the three-bit controlled default case, the primitive stream has 14 CX,
5 CCX, 4 CZ, 3 explicit R and 3 HMR. POP/PUSH records and component marks
are separate. The three-bit controlled narrow case has 13 CX and 9 CCX
with zero resets or outcomes. Idle or borrowed storage is restored and
all active clean wires end in zero. These are emitted record counts, with
the CCZ-state estimate still provided separately through the resource chart.

## Open source paths

| Source mode | Remaining physical alignment |
| --- | --- |
| `iadd_classical` | Match clean-wire resets at their allocation sites and preserve the HMR result as the outer condition in erasure feedback. |
| `iadd_classical_simple` | Match reset interleaving and HMR condition nesting in the one-clean-wire-per-lane schedule. |
| `imul2_mod` | Match helper resets, nested component marks, outer HMR conditions, and result slot allocation after the full physical modulus register, including fixed-endpoint slots. Cover both constant and register modulus forms. |

The K-0036 source inventory remains open for these three modes.
