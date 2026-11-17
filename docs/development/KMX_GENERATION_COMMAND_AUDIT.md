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
Thirteen cases cover both serializations, zero-width and width-one table
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

## Open source paths

| Source mode | Remaining physical alignment |
| --- | --- |
| `flip_if_lt` | The default-clean quantum path dispatches through the measured comparison ladder in `gen_cmp.cc`. The existing quantum comparison emitter follows the other `gen_cmp_qq.cc` route. Port the dispatch, ladder, and tolerance recursion. |
| `iadd_classical` | Match clean-wire resets at their allocation sites and preserve the HMR result as the outer condition in erasure feedback. |
| `iadd_classical_simple` | Match reset interleaving and HMR condition nesting in the one-clean-wire-per-lane schedule. |
| `imul2_mod` | Match helper resets, nested component marks, outer HMR conditions, and result slot allocation after the full physical modulus register, including fixed-endpoint slots. Cover both constant and register modulus forms. |

The K-0036 source inventory remains open for these four modes.
