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

## Classical-offset addition generation

Both classical modes reuse the arithmetic schedules from
`gen_iadd_classical.cc` (blob `0729f224be462768b49023bab9214637b40f8e96`).
The general mode declares named target and offset registers and a literal
zero carry-in. The simple mode declares a separate quantum carry-in.
Clean and dirty reservations precede the optional control. The exporter
records helper resets, nested carry calls and outcome scopes at their
generator boundaries. It retains the physical classical slot numbers,
including unreferenced allocator slots and zero-initialized vent slots that
need no emitted stores in the source file format.

The initial grid checks 132 admitted KMX streams and 35 KMB captures.
It also compares rejection for all 36 source aborts. The additional
126-case sweep covers widths 4–10 and from two to ten clean wires.
Every one of its 98 admitted cases matches both formats, including mixed
recursive-prefix and vented routes. The remaining 28 cases reject in
both implementations. Fourteen representative locked cases are fixtures.

For the simple and short/vented general paths, exhaustive reset and HMR
histories retain the full complex amplitudes on superposed targets and
enabled/disabled inputs, with clean work returning to zero and borrowed
dirty work restored. The known extra-clean prefix arithmetic and phase
defect remains present in the unchanged source schedule. The physical
route matches the source records for that branch. Its arithmetic defect
is tracked in `test_kickmix_mixed_addition.py`.

The controlled four-bit vented example uses 27 CX, 6 CCX, 3 CZ, 2 R,
2 HMR and one BIT_INVERT. Condition scopes and component marks are
reported separately from these physical operations. The analysis SVG
continues to report the source's CCZ-state estimate separately.

## Modular doubling generation

Both fixed modulus and classical-register modulus forms follow
`gen_imul2_mod.cc` (blob `f2c07c7825e05cd44bd2f793c0c8a826cf426cb5`).
The register form keeps both forced endpoint slots in the declaration so
result allocation starts beyond the entire physical word. The fixed form
accepts decimal, binary and hexadecimal literals and retains binary width.
A control register follows the modulus declaration and precedes reserved
work. The exact path retains rotate tags, helper resets, and measurement
feedback. The finite-tolerance path composes comparison, prefix phase,
mixed addition and rotation through their shared emitters. A recycled
classical vent slot emits the source's explicit store before the vented call.

The first locked CLI grid contains 168 cases. Its 78 admitted KMX and 24
captured KMB streams match. Native validation rejects all 90 source
aborts. The extended register-modulus, fixed-modulus, and wide all-ones
sweeps match 460 more KMX and 460 KMB streams, exercising widths through
33 and both exact and low-workspace paths. Fourteen locked fixtures
cover output bytes and argument boundaries. Small exact doubling on
fixed and variable moduli also checks all reset and HMR histories with
complex superpositions, including disabled control and work reset to zero.

With constant modulus five and width three, the exact uncontrolled route
uses 2 SWAP, 2 CX, 2 CCX, 1 reset and 1 HMR. Register modulus width three
with a quantum control emits 2 CSWAP, 6 CX, 4 CCX, 3 resets and 3 HMR.
These are physical counts. The analysis output separately reports the
source's CCZ-state estimate.

The locked all-ones finite-tolerance route reads an empty span while
trimming the complemented half-modulus. At tested four-lane allocation
boundaries it pads an empty word; elsewhere it retains the high bit.
The physical export reproduces the observed fixture behavior in bounds.
The existing algorithm route keeps its explicit rejection of the empty
complemented word. Physical compatibility here records the locked
runtime's observed behavior under this source defect.

K-0036 is accepted for all seven generation modes. The retained upstream
defects and native validation behavior are documented above.
