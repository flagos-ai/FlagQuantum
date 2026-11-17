# Raw circuit structure and file audit

The locked Kickmix source files under review are `circuit.cc` (K-0014, blob
`fa9ce4093c1e37c416774b1d87447ad73d64a79d`), `circuit.h` (K-0015, blob
`1456cb523d4a1cb29896a5c2ab09ab4e781dbb53`) and `circuit.test.cc` (K-0017,
blob `f63b40b44c8822344f0a1c3d026f362af4f2341f`). All 48 unchanged C++
tests ran successfully against the locked implementation. The three source
files remain open pending HTML and SVG circuit drawing and complete native
test mapping.

## Structure and composition

The raw format boundary in `compiler/formats/kmx_circuit.py` uses the same
validated operand columns as the KMB codec. It detects binary format when
the first byte is the binary magic prefix and verifies the remaining magic
before entering the binary decoder. Empty input yields empty KMX. The text
reader supports the source's CR and LF handling and normalizes file-specific
raw hash characters in quoted names or tags. Text consumed by the KMX
string encoder follows the source string constructor's comment rules.

Structural statistics include register and operand widths, operation
columns, classical destinations and conditions, exact-angle records, UTF-8
tag bytes and peak condition nesting. Standalone unmatched POP leaves
structural scope depth at zero. Executable KMX is validated separately by
the existing reversible-phase program. Tag-only declared widths remain
recoverable by the executable decoder even when raw columns imply zero
qubits and bits.

Concatenation retains physical wire IDs and uses left register metadata if
present, otherwise right metadata. Repetition preserves register metadata
at zero repetitions and copies operation, angle and tag columns in order.
The existing typed execution entry point binds the supported raw operation
subset after it has received its executable header.

The CCZ-state estimate counts three-qubit operation records and half of
rotation records rounded up. Guarded records count once. It is separate
from the actual opcode histogram, reaction depth and the per-angle
analysis-chart model. A composed with-phase example emits two CX and two
Z_POW records and has model weight one. Its complete complex state agrees
with the analytic result after append and execution. No work wires are
needed by that example.

## Physical resource analysis SVG

The drawer reads the same validated records used to render raw KMX. It
preserves operation order, guards, exact angles, and decoded tag text. The
chart takes an active-wire snapshot when entering a block, adds touched
wires to the innermost frame, and removes HMR targets from the active set.
An exit closes the nearest matching block and any inner frames. Unmatched
exits leave the stack intact. End of input closes any remaining frames.
Groups accumulate hits and weights by the complete source colon path.

The chart assumes one CCZ state per CCX, CCZ, or CSWAP, half a state per
odd 45-degree multiple, four for finer rotations, and zero for 90-degree
multiples. These are modeled weights. Serialized guards and measurements
retain their positions and the renderer does not execute operations.

Six locked SVG captures verify ordered rectangles, markers, labels and
numeric attributes, including nested marks, malformed marks, repeated
groups, the mixed gate stream, and exact-angle costs. The source emits
nonfinite vertical coordinates for an empty circuit. Native empty output
uses finite zero coordinates. Both the raw text and KMB round-trip routes
produce the same native chart.

## Physical text diagrams

The complete locked `circuit_diagram.cc` (K-0018, blob
`9bc1d00dec8536bf941ab4d2ba4d63605a25b227`) maps to the text diagram drawer
and shared raw KMX records. The drawer lays out byte-width register labels,
fusion of consecutive controlled X and Z records, conditions, measurement
destinations, reset wire transitions, and classical metadata. It applies
Unicode glyphs after layout. The source string, FILE, and ostream routes
agree in the reference probe for the applicable mode. Python's single
string result can be written to a file or stream with the caller's encoding.

The 11 frozen pairs replay ASCII and Unicode output exactly, including the
five original golden diagrams, the complete opcode stream, UTF-8 names,
empty circuits, and classical-only input. Raw KMX and the KMB bridge produce
the same output. The diagram changes presentation only. In particular,
merging the visual controls preserves all original CX and CZ records.

## Reference evidence and remaining ownership

The native structure fixture records 20 cases captured by an unchanged
locked Circuit reader, including all four original state-weight cases and
all 12 original condition-depth cases. The original 17 malformed binary
byte fixtures are preserved and rejected. Four composition paths were
captured from the source operators and replayed locally. Existing KMB
round-trip, opcode, reaction-depth, phase, and large-file evidence remains
applicable.

| Reference areas | Native disposition |
| --- | --- |
| Auto-detection, widths, columns, resources, scope depth and composition | Raw format and structure tests cover locked fixtures and the source ordering. |
| KMB layout, refusals and full-file round trip | Existing complete opcode and large-file fixtures plus the original malformed payloads. Map remaining original positive cases and limit assertions individually before closing K-0017. |
| Named registers, tag rendering in KMX, ordinary stream output and exact rotation spelling | Existing codecs and exact angle owner. The full native assertion map remains open. |
| Reaction depth and touched qubits | Reuse the reaction-depth implementation and accepted source fixtures. |
| Text diagrams and control merging | K-0018 closed by the exact ASCII and Unicode layout fixtures. |
| HTML and circuit SVG diagrams | K-0019 and K-0020 own the remaining graphical path. |
| Analysis SVG and flame-chart assertions | Physical block walk, per-angle CCZ model, and SVG structure verified against the locked reference. The logical flame graph retains its own model. |
| C++ aligned mutable buffers, copy and move | Native raw circuit operations use owned Python structures. Mutation and low-level allocation parity require a separate mapping. |
