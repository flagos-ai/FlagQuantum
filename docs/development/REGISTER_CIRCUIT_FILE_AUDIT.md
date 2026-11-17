# Raw circuit structure and file audit

The locked Kickmix source files covered here are `circuit.cc` (K-0014, blob
`fa9ce4093c1e37c416774b1d87447ad73d64a79d`), `circuit.h` (K-0015, blob
`1456cb523d4a1cb29896a5c2ab09ab4e781dbb53`) and `circuit.test.cc` (K-0017,
blob `f63b40b44c8822344f0a1c3d026f362af4f2341f`). All 48 unchanged C++
tests ran successfully against the locked implementation. Their native
coverage is listed by original test name below.

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
analysis-chart model. A repeated phase circuit emits two CX and two
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

## Physical graphical diagrams

The complete graphical renderer (K-0019, blob
`cf4bd18a609ff39336bf11ceef913dab6fa7d11d`) and its original tests
(K-0020, blob `0441e1299093dd8f545e1ed899cda88e36fc100b`) map to
`drawer/reversible_phase_graphics.py` and the embedded browser viewer.
Both output modes use the same physical column occupancy, unguarded CX
fusion, gate glyphs, register labels, HMR/reset wire intervals, and tag spans.
HTML adds the source's near and far geometry, visible-range selection,
zoom, pan, and URL hash state. The display statistics reuse reaction
depth and count serialized gate classes and physically touched wires. A
visual group does not reduce its constituent physical opcode counts.

The five unchanged original C++ tests passed. Seven locked SVG/HTML cases
compare SVG nodes and their coordinates, near and far geometry, interval
boundaries, tag labels, viewport bounds, and circuit statistics. The
source all-gate SVG contains an unescaped ampersand in the classical
`b1&=!b0` label. The native writer escapes the ampersand and preserves
the displayed text. The HTML payload carries escaped JSON and imports
the SVG fragments as parsed XML. Labels containing script delimiters or
JavaScript interpolation remain data.

Chrome verified mouse-wheel zoom, drag, tag labels, and URL hash updates
against both the locked and native pages. It also verified native hash
restoration and far-view sampling of 2,200 physical records. The headless
probe used the browser timer queue to advance the animation callbacks.

## Circuit ownership and source dispositions

The native circuit is a structural value. Its frozen registers and records
own their labels, ordered physical operands, guards, and exact angles.
Inferred widths include declared register members and operation operands.
Equality compares ordered fields. Iteration and per-record KMX formatting
produce the source order. Copy and deepcopy preserve the full value, and
rebinding or replacing a circuit with an empty value leaves existing
references valid. Composition operates on these values directly.

The unchanged C++ ownership probe copied and assigned the 45-record
all-operation circuit, verified independent aligned buffers, moved both
copies, and verified the moved-from objects were empty. Repeated source
`clear()` releases operation buffers but retains declared registers and
cached widths. The native ownership boundary has two consistent states,
an intact immutable value and an explicitly replaced empty value.
Low-level aligned allocation, manual destruction and pointer diagnostics
map to Python value ownership and validation. Native error messages and
record repr use Python diagnostics. The binary and raw text boundaries
retain explicit index, payload, duplicate-wire, opcode, and record limits.

Eight original positive binary literals are reproduced byte for byte. The
17 original malformed payloads are retained, and the unknown-opcode test
covers every unassigned byte, including the original `0x05`. The locked
2,207,593-byte file is kept in the reference checkout and was verified
independently in both directions. Its 263,405 records are distinct from
the number of canonical KMX lines, which includes declaration lines.

## Original Circuit test map

The source names in this table are from the pinned test file. Test names
starting with `test_` refer to the native unit suites. The external large
fixture is evaluated by replay against the pinned source extension and
the native codec in the same environment.

| Source test | Native evidence |
| --- | --- |
| `circuit.text_diagram` | `test_locked_text_diagram` |
| `circuit.text_diagram_merge_cx` | `test_locked_text_diagram` |
| `circuit.text_diagram_supports_all_gates` | `test_locked_text_diagram` |
| `circuit.text_diagram_z_pow` | `test_locked_text_diagram` |
| `circuit.estimate_ccz_states` | `test_source_file_detection_and_structure` |
| `circuit.reaction_depth` | `test_locked_reaction_depth` |
| `circuit.write_to` | `test_original_all_operation_file_and_record_streams_agree` |
| `circuit.kmb_every_operation` | `test_all_physical_records_are_owned_values` |
| `circuit.kmb_empty` | `test_original_literal_kmb_bytes` |
| `circuit.kmb_x` | `test_original_literal_kmb_bytes` |
| `circuit.kmb_parsing_rejects_unknown_operation` | `test_all_unsupported_opcode_bytes_are_rejected` |
| `circuit.kmb_cx` | `test_original_literal_kmb_bytes` |
| `circuit.kmb_ccz_if` | `test_original_literal_kmb_bytes` |
| `circuit.kmb_cswap_if` | `test_original_literal_kmb_bytes` |
| `circuit.kmb_registers` | `test_original_literal_kmb_bytes` |
| `circuit.kmb_info_tag` | `test_original_literal_kmb_bytes` |
| `circuit.kmb_circuit_with_every_operation` | `test_original_literal_kmb_bytes` |
| `circuit.kmb_validate_unknown_version` | `test_original_malformed_kmb_cases` |
| `circuit.kmb_validate_num_qubits` | `test_original_malformed_kmb_cases` |
| `circuit.kmb_validate_num_bits` | `test_original_malformed_kmb_cases` |
| `circuit.kmb_validate_num_registers` | `test_original_malformed_kmb_cases` |
| `circuit.kmb_validate_num_registers2` | `test_original_malformed_kmb_cases` |
| `circuit.kmb_validate_num_ops` | `test_original_malformed_kmb_cases` |
| `circuit.kmb_validate_num_qqq_ops` | `test_original_malformed_kmb_cases` |
| `circuit.kmb_validate_num_qq_ops` | `test_original_malformed_kmb_cases` |
| `circuit.kmb_validate_num_q_ops` | `test_original_malformed_kmb_cases` |
| `circuit.kmb_validate_num_b_ops` | `test_original_malformed_kmb_cases` |
| `circuit.kmb_validate_num_c_ops` | `test_original_malformed_kmb_cases` |
| `circuit.kmb_validate_magic_bytes` | `test_original_malformed_kmb_cases` |
| `circuit.kmb_validate_non_zero_num_qqq` | `test_original_malformed_kmb_cases` |
| `circuit.kmb_validate_info_tags` | `test_original_malformed_kmb_cases` |
| `circuit.kmx_file_and_stream_agree` | `test_original_all_operation_file_and_record_streams_agree` |
| `circuit.named_registers` | `test_original_named_registers_and_tag_offsets` and `test_original_register_and_tag_refusals` |
| `circuit.info_tags` | `test_original_named_registers_and_tag_offsets` and `test_original_register_and_tag_refusals` |
| `circuit.write_analysis_svg_to_every_gate` | `test_locked_physical_analysis_svg` |
| `circuit.write_analysis_svg_to_info_tags` | `test_locked_physical_analysis_svg` |
| `circuit.compute_max_condition_depth` | `test_source_file_detection_and_structure` |
| `circuit.reaction_depth_bit_invert_preserves_depth` | `test_original_equivalent_rotation_and_invert_depth` |
| `circuit.parse_z_pow` | `test_original_file_text_and_exact_phase_writes` |
| `circuit.parse_z_pow_if` | `test_original_file_text_and_exact_phase_writes` |
| `circuit.parse_multiple_z_pow` | `test_original_file_text_and_exact_phase_writes` |
| `op.kmx_str_works_on_every_op` | `test_all_physical_records_are_owned_values` |
| `circuit.reaction_depth_of_zpow_1_if` | `test_original_equivalent_rotation_and_invert_depth` |
| `circuit.flame_chart_every_op` | `test_locked_physical_analysis_svg` |
| `circuit.compute_has_unknown_ops` | `test_all_unsupported_opcode_bytes_are_rejected` and `test_all_physical_records_are_owned_values` |
| `circuit.parse_big_circuit_kmb` | Locked large KMB replay and native byte-for-byte round trip. |
| `circuit.max_num_registers` | `test_original_header_and_identifier_bounds` and `test_corrupt_kmb_header_or_payload_is_refused` |
| `circuit.validate_max_circuit_instructions_error_message` | `test_original_header_and_identifier_bounds` |
