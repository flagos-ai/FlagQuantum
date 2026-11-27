# QEC private-binding retirement

QEC subset of the complete independent private-binding closure, implemented
after CI-qualified PR1336 head `4f04ec407066f7b93582713d05a867b6cc4f6bee`.
The child retains its own exact-head CI requirement.
Nine unused import aliases are retired without forwarding. Existing algorithms,
public exports, registered pickle identities, precision, scientific fixtures and
original assertions remain unchanged. The five changed Python ASTs are equal
after imports are excluded. The only operational consumer imports the existing
Simulation fault-signature owner directly.

## Exact retained compatibility obligations

Private spelling alone does not authorize removal. The existing QEC retirement
contract preserves surviving pickle identities, frozen surviving namespaces and
specific replacement owners for historical definitions. The existing fixed
fixtures and `tests/qec/_retirement.py` enforce all three. Thirty-three existing
bindings below remain; none is newly introduced by this migration. The broader
102-entry historical facade inventory remains unchanged and is not represented
as an exhaustive inventory of all QEC compatibility globals.

| Module | Binding | Existing concrete basis |
| --- | --- | --- |
| `flagquantum.qec.adapters` | `_BOUNDARY` | Frozen surviving namespace: tests/qec/fixtures/restructure_006_metadata.json |
| `flagquantum.qec.adapters` | `_EXTRA_NAME` | Frozen surviving namespace: tests/qec/fixtures/restructure_006_metadata.json |
| `flagquantum.qec.adapters` | `_pymatching` | Registered pickle flagquantum.qec.adapters._pymatching; Frozen surviving namespace: tests/qec/fixtures/restructure_006_metadata.json; Frozen replacement definition: flagquantum.qec.adapters._pymatching in tests/qec/fixtures/restructure_006_metadata.json |
| `flagquantum.qec.adapters` | `_translate` | Registered pickle flagquantum.qec.adapters._translate; Frozen surviving namespace: tests/qec/fixtures/restructure_006_metadata.json; Frozen replacement definition: flagquantum.qec.adapters._translate in tests/qec/fixtures/restructure_006_metadata.json |
| `flagquantum.qec.adapters` | `_defects` | Frozen surviving namespace: tests/qec/fixtures/restructure_006_metadata.json |
| `flagquantum.qec.decoders` | `_repetition_wire` | Registered pickle flagquantum.qec.decoders._repetition_wire; Frozen surviving namespace: tests/qec/fixtures/restructure_006_metadata.json; Frozen replacement definition: flagquantum.qec.decoders._repetition_wire in tests/qec/fixtures/restructure_006_metadata.json |
| `flagquantum.qec.decoders` | `_validate_repetition_history` | Registered pickle flagquantum.qec.decoders._validate_repetition_history; Frozen surviving namespace: tests/qec/fixtures/restructure_006_metadata.json; Frozen replacement definition: flagquantum.qec.decoders._validate_repetition_history in tests/qec/fixtures/restructure_006_metadata.json |
| `flagquantum.qec.dem` | `_memory_circuit_entries` | Frozen surviving namespace: tests/qec/fixtures/restructure_005_metadata.json |
| `flagquantum.qec.dem` | `_code_matrix_entries` | Frozen surviving namespace: tests/qec/fixtures/restructure_005_metadata.json |
| `flagquantum.qec.dem` | `_combine_two` | Registered pickle flagquantum.qec.dem._combine_two; Frozen surviving namespace: tests/qec/fixtures/restructure_005_metadata.json; Frozen replacement definition: flagquantum.qec.dem._combine_two in tests/qec/fixtures/restructure_005_metadata.json |
| `flagquantum.qec.dem` | `_count` | Registered pickle flagquantum.qec.dem._count; Frozen surviving namespace: tests/qec/fixtures/restructure_005_metadata.json; Frozen replacement definition: flagquantum.qec.dem._count in tests/qec/fixtures/restructure_005_metadata.json |
| `flagquantum.qec.dem` | `_error_id` | Registered pickle flagquantum.qec.dem._error_id; Frozen surviving namespace: tests/qec/fixtures/restructure_005_metadata.json; Frozen replacement definition: flagquantum.qec.dem._error_id in tests/qec/fixtures/restructure_005_metadata.json |
| `flagquantum.qec.dem` | `_normalized_indices` | Registered pickle flagquantum.qec.dem._normalized_indices; Frozen surviving namespace: tests/qec/fixtures/restructure_005_metadata.json; Frozen replacement definition: flagquantum.qec.dem._normalized_indices in tests/qec/fixtures/restructure_005_metadata.json |
| `flagquantum.qec.dem` | `_probability` | Registered pickle flagquantum.qec.dem._probability; Frozen surviving namespace: tests/qec/fixtures/restructure_005_metadata.json; Frozen replacement definition: flagquantum.qec.dem._probability in tests/qec/fixtures/restructure_005_metadata.json |
| `flagquantum.qec.dem` | `_DETECTOR_PREFIX` | Frozen surviving namespace: tests/qec/fixtures/restructure_005_metadata.json |
| `flagquantum.qec.dem` | `_OBSERVABLE_PREFIX` | Frozen surviving namespace: tests/qec/fixtures/restructure_005_metadata.json |
| `flagquantum.qec.dem` | `_parse_stim_text` | Frozen surviving namespace: tests/qec/fixtures/restructure_005_metadata.json |
| `flagquantum.qec.codes` | `_block_width` | Registered pickle flagquantum.qec.codes._block_width |
| `flagquantum.qec.codes` | `_mask` | Registered pickle flagquantum.qec.codes._mask |
| `flagquantum.qec.codes` | `_matrix_rows` | Registered pickle flagquantum.qec.codes._matrix_rows |
| `flagquantum.qec.codes` | `_minimum_logical_weight` | Registered pickle flagquantum.qec.codes._minimum_logical_weight |
| `flagquantum.qec.codes` | `_odd_overlap` | Registered pickle flagquantum.qec.codes._odd_overlap |
| `flagquantum.qec.codes` | `_refuse_non_logicals` | Registered pickle flagquantum.qec.codes._refuse_non_logicals |
| `flagquantum.qec.codes` | `_supports` | Registered pickle flagquantum.qec.codes._supports |
| `flagquantum.qec.codes` | `_surface_ancilla_sites` | Registered pickle flagquantum.qec.codes._surface_ancilla_sites |
| `flagquantum.qec.dem.construction` | `_checked_round` | Frozen replacement definition: flagquantum.qec.dem_construction._checked_round in tests/qec/fixtures/restructure_005_metadata.json |
| `flagquantum.qec.dem.construction` | `_checked_wire` | Frozen replacement definition: flagquantum.qec.dem_construction._checked_wire in tests/qec/fixtures/restructure_005_metadata.json |
| `flagquantum.qec.dem.construction` | `_executed_flip_set` | Frozen replacement definition: flagquantum.qec.dem_construction._executed_flip_set in tests/qec/fixtures/restructure_005_metadata.json |
| `flagquantum.qec.dem.construction` | `_forced_flip` | Frozen replacement definition: flagquantum.qec.dem_construction._forced_flip in tests/qec/fixtures/restructure_005_metadata.json |
| `flagquantum.qec.dem.construction` | `_forced_signature` | Frozen replacement definition: flagquantum.qec.dem_construction._forced_signature in tests/qec/fixtures/restructure_005_metadata.json |
| `flagquantum.qec.dem.construction` | `_inject_data_flip` | Frozen replacement definition: flagquantum.qec.dem_construction._inject_data_flip in tests/qec/fixtures/restructure_005_metadata.json |
| `flagquantum.qec.dem.construction` | `_inject_measurement_flip` | Frozen replacement definition: flagquantum.qec.dem_construction._inject_measurement_flip in tests/qec/fixtures/restructure_005_metadata.json |
| `flagquantum.qec.dem.construction` | `_lowered_source` | Frozen replacement definition: flagquantum.qec.dem_construction._lowered_source in tests/qec/fixtures/restructure_005_metadata.json |

API/Core owners retain these contract-owned locators. Removal requires an
explicit API-owner-approved contract amendment and the applicable release
migration; this structural batch authorizes neither. Existing retired modules
remain absent and their old pickles retain the originally approved refusal.

An earlier broader source preview removed seventeen aliases. Three unchanged
QEC scenarios failed because eight of those aliases are fixed historical
replacement definitions. Those eight bindings were restored. The original
failure is retained, and neither the fixture, resolver, contract nor assertion
was changed. Corrected focused validation passes 157 tests with two existing
conditional skips. This is preview evidence, not formal or pushed-head CI.

## Retired import aliases

| Source module | Binding | Existing defining owner |
| --- | --- | --- |
| `flagquantum.qec.execution.signatures` | `_pauli_channel_terms` | `flagquantum.simulation.stabilizer._fault_signatures._pauli_channel_terms` |
| `flagquantum.qec.dem.matrices` | `_binary_matrix` | `flagquantum.qec.codes.matrices._binary_matrix` |
| `flagquantum.qec.dem.construction` | `_binary_matrix` | `flagquantum.qec.codes.matrices._binary_matrix` |
| `flagquantum.qec.dem.construction` | `_code_matrix_entries` | `flagquantum.qec.dem.matrices._code_matrix_entries` |
| `flagquantum.qec.dem.construction` | `_column_supports` | `flagquantum.qec.dem.matrices._column_supports` |
| `flagquantum.qec.dem.construction` | `_FAULT_KIND_BY_LETTER` | `flagquantum.qec.execution.signatures._FAULT_KIND_BY_LETTER` |
| `flagquantum.qec.codes` | `_DISTANCE_SEARCH_WEIGHT` | `flagquantum.qec.codes._css_helpers._DISTANCE_SEARCH_WEIGHT` |
| `flagquantum.qec.codes` | `_STEANE_CHECK_SUPPORTS` | `flagquantum.qec.codes.steane._STEANE_CHECK_SUPPORTS` |
| `flagquantum.qec.codes` | `_STEANE_LOGICAL_SUPPORT` | `flagquantum.qec.codes.steane._STEANE_LOGICAL_SUPPORT` |

The cumulative ledger for the complete independent binding batch has 560 rows. Definition lists are empty for these
retirement-only rows: no original source definition identity is reassigned.
Full incoming-consumer review, frozen QEC regression, native installed behavior,
formal checkout regressions and this child's exact-head CI remain required.
Construction/IR and directly PR1084-dependent work keep their existing gates.

This QEC subset is included in the [complete binding closure](internal-private-binding-closure.md).
Its original narrow preview is retained independently; the expanded source,
artifact and formal regressions must qualify separately.
