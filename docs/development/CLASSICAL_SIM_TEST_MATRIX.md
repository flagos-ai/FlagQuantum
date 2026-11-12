# Classical simulator original-test replacement matrix

Reference: unchanged Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`,
`qualtran/simulation/classical_sim_test.py`. Its 19 top-level test functions
collect as 29 cases: 28 numerical cases and one notebook case. This matrix
accounts for every test and its supporting fixture; it does not expand Core
metadata admission or certify the notebook.

`test_classical_sim_source_replacement.py` executes 12 original test functions
(22 parameterized cases) with their assertion bodies unchanged and native
numerical entry points substituted at the test boundary. Six other numerical
functions have explicit parallel native checks because their diagnostics,
stepping sentinels or stored identities intentionally differ. The new file has
34 cases, including one additional complete complex-state check. Foreign fixture
objects and transient substitutions stay entirely in tests; frozen files are
never edited and production has no Qualtran import.

## Complete test mapping

All replacement names below are in `tests/unit/test_classical_sim_source_replacement.py`.
“Original body” means the unchanged upstream test function executes its own
assertions against native results. “Parallel” records actual differences rather
than changing the assertion to pretend foreign object identity exists natively.

| Original function | Native evidence and disposition |
| --- | --- |
| `test_dtype_validation` | Parallel: `test_original_validation_input_cases_through_native_binding`, all four original input dictionaries. Existing typed binders retain scalar objects, array element types and the assignment prefix before bad-bit/wrong-shape failure. Native register diagnostics replace foreign instance text. |
| `test_apply_classical` | Original body: `test_original_array_assertions_with_native_execution`, zero/one five-bit inputs, original output types, uint8 storage and XOR constant. Native record has an actual allocation/CX/X body. |
| `test_apply_phased_classical` | Original body: same replacement entry in phased mode, including the exact original exponential phase. This fixture's arbitrary numerical phase callback is not claimed as a newly compiled physical phase circuit. |
| `test_phased_classical_on_normal_classical` | Original body: same replacement entry without a phase callback, preserving phase one. |
| `test_normal_classical_on_phased` | Parallel: `test_pure_array_phase_rejection_retains_native_diagnostic_and_outputs`. Numerical identity phase is rejected after assignment. Native guidance names phased numerical evaluation rather than a foreign helper. |
| `test_cnot_assign_dict` | Parallel: `test_cnot_assignment_storage_has_explicit_native_identity_mapping`, all four inputs. Original six Soquet entries are preserved in the reference; native stores two input and two output SSA values. Each leaf output is matched through an explicit port mapping; duplicate dangle aliases are not invented. |
| `test_apply_classical_cbloq` | Parallel: `test_original_two_call_array_graph_and_step_assignments` reproduces the exact five-bit two-call graph, input intc storage, intermediate outputs, external renames and final uint8 arrays. |
| `test_step` | Parallel: the same test compares each call's complete output assignments and last position. The source first visits LeftDangle; a native step visits the first actual call. Finalize and exhausted iteration remain inspectable. |
| `test_add_ints_unsigned` | Original body: `test_original_exhaustive_add_assertions_with_native_arithmetic`, widths 1–4, all operand pairs. Existing `add_integer_values` supplies arithmetic. |
| `test_add_ints_signed` | Original body: same replacement, widths 2–4 and all signed operand pairs. |
| `test_notebook` | Subsequent [Q-0881 asset migration](CLASSICAL_SIMULATION_NOTEBOOK.md) executes all12 original and12 native code cells, disposes13 Markdown cells and renders the native value table. Reference blob9f99d7e8 remains unchanged; Q-0884 is the separate supertensor notebook. |
| `test_multidimensional_classical_sim_for_dtypes` | Original body: `test_original_multidimensional_assertions_with_native_numerical_codecs`, all six original 5-bit types and complete domains. Original action assertions still check element types. Native typed bind/collect owners supply numerical arrays; this is not wider executable graph admission. |
| `test_multidimensional_classical_sim_for_large_int` | Original body: same codec entry with exact QInt(100) values, including negative and greater-than-64-bit values, retaining object storage. |
| `test_multidimensional_classical_sim_for_gqf` | Original body: same codec entry with GF(4), retaining its field array subclass and element identity. Existing binary-field graph tests separately prove executable paths. |
| `test_classical_distribution` | Original body: `test_original_distribution_assertions_with_native_state`, all 100 draws through the existing measurement value action and state owner. Existing seeded tests separately compare full sequences and generator states. |
| `test_fixed_distribution` | Original body: same state entry translates the supplied fixed-handler map at the test boundary; all 100 original fixed results are checked. |
| `test_banned_distribution` | Parallel: `test_banned_distribution_rejects_before_output_assignment`. Both engines reject without assigning a result; native non-determinism diagnostics retain native wording. |
| `test_phased_classical_distribution` | Original body: same state replacement, all 52 original calls, RNG and fixed results, and outcome-dependent signs. Existing measured-branch tests check unnormalized complex amplitudes and clean work. |
| `test_derive_phasing_from_composed_bloq` | Original body: `test_original_composed_phasing_assertions_with_stored_native_bodies`, n=0/1/2/3. Existing `phase_fixture` and recursive evaluator supply stored allocation/X/Z bodies. Existing physical tests compare gates, append output and clean spectators. |

## Fixture ownership and execution boundaries

`ApplyClassicalTest` maps to the test's `array_record`: five THRU input bits and
five RIGHT output bits, row-major array order, MSB-first flattened bit order.
Each output is freshly zero-allocated, copied by CX and flipped for constant
`[1,0,1,0,1]`. The two-call version retains x, y and z independently.
`ApplyPhasedClassicalTest` adds a supplied numerical phase callback to the same
state owner. `TestMultiDimensionalReg` retains its original identity action and
type assertions; bind/collect replace only the numerical plumbing.
`ClassicalDistributionTest` and `ClassicalDistributionWithPhaseTest` reuse the
existing MeasureX value/phase action and actual typed measurement function.
`ComposedPhasing` reuses the previously accepted stored-body fixture. No new
production adapter, dtype dispatcher, abstract handler or component registry is
needed for these tests.

The new full-array test exhausts 32 logical inputs and reconstructs the complete
1024-by-32 isometry. A complex superposition executed with ordinary
`circuit.append(op)` matches it without phase normalization, and the unused clean
spectator remains zero. The five fresh target bits are declared **logical
outputs**, not scratch expected to return to zero. This unmeasured graph has no
measurement branches. Existing state/phased/pure tests separately retain all
small measurement histories, work erasure, full complex amplitudes and RNG
state comparisons; these are included in the concentrated regression.

## Separate resource accounts

| Workload | Measured primitive stream | Original quantum resource model | Ownership |
| --- | --- | --- | --- |
| Five-bit array copy/XOR realization | CX5, X3; measurement/feedback 0 | Unavailable: the original test-only action has no decomposition/call graph, and its QEC query raises NotImplementedError | Five caller input bits, five fresh logical outputs, one unused clean spectator |
| Numerical test entry and codecs | Zero emitted gates | Not applicable | NumPy/field arrays and SSA assignments; no physical work |

The physical array realization is a test fixture proving the numerical-to-append
workflow. Its eight gates are not presented as an upstream decomposed gate stream.
Existing Add/MeasureX/ComposedPhasing primitive comparisons and QEC models stay
in their original evidence and are rerun without changing those emitters.

## Remaining acceptance work

Q-0883 remains open: wider typed numerical cases do not establish executable IR
admission, and parallel tests retain explicit representation differences.
Q-0882 retains the source/native empty-array and input-sentinel differences and
foreign lazy-decomposition boundaries listed in
[the source disposition](CLASSICAL_SIM_SOURCE_DISPOSITION.md). An explicit
matrix removes missing accounting, not these actual boundaries. Q-0881 now has its complete separate notebook audit; Q-0885's accepted tensor notebook is not repeated.
No test count alone closes a parent, and G3 remains unstarted.

Local acceptance: **34 new cases**, **1,591 related tests in 32.77 s**
(18 numerical warnings), **84 CPU/TN tests in 5.42 s** (two optional dependency/
deprecation warnings), **28 unchanged source-only tests in 0.85 s** (one notebook
deselected, two collection/marker warnings), strict mypy of the new Python file,
Black/Ruff, scope/architecture/IR/API and Bell/measurement examples pass.
State coverage remains97% (134 statements,48 branches). A separate AST inventory
check confirms19/19 matrix rows; four frozen blobs match. Test commit `f6076c45`.
Default-suite timeout evidence is retained without repeating that run.
