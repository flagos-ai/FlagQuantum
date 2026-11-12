# Classical simulator source disposition

Reference: unchanged Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, complete
`qualtran/simulation/classical_sim.py`. Existing numerical owners retain
Apache-2.0 attribution. This maps all definitions to their FlagQuantum owners;
it does not certify whole-file closure or arbitrary foreign graph execution.
The execution classification is `single_device_fast_path` numerical inspection.

## Definition and dependency ownership

| Source definition | Native owner and disposition |
| --- | --- |
| `ClassicalValT`, `ClassicalValRetT`, imports | Native typed metadata and numerical callback annotations replace foreign Soquet, Bloq, NetworkX and attrs objects. NumPy remains the numerical array owner; SymPy expressions remain symbolic values. No foreign graph enters production. |
| `_numpy_dtype_from_qlt_dtype` | Existing word/opaque storage codecs and `classical_typed_values` select width-dependent signed/unsigned NumPy storage or object arrays. Core executable metadata admission is a separate, narrower boundary. |
| `_empty_ndarray_from_reg`, `_get_in_vals` | `collect_typed_register_value` composes existing word/bit/field collectors. Scalar values preserve identity; arrays reconstruct declared shapes in row-major order, using field arrays when resolved. No second dtype dispatcher is introduced. |
| `ClassicalValDistribution` | Frozen `ClassicalDistribution` carries choice and probability arguments without sampling during construction. |
| `_ClassicalValHandler.get`, random/fixed handler constructors and `get`, banned `get` | `draw_classical_value` owns the same RNG-before-fixed-before-refusal policy. Native occurrence IDs replace `BloqInstance.i`; fixed values preserve identity and random draws preserve generator state/choice semantics. An abstract handler hierarchy adds no native responsibility. |
| `MeasurementPhase` | Frozen `MeasurementOutcomePhase` names a current output register and exact shape index. It is not a physical measurement instruction. |
| `ClassicalSimState.__init__`, `from_cbloq` | `ComponentClassicalState` validates stored Core records and a Compiler-supplied dependency order, initializes typed input assignments and records optional persistent occurrence IDs. The foreign graph factory is replaced by this explicit entry, not mirrored. |
| `ClassicalSimState._update_assign_from_vals` | State `_assign` uses `bind_typed_register_values` and existing distribution/domain/shape validators. Partial assignment prefixes are observable on failure; scalar symbolic expressions bypass numerical domain checks as in source. |
| `ClassicalSimState._recurse_impl`, `_recurse` | `simulate_pure_component_calls` composes stored function records with explicit actions. Pure recursion uses the existing separate default RNG policy. Explicit recursive NotImplementedError gains component context and preserves its cause. Foreign lazy decomposition exception classes stay at the adapter boundary. |
| `ClassicalSimState._update` | `commit_classical_outputs` assigns outputs first, then rejects every non-None phase, including numerical identity. It does not roll assignments back. |
| `ClassicalSimState.step`, nested `_in_vals` | State `step` and `_collect` consume SSA operands directly, dispatch values before phase, and recurse only when values are NotImplemented and phase is None. `evaluate_classical_action` owns return-type and phase/decomposition rejection. Native steps visit actual calls rather than foreign dangle nodes. |
| `ClassicalSimState.finalize`, nested `_f_vals`, `simulate` | State `finalize` collects explicit return values; `simulate` steps until StopIteration. Missing live values retain KeyError. Native empty-array returns differ from the source RightDangle lookup defect. |
| `PhasedClassicalSimState.__init__`, `from_cbloq` | The same state owner accepts explicit phased mode, initial phase and RNG/fixed policy. `assert_unit_classical_phase` owns the source tolerance check. No separate graph representation is created. |
| `PhasedClassicalSimState._recurse_impl` | `simulate_phased_component_calls` shares the numerical state, explicit stored call orders and handler policy; nested phase multiplies into the caller after output assignment. |
| `PhasedClassicalSimState._update` | `commit_phased_classical_outputs` and `apply_classical_phase` own assignment-before-phase application. State resolves measurement phases against this call's declared outputs before the generic mapping resolver. |
| `call_cbloq_classically` | `simulate_pure_component_calls` supplies the pure stored-record entry; inspectable state supplies assignments when needed. The source function's persistent default random handler is retained by the existing pure-entry policy. |
| `_assert_valid_phase` | `assert_unit_classical_phase` retains the source absolute modulus tolerance; it does not normalize phase. |
| `do_phased_classical_simulation` | `simulate_phased_component_calls` returns output mapping and accumulated phase, reusing the state owner rather than discovering foreign decomposition methods. |
| `get_classical_truth_table`, `format_classical_truth_table` | `classical_truth_table`, `register_classical_truth_table` and `format_classical_truth_table` enumerate ordered scalar domains, reject shaped inputs and preserve formatting/newline behavior. Supplied numerical actions remain explicit. |
| `add_ints` | `classical_addition.add_integer_values` owns arbitrary-precision addition and unsigned/signed modular wraparound. No circuit decomposition is inferred from numerical addition. |

Register shape traversal is row-major. Bit/word codecs retain MSB-first bit order.
SSA values and assignment storage belong to the numerical state; physical work
qubits and measurement slots belong to existing append emitters. Output lookup
does not allocate or release physical workspace.

## Current-output phase lookup correction

The source writes every output before `_get_soquet(..., right=True)` locates the
measurement result. An absent output name, out-of-range index or wrong index
rank raises ValueError at that point, retaining assignments and the incoming
phase. Native state previously exposed the generic mapping resolver's KeyError.
State now validates the current output key **after assignment** and raises
ValueError with a native port diagnostic. Generic `apply_classical_phase`
continues to expose its mapping contract unchanged.

Thirteen direct comparisons against the unchanged source cover scalar, vector
and matrix outputs, valid/missing/out-of-range/wrong-rank lookups and a second
call whose LEFT-only name matches a previous call's output. The latter must not
resolve a stale output. Every failure retains the assignment prefix, final
output and unchanged phase. Valid lookups apply exactly one minus sign.

## Verification and resource separation

Concentrated regression includes existing pure/phased recursive evaluation,
distribution/RNG policy, typed binding/collection, truth tables and tensor
consumers. Existing physical tests cover complete complex amplitudes, small
measurement histories, ordinary `circuit.append(op)`, operand/feedback order
and clean work. These consumers are reused; this lookup correction does not
change their emitted instruction streams.

New physical primitives, measurements, feedback slots and work bits: **zero**.
Quantum resource model for a numerical port lookup: **not applicable**. Existing
physical measured counts and source QEC models remain separately recorded in
[state execution evidence](COMPONENT_CLASSICAL_STATE.md); model categories are
not measured primitive counts.

## Remaining parent boundaries

Q-0882 and Q-0883 remain open. The subsequent
[original-test replacement matrix](CLASSICAL_SIM_TEST_MATRIX.md) accounts for
all 19 original test functions, with native numerical execution and explicit
representation differences. Notebook disposition and remaining foreign
lazy-decomposition failure paths still need acceptance evidence. Source-only
execution of the original tests is not native replacement. Numerical codecs can support types that
Core executable graph admission excludes; this document does not expand Core.

Input distributions use native sentinel ID -1; source fixed-input handling may
fail because LeftDangle has no instance index. Source empty returned arrays can
raise KeyError(RightDangle), whereas native explicit empty return metadata can
collect them. Source sentinel visits, object names and diagnostic text are not
native API promises. These differences remain visible rather than modifying
the reference or declaring the parent complete. G3 remains unstarted.

Local acceptance: **13 new cases**, **12,943 related tests in 146.82 s**
(997 numerical and other warnings), **84 CPU/TN tests in 9.70 s** (two optional
dependency/deprecation warnings), and **28 unchanged original tests in 1.44 s**
(one notebook deselected, two collection/marker warnings) pass. Strict mypy
covers both changed Python files; Black/Ruff, Simulation scope, architecture,
IR, protected API, Bell and measurement examples pass. State owner coverage is
97% (134 statements, 48 branches; refusal lines 91, 94 and 136 uncovered).
Three reference blobs match the frozen inventory. Implementation `c629405e`.
The prior default-suite timeout remains recorded; no full-suite green claim.
