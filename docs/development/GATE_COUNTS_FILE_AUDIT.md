# Gate counts protocol source-file audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Frozen source behavior | FlagQuantum owner |
| --- | --- |
| `BloqCount` and its tuple conversion | `selected_call_count.py` owns equality-based cutoffs, the three preset gate sets, weighted selected calls and explicit graph-leaf targets. T and T-adjoint remain distinct. |
| `GateCounts` | `QECGateCost` owns ordered nonzero categories and concrete/symbolic count algebra, sharing the existing fused accumulation. `gate_budget.py` owns T/CCZ, Toffoli-only, legacy and Beverland conversions. Concrete T-equivalent conversion is also available on `GateCost`. |
| `QECGatesCost` | `query_qec_gate_cost` owns typed leaves, native/legacy policies, recursive declarations, seed and validated static override precedence, and legacy annotations. |

Q-0799 covers all 14 unchanged original cases. The native tests compare the
full category rows, presets and a symbolic graph-leaf target, arithmetic and
ordered display, static rejection and seed precedence. Existing checks also
cover the 12-way fanout, binary AND compute and erase, measurements and
undeclared-call refusals. Model counts and the append-path primitive evidence
are independent.

Q-0798 also covers the executable typed Program cost query.
`query_program_qec_gate_cost` uses the generic Program cache policy for
call-only bodies and the existing QEC classifier for named terminal keys.
A computed composite bypasses seeded values and is omitted from the cache.
Validated static costs can still be stored. A program leaf requires an
explicit resource identity unless the caller supplies its cost. Legacy
annotations on resource identities use the existing declared-key query.

The five-control MultiAnd fixture verifies four modeled compute ANDs against
the frozen decomposition. Native binding emits 16 T/T-adjoint primitives
and the appended operation preserves the complex amplitudes and prefix
outputs. In the frozen source, mixed controls contribute four explicit
Cliffords to the abstract default MultiAnd model for two negative controls.
Its default decomposed AND leaves contribute zero Clifford, and its legacy
decomposed AND leaves count the four inversions. The native Program adapter
matches the decomposition profile.
