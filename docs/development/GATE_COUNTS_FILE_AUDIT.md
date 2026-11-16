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

Q-0798 retains the composite-cache boundary in the QEC-specific query. The
generic typed Program cost query separately supports the source's policy of
bypassing composite seeds and omitting computed composite rows. Plain QEC
resource identities remain memoized as declared keys. The native model does
not use a source Bloq class as its identity. Legacy annotations are admitted
explicitly at the query boundary.
