# Dense entry source disposition

Reference: unchanged Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`,
`qualtran/simulation/tensor/_dense.py`. Numerical dependencies are pinned to
Quimb 1.15.0 and Cotengra 0.8.3. This is a single-device numerical path.

The complete source file contains four functions:

| Source function | Native owner and executable evidence | Remaining boundary |
| --- | --- | --- |
| `_order_incoming_outgoing_indices` | `register_axis_groups` and register binders preserve declared side, shaped word and bit order; original dense tests and axis audit cover pure/mixed groups. | Native typed port validation remains stricter than raw foreign objects. |
| `_group_outer_inds` | The same owner groups actual exposed descriptors. Consumers capture output groups before transformations, preserving D-created output hyperedges. | Foreign descriptor objects stop at the test adapter. |
| `quimb_to_dense` | Pure/mixed register binders reuse the numerical executor and explicit width policy; source tests run unchanged and with native numerical replacement at both default and explicit tolerance. | Native `None` retains arbitrary ambient gradients; explicit finite nonnegative tolerance opts into reported width > 12 and ADCR. Source default is 1e-12. |
| `bloq_to_dense` | Compiler owns root wrapping, full/partial expansion and visitation. Explicit providers feed existing node wiring and `component_dense` consumers. The fresh-label pure entry now composes factorized CNOT providers without manual wiring; Bell is the executable golden path. | Foreign flatten heuristics and exception cases remain Q-0889 work. Source logging and object discovery are replaced by native explicit composition. |

`contract_component_tensors_with_labels` is the numerical counterpart of the
existing four-argument node supplier. It reuses collision-free labels, SSA/port
validation and the same register consumer. It adds no graph, registry, physical
wire, public export, compiler dependency in Simulation, or executor. Supplier
failures propagate unchanged without retries.

The replacement test covers prepared states and unitaries, dense and factorized
CNOT providers, default/zero/positive tolerance and wide zero-work outputs.
It compares complete complex tensors and phase gradients against existing manual
wiring, verifies zero ancillary output, and observes the actual width-triggered
execution path. The five original source tests are additionally run through the
explicit policy; their function bodies and the reference checkout are unchanged.

Known source differences remain explicit: repeated-label A reduction, scalar
redistribution and source zero/NaN behavior, local tolerance without a global
error bound, discarded ambient gradients, and Cotengra's singleton raw-size
width report. See `TENSOR_WIDTH_POLICY.md` and the linked reduction audits.
Mutable foreign strategy overrides and arbitrary optimizer versions are not
certified. These differences and remaining flattening work keep Q-0887 open;
additional test counts do not close its parent under the closure rules.

Actual added quantum primitives, measurements, feedback slots and work bits:
**zero**. Dense output has exponential storage; the wide test holds 2,048 or
8,192 complex128 values (32,768 or 131,072 bytes), not a process memory bound.
Quantum gate-cost model: **not applicable**. Existing circuit primitive counts
and model estimates remain separate.

Next source audit: Q-0889 `_flattening.py`, especially foreign flatten predicates
and failure disposition. Width/ADCR dependencies are already implemented.

Local acceptance: **30 added scenarios**, **2,425 related tests in 128.72 s**,
**84 CPU/TN tests in 11.9 s**, strict mypy for four Python files, Black/Ruff,
Simulation scope, architecture, IR, protected API and the Bell example pass.
Three frozen source blobs are unchanged. Implementation `84e51b4c`. Historical
source warnings and default-suite timeout remain; this is not full-suite certification.

Coverage: the dense consumer has 24 statements at 100%; together with the
Bell example, 103 statements and 10 branches total 98%. The example main
guard is outside pytest coverage; the executable example passes separately.
