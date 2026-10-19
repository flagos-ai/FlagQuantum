# FlagQuantum compiler and IR implementation map

Reviewed against main commit `0f389c9d8f30179557eb14920bbc9e9dd49b1bd0`
on 2026-10-19 for the M0 baseline.

The current compiler operates on Core-owned `CircuitIR`. The old private
`flagquantum._compiler` package and its phase-based implementation inventory
are not the current source layout. Historical multi-level IR designs remain
useful rationale, but their phase completion labels do not describe this tree.

## Circuit compilation and execution

| Responsibility | Current implementation |
| --- | --- |
| Public circuit interchange | `flagquantum/core/ir/__init__.py` |
| Root compile dispatch | `flagquantum/_api.py` |
| Optimization and circuit compilation | `flagquantum/compiler/pipeline.py` |
| Named circuit passes, analyses, preservation, and invalidation | `compiler/pass_manager.py`, `compiler/optimization_levels.py` |
| Native-gate and topology legalization | `compiler/native_gate_legalization.py`, `compiler/topology_legalization.py` |
| Target requirements and scheduling | `compiler/target_legalization.py`, `compiler/schedule_legalization.py` |
| Target emission and independent conformance | `compiler/target_emission.py`, `compiler/target_conformance.py` |
| Plan construction and execution | `runtime/planner/`, `runtime/execution.py`, `runtime/plan_execution.py` |

Paths without the package prefix are relative to `flagquantum/`.
`fq.compile` exists and dispatches to the native circuit compiler or a requested
compiler extension. Native target emission produces audited text; it does not
imply a new public executable-artifact API or hardware certification.
`fq.run(program)` composes planning and execution. `fq.run(plan)` executes the
accepted plan without compiling or planning it again.

## Internal Core program boundary

`flagquantum/core/ir/levels.py` owns the internal level vocabulary and conversion
entry points. The realized exit is `program`, implemented under
`core/ir/program/`; the declared `quantum` and `target` exits return explicit
unsupported diagnostics. The public `CircuitIR` remains schema 1.0.

This boundary already provides typed values, verification, and exact bounded
circuit/program conversion. Reuse it and the existing Compiler pass manager
when extending common program semantics. Their presence does not establish
mixed-dimensional state spaces, physical controls, pulse scheduling, or a
complete multi-level compiler. Do not create a parallel IR hierarchy or pass
registry for a new physical route.

## Private structured programs

`flagquantum/compiler/_hybrid/` implements capture, typed SSA values, verification,
normalization, specialization, and static/dynamic lowering for bounded
quantum-classical profiles. It is absent from the root public exports and does
not replace public `CircuitIR`. Read its
[implementation details](../../flagquantum/compiler/_hybrid/IMPLEMENTATION.md)
for supported constructs, parameter flow, and lowering restrictions.

The existence of this private path must not be summarized as either “ProgramIR
not started” or completion of every proposed multi-level IR phase. Timing,
advanced lowering, and provider behavior require their own implementation and
evidence; a design document alone establishes none of them.

## Verification and further reading

- `tests/integration/test_cpu_vertical_slice.py`: public local plan/run behavior.
- `tests/hybrid_compiler/`: structured-program semantics, lowering, gradients,
  target legalization, emission, and conformance scenarios.
- [Compiler ownership](../../flagquantum/compiler/IMPLEMENTATION.md).
- [Runtime ownership](../../flagquantum/runtime/IMPLEMENTATION.md).
- [Capability catalog](../generated/CAPABILITIES.md) and
  [known limitations](../reference/KNOWN_LIMITATIONS.md).
- [Historical multi-level IR design](../architecture/MULTI_LEVEL_IR_ARCHITECTURE.md)
  and [Phase 0–1 plan](../architecture/MULTI_LEVEL_IR_PHASE_0_1_EXECUTION_PLAN.md).

Update this map when implementation ownership or execution paths change. Keep
historical approvals in their original context rather than treating them as
current code contracts.
