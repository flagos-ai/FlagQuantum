# Bounded measurement-feedback lowering

This directory transforms verified `HybridProgram` operations into the existing
Core `CircuitIR` with measurement conditions and ordered parameter bindings.
Runtime executes feedback and Simulation evolves numerical state.

## Implementation ownership

| Module | Responsibility |
| --- | --- |
| `program.py` | Entry validation, pass evidence, SSA traversal, loop and measurement budgets, parameter binding and CircuitIR emission |
| `expressions.py` | Measurement references, predicate canonicalization, boolean comparisons and condition-partitioned classical values |

`program.py` owns the single lowering state. Expression functions receive that
state for its existing diagnostics and clause budget. Its type is imported only
under `TYPE_CHECKING`. They preserve clause ordering and expansion rejection.
Dependencies are the parent hybrid model, verified passes, shared runtime input
validation and diagnostics, Core IR and parameters, and the Python standard library. Backend
selection, job submission and numerical execution belong to their owning domains.

The package-level `lower_dynamic_program` and `LoweredDynamicProgram` entries
point directly to `program.py`. The existing private `dynamic_lowering.py` path
retains aliases to the same function and result type. Compiler owns this adapter,
scoped to those two names. Remove it after its remaining consumers migrate in an
explicit private-import cleanup targeting the next minor release, 0.5.0.

## Make and test a change

For predicate behavior, edit `expressions.py` and extend an accepted and rejected
scenario in `tests/hybrid_compiler/test_dynamic_session.py`. For traversal or
emission, edit `program.py` and check the same program with optimization enabled
and disabled. Preserve source-located errors, binding order, optimization records
and independent unroll, clause and measurement ceilings.

The shortest executable path captures an H gate, measurement and two-way feedback,
then checks both Runtime trajectory strategies and seeded reproducibility.
Run from the repository root.

```bash
python -m pytest tests/hybrid_compiler/test_dynamic_session.py -q
python -m pytest tests/hybrid_compiler/test_passes.py tests/hybrid_compiler/test_lowering_identity_cost.py -q
```

The feedback scenario emits `h`, `measure`, `x`, `x`, with the two X gates
conditioned on bit 0 equaling 1 and 0. Every sampled shot keeps wire 1 equal to
the measured bit and wire 2 equal to its complement.

The existing QEC entry exercises source capture, dynamic lowering and local
feedback execution through a user-facing workflow.

```python
import flagquantum.qec as qec

result = qec.run_repetition_memory_experiment(
    error_schedule=qec.ErrorSchedule((qec.ErrorEvent(1, 1),)),
    rounds=3,
    shots=8,
    seed=53,
    feedback_mode="compiled_lookup",
)
assert result.logical_error_rate == 0.0
assert all(shot.raw_final_data_bits == (0, 0, 0) for shot in result.shot_records)
assert all(shot.executed_feedback[1].qubit == 1 for shot in result.shot_records)
```

The injected error on wire 1 in round 1 is corrected in every shot. Round indices
are zero-based. This local CPU workflow requires no credentials or hardware.

For broader coverage, run the hybrid compiler and QEC consumers.

```bash
python -m pytest tests/hybrid_compiler tests/qec -q
python tools/architecture/check_architecture.py
```
