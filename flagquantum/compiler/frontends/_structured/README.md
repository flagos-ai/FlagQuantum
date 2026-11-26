# Hybrid program compilation

This private Compiler package represents structured quantum-classical programs
with typed SSA values, control-flow regions, and ordered quantum effects.
Its design goal is to lower those programs into the shared Core circuit IR
without losing classical dependencies, parameter bindings, or quantum meaning.

Runtime owns measurement execution and feedback; Simulation owns numerical
evolution. This package does not introduce a second public IR or execution API.

## Follow a program

[Capture](frontend/README.md) → [verify](verifier.py) → [normalize](passes.py) →
[specialize](specialize.py) → [lower](lowering.py).
[Dynamic lowering](dynamic/README.md) handles measurement-dependent control.

For a small change, edit the owning stage and add an accepted and rejected
scenario under `tests/hybrid_compiler`. Run from the repository root:

```bash
python -m pytest tests/hybrid_compiler -q
python tools/architecture/check_architecture.py
```

Check optimized and unoptimized behavior, including gradients where supported.
Loop expansion must obey its budget; runtime tensor values must not become
part of reusable structure identity.

## Input validation and specialization ownership

| Module | Responsibility |
| --- | --- |
| [runtime_inputs.py](runtime_inputs.py) | Shared scalar, tensor, index and bool validation, shape/dtype/device descriptors and structural input signatures |
| [diagnostics.py](diagnostics.py) | The existing `SpecializationError` shared by input validation, static specialization and dynamic lowering |
| [specialize.py](specialize.py) | Bounded control-flow evaluation, loop expansion and selected quantum gate traces |
| [dynamic/program.py](dynamic/program.py) and [dynamic/expressions.py](dynamic/expressions.py) | Measurement-dependent control and Core CircuitIR emission; `dynamic_lowering.py` retains entry aliases |

Both compilation paths import input validation and diagnostics directly from
their owners. These shared modules depend only on the hybrid model and the
Python standard library. They inspect input metadata without executing quantum
kernels, selecting resources or hashing tensor contents.

`specialize.py` retains direct aliases for `SpecializationError`,
`_normalized_dtype`, `_runtime_shape`, `_runtime_device_class`,
`_input_descriptor` and `_input_signature`. The package-level error import
remains available. Compiler owns these private helper aliases; remove them after
consumer migration in an explicit private-import cleanup targeting the next
minor release, 0.5.0. The specialization entry and trace types remain implemented
at their existing path.

For an input validation change, edit `runtime_inputs.py` and extend an accepted
and rejected scenario in `tests/hybrid_compiler/test_runtime_inputs.py`. For a
static control-flow change, edit `specialize.py` and extend `test_specialize.py`.
The shortest test path compiles a parameterized rotation through both entries,
checks its wire and angle binding, preserves baseline signature hashes and
rejects incompatible input shapes and dtypes with optimization enabled and
disabled. Run from the repository root.

```bash
python -m pytest tests/hybrid_compiler/test_runtime_inputs.py tests/hybrid_compiler/test_specialize.py tests/hybrid_compiler/test_dynamic_session.py -q
```

For a user-facing workflow through source capture, dynamic lowering and local
feedback execution, run this CPU experiment.

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

Every shot corrects the injected error on wire 1 in zero-based round 1. The
experiment requires no credentials or remote resources.

[Supported constructs and lowering details](IMPLEMENTATION.md) describe the
bounded profiles, parameter flow, and migration constraints.

`unitary_effects.py` owns the closed inverse rules for checked register-unitary
effects. Native construction and the external source adapter share these rules;
source-specific diagnostics and storage binding checks remain at their capture
boundaries. No execution or new IR representation is introduced.

Python capture implementation ownership and its shortest executable workflow are
documented in [frontend/README.md](frontend/README.md). The existing
`capture.py` module retains the source entry points and diagnostic aliases.

Bounded measurement-feedback implementation ownership and its executable
feedback workflow are documented in [dynamic/README.md](dynamic/README.md).
The existing `dynamic_lowering.py` entry retains aliases to the implementation.
