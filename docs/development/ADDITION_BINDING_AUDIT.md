# Complete in-place addition binding

## Source and interface intake

The frozen reference is Kickmix `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`,
Copyright 2026 Google LLC, Apache-2.0. Qualtran remains locked to v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`.

The complete `py/circuit/subs/iadd.pybind.cc`, `.h` and `iadd_test.py` group is
K-0229 through K-0231. Compiler owns the binding. Existing operand-word conversion,
physical in-place quantum addition, physical mixed addition and classical scope
owners retain their responsibilities. Runtime and Core retain execution ownership.

`physical_addition_binding(offset, *, target, control=True, workspace=(),
btol=inf, subtract=False)` receives MSB-first native words. Workspace is caller-
owned clean storage in source allocation order. The binding lends at most the
target width, returns all leases, preserves source call marks and measurement
feedback, and leaves target/offset/control ownership with the caller. Integer
literals admit the source signed lower bound and wrap once. Empty targets return
before converting offset or enable. A disabled enable skips generator validation.
Classical enables surround the complete physical call. Mixed quantum/classical
words follow source rejection instead of silently choosing a different schedule.

Primitive counts come from emitted records. Existing generator resource models
remain separate. Existing mixed extra-clean-prefix arithmetic defects remain
under the original source parents and require explicit disposition here.

The complete binding source group and direct conversion/control/generator dependencies are verified below.

## Complete source disposition

| Parent | Frozen Git blob | Native owner |
| --- | --- | --- |
| K-0229 implementation | `919978108730d711dcd2465e29678fe238c0d3a6` | `registers/arithmetic/addition_binding.py` maps the shared dispatch and both wrappers. |
| K-0230 declarations | `42b6c225590a5ef0deb11fd40b85d39c92ce6e39` | The same binding selects subtraction explicitly. Offset remains positional, target keyword-only. |
| K-0231 original tests | `fc5a2f6731b3f5d81167725d578a03d45c0c99b0` | All 28 unchanged source tests execute with the actual frozen Simulator. Independent native record and execution consumers verify the mapped behavior. |

The reproducible generator verifies all three immutable source blobs and loads
the actual Kickmix 0.1 extension with Simulator before generating evidence.
All 6,070 accepted configurations match the complete ordered source records,
including call marks, reset locations, measurement results, inline feedback and
outer classical guards. Each source case executes two consecutive append calls
and checks that all clean workspace has returned to the pool. Another 1,578
source and native rejections agree. Widths zero through six, empty/short/equal/
long offset words, quantum/classical/literal/integer offsets, both operations,
all four enable kinds, zero/full/excess workspace and finite/positive-infinite/
negative-infinite/NaN tolerance dispatch are covered. KMX decoding and re-encoding
preserve each complete stream.

NaN is significant at this binding. Original floating-point comparisons select
the full exact classical schedule. The binding passes infinity to the existing
mixed generator to select that same schedule, while retaining the generic
mixed generator's own ordered-tolerance validation. Quantum offsets accept
nonfinite tolerances and reject finite ones, as in the source. False controls
skip this generator-specific validation after operand conversion. Empty targets
skip offset and control conversion altogether.

Input coefficient and enable slots are reserved before generator measurements.
The mixed exporter retains its existing contiguous classical input prefix,
including unused holes. Quantum output/control and selected clean work all
remain caller-owned. Excess clean entries are unused. Classical enables are
read-only across the body, so the shared physical scope owner wraps the whole
call without a new guard or result representation.

## Coherent execution and measured resources

Independent native execution checks every reset and HMR branch for two-bit
addition/subtraction, classical or quantum offsets, classical or quantum enables
and both classical enable values. A three-bit classical offset exhausts all eight
word values for both operations. These checks use arbitrary complex amplitudes
across all logical inputs, including an entangled spectator. They compare full
unnormalized branch amplitudes and probability, with clean work returning to
zero in every branch. Inactive classical guards retain the full state and give
weight only to the zero postselection history. Normal DynamicCircuit append
independently computes zero minus one modulo four under a classical guard.

The following actual two-bit streams use a quantum enable and two clean leases.
Marks are separate from operation counts.

| Offset | Actual primitive records |
| --- | --- |
| Quantum word | 4 CCX, 1 reset, 1 HMR, 1 feedback CZ |
| Classical word | 4 X, 4 CX, 1 CCX, 1 reset, 1 HMR, 1 feedback CZ |
| Integer minus one | 4 X, 2 CX, 1 CCX, 1 reset, 1 HMR, 1 feedback CZ |

These binding files declare no analytical resource formula. Existing arithmetic
model owners retain their separate estimates. No model estimate substitutes for
the measured source/native record counts.

Batch verification passes 228 concentrated tests, all 28 original tests,
29 integration cases, strict typing for three changed Python files, Black,
Ruff, Compiler scope, architecture, IR, API baseline and whitespace checks.

## Retained boundaries

K-0056.04's existing five-bit extra-clean-prefix defect uses a mixed quantum/
classical offset and quantum carry. This binding requires either an entirely
quantum or entirely classical offset and fixes carry/borrow to false. The mixed
word witness rejects in this binding before reaching that generator path.
The generic generator defect remains open under its existing parent. The
original records and implementations remain unchanged. General fuzz helper
K-0225, comparator K-0222's original crash and pending Proposals 116–119 retain
their independent scopes. This batch declares no whole-suite or G2 completion.

## Normal append consumer

```python
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.lowering.reversible_phase_execution import bind_reversible_phase_execution
from flagquantum.compiler.registers.arithmetic.addition_binding import physical_addition_binding
from flagquantum.compiler.registers.finite_field.coefficients import ClassicalFieldCoefficient as Bit
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic

program, calls = physical_addition_binding(
    -1, target=(1, 0), control=Bit(3), workspace=(3, 2)
)
circuit = DynamicCircuit(program.n_qubits)
circuit.append(FieldOperation(program.n_qubits, (
    ClassicalXor((), 3, True).instruction(),
    *bind_reversible_phase_execution(program),
)))
result = run_dynamic(circuit, shots=1, seed=19, strategy="batched")
assert result.final_states.reshape(1, -1).abs().argmax().item() == 12
```

Regenerate with
`python tests/unit/fixtures/generate_addition_binding.py --source-root <frozen-kickmix-root> --output tests/unit/fixtures/addition_binding.json`.
Run `python -m pytest -q tests/unit/test_addition_binding.py`.
