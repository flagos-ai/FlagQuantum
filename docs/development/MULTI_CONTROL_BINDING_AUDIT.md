# Complete multi-control X binding

## Source intake and ownership

The frozen Kickmix revision is `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`,
Copyright 2026 Google LLC, Apache-2.0. Qualtran stays at v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The complete `multi_control_x.pybind.cc`, `.h` and `multi_control_test.py` source
family is K-0234 through K-0236. Compiler owns typed operand admission and physical
control emission. Existing mixed primitive dispatch, reset/measurement cleanup,
physical lowering and classical scope owners retain their responsibilities.

`physical_multi_controlled_x(controls, target, *, workspace=())` receives an
ordered control sequence and a scalar or ordered quantum target word. Wire order
is physical, so no bit-endian reversal occurs at this boundary. Boolean false
controls skip the complete operation after both words convert. Classical controls
wrap the body in their original order, including duplicates. Zero/one quantum
controls broadcast X/CX. At least two quantum controls use the original linear
first-target fanout. For three or more controls, the caller supplies n-2 clean
work wires in allocation order. All logical ports stay caller-owned, and the
measured ladder clears its clean work.

The original helper releases these work wires in allocation order. Therefore a
second source call allocates their reverse order from the same LIFO pool. The
explicit native workspace interface exposes this order without introducing an
additional persistent allocator. Direct source/native comparison must retain
that reversal. Existing Qualtran multi-control decomposition retains its own
primitive schedule and resource models.

The complete source and direct dependency evidence is recorded below.

## Complete source disposition

Both complete C++ definitions are mapped. The internal quantum helper emits
X/CX/CCX directly for zero, one or two quantum controls and the measured ladder
for larger controls. The Python wrapper converts both words before skip, filters
literal/classical controls in source order, broadcasts the zero/one-control case,
and fans out around one first-target ladder for larger control counts. The header
maps to the same native entry. All 26 unchanged original Python tests execute
with the frozen extension and Simulator, including fifty-control consumers.

The fixture generator verifies the three original source blobs plus complete
`circuit_pattern.cc` and `.h` direct dependencies. That dependency reverses
LoopBuilder CCX control operands. Direct CircuitBuilder CCX and feedback CZ retain
their separate original order. No operand sorting or canonicalization is applied.
All 288 complete repeated-call configurations agree byte for byte, covering zero
through six, ten and fifty quantum controls, zero/one/two/four targets, plain,
mixed, duplicate classical and false-control words and exact/excess clean pools.
Each source call returns all leased qubits. The second native call explicitly
uses the source's reversed free-stack order. Both streams survive KMX round trips.

Independent native execution verifies every reset and HMR history for zero through
four quantum controls, one/two targets, both classical guard values and repeated
outer guard scopes. Arbitrary complete complex amplitudes include an entangled
spectator and every logical input. Active histories retain the exact expected
unnormalized amplitude and branch probability. All clean work returns to zero.
Inactive scopes preserve the full input and give zero weight to other histories.
Normal DynamicCircuit append separately flips two targets under four quantum and
one classical controls and verifies the full final state.

## Physical counts and models

A four-quantum-control, two-target call emits 3 CCX, 2 CX, 2 resets, 2 HMR and
2 feedback CZ. One classical control adds one PUSH_CONDITION/POP_CONDITION pair.
These are measured physical records. The two clean quantum leases return to zero,
and one scratch classical slot is reused by both HMR operations. The binding
source declares no analytical resource model. Existing logical multi-control
resource estimates remain owned and reported by their existing model modules.

Batch checks pass 220 concentrated cases, all 26 unchanged original tests,
29 integration cases, strict typing for all three Python files, Black, Ruff,
Compiler scope, architecture, IR, API baseline and whitespace checks.

## Failure and remaining boundaries

Controls require an explicit sequence. X-basis controls and nonquantum targets
reject before any skip. Empty targets and false controls require no clean work.
Otherwise insufficient or overlapping used work rejects before returning a native
artifact. Repeated targets in zero/one-control broadcasts retain repeated gate
records. Source helper K-0225, comparator K-0222's original crash, generic mixed
adder source defects and pending Proposals 116–119 remain independently open.
The stateless native API makes the required work order explicit at each call.

## Normal append consumer

```python
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.lowering.reversible_phase_execution import bind_reversible_phase_execution
from flagquantum.compiler.registers.control.physical_multi_control import physical_multi_controlled_x
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic

program = physical_multi_controlled_x((0, 1, 2), (3,), workspace=(4,))
circuit = DynamicCircuit(5).x(0).x(1).x(2)
circuit.append(FieldOperation(5, bind_reversible_phase_execution(program)))
result = run_dynamic(circuit, shots=1, seed=17, strategy="batched")
assert result.final_states.reshape(1, -1).abs().argmax().item() == 30
```

Regenerate with
`python tests/unit/fixtures/generate_multi_control_binding.py --source-root <frozen-kickmix-root> --output tests/unit/fixtures/multi_control_binding.json`.
Run `python -m pytest -q tests/unit/test_multi_control_binding.py`.
