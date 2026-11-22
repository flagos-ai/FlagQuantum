# Complete classical broadcast store binding

The reference is Kickmix revision
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`, Copyright 2026 Google LLC,
Apache-2.0. Qualtran stays locked to v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`.

## Complete source and dependency ownership

| Parent | Complete source file | Native disposition |
| --- | --- | --- |
| K-0187 | `py/circuit/ops/bit_store.cc` | All three overloads map to the same ordered classical store artifact emitter. Homogeneous bit, Boolean scalar/word and mixed classical control dispatch retain their exact physical records. |
| K-0188 | `py/circuit/ops/bit_store.pybind.cc` | Value conversion precedes existing width and operand binding. Targets require classical bits. Controls admit classical bits or Boolean literals. |
| K-0189 | `py/circuit/ops/bit_store.pybind.h` | The declared target/value/control ports map to `registers/primitives/classical_store.py::physical_classical_store`. |
| K-0190 | `py/circuit/ops/bit_store_test.py` | All five original tests run unchanged. Native complete record, reduction, alias and feedback consumers supply independent execution evidence. |

`registers/primitives/broadcast_binding.py` and
`registers/layout/operand_word.py` retain width resolution and typed conversion.
`artifacts/reversible_phase.py` already owns the exact store opcodes and validated
classical storage. `lowering/reversible_phase_execution.py` translates those
records to existing Core ClassicalXor instructions. Runtime evaluates their
conditions and assignments. No new IR vocabulary, result type, backend,
external package or public export is introduced.

## Interface and ownership

The caller supplies target storage and optional control storage through existing
classical coefficient or OperandIdentity values. A scalar broadcasts over the
first explicit word. Explicit word widths must agree, including zero. The
selected lane order is preserved, so strided/reversed physical IDs and repeated
targets remain significant. Conversion validates the complete bound words before
returning an artifact.

Value conversion follows the source scalar Boolean protocol. None becomes
false. Boolean, integer, floating and objects with the type-level Boolean
protocol use their scalar truth value. Length-only containers, strings and
index-only objects are rejected. A failed Boolean conversion raises native
ValueError with its original cause. The source pybind boundary reports its
conversion failures as RuntimeError. This domain exception adaptation is
explicit and retains the cause. Both validate the value before target/control
conversion or a disabled operation.

Every target and control bit remains caller-owned and is declared as an input.
Sparse IDs retain their original positions and determine classical storage width.
A disabled store preserves the target's old value. Duplicate target/control
aliases read the state immediately before their own lane. A preceding write
can change the enable of a later write. No whole-broadcast guard snapshot is
introduced. This preserves source scalar-target OR reduction and cascaded
self-alias behavior.

The shared artifact requires at least one quantum wire. Its one-wire minimum
is an untouched spectator for native append. The source store uses zero quantum
operands, and the native store emits zero quantum instructions. That artifact
width is separate from allocated algorithm work or quantum gate resources.

## Source and execution evidence

All four files are Git-blob verified by the reproducible fixture generator.
The actual frozen extension exposes version 0.1 and Simulator. It passes all
five original test cases, including the sixteen-input OR reduction and atomic
quantum-control rejection.

The complete binding fixture records 587 configurations. It includes 576 target/
control broadcast combinations with both stored values and eleven additional
scalar conversion witnesses. Physical BIT_STORE0/BIT_STORE1 records retain
ordered result IDs and inline classical guards byte for byte. Native KMX
encoding and decoding preserve the whole artifact. Invalid target/control,
word-width and scalar conversion cases reject before returning any instructions.

Independent native execution exhausts all sixteen initial four-bit values for
both stored literals and four alias/reduction/mixed layouts. Every output bit
agrees with a sequential classical reference. The complete complex state of two
entangled arbitrary spectator qubits remains exactly unchanged. The scalar
reduction independently checks all eight control words. Sparse inputs, empty
broadcasts, inactive old values, explicit Boolean protocol and conversion-error
causes are retained. No source or native store measures a qubit, so this source
family has no measurement branches or quantum work to erase.

## Physical resources and analytical models

| Bound lane | Emitted record count |
| --- | ---: |
| Enabled literal control | One BIT_STORE0 or BIT_STORE1 |
| Disabled literal control | Zero |
| Classical control | One BIT_STORE0 or BIT_STORE1 with the original inline guard |
| Scalar target with three classical control bits | Three ordered conditional stores |

The stores emit zero quantum gates, resets, HMR operations or workspace leases.
A scalar-target three-bit reduction uses four classical input slots. Its
physical stream contains three store records. Execution expands those same
records to three ClassicalXor assignments through the existing owner. Generic
re-lowering of logical classical assignments has its own clearing/inversion
schedule and is kept separate from the measured source physical count.
The binding files declare no analytical resource formula. Existing Program
word-write models and native target resource estimators retain their own owners.

## Normal append consumer

```python
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.lowering.reversible_phase_execution import bind_reversible_phase_execution
from flagquantum.compiler.registers.finite_field.coefficients import ClassicalFieldCoefficient as Bit
from flagquantum.compiler.registers.primitives.classical_store import physical_classical_store
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic

program = physical_classical_store(Bit(0), True, control=(Bit(1), Bit(2)))
setup = tuple(ClassicalXor((), slot, value).instruction()
              for slot, value in enumerate((False, True, False)))
circuit = DynamicCircuit(1)
circuit.append(FieldOperation(1, (*setup, *bind_reversible_phase_execution(program))))
result = run_dynamic(circuit, shots=1, seed=41, strategy="batched")
assert result.classical_bits[0, :3].tolist() == [1, 1, 0]
```

Run `python -m pytest -q tests/unit/test_classical_store_binding.py`.
Regenerate with
`python tests/unit/fixtures/generate_classical_store_binding.py --source-root <frozen-kickmix-root> --output tests/unit/fixtures/classical_store_binding.json`.
The generator verifies all four source blobs and actual extension availability.
Previously accepted operand/broadcast dependencies are reused. Broader generic
op-testing helpers, the K-0222 mixed/empty-pool source crash, other original
source failures and pending Proposals 116–119 remain under their own parents.
