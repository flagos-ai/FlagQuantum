# Bit operations

This package owns xOR, bitwise transforms, population count, signed encodings and register rotations.

Core, shared Compiler facilities and the register components used by these
implementations are allowed dependencies. Backend selection and numerical
execution belong to Runtime and Simulation. Public APIs and IR schemas follow
their existing change-proposal process.

Start with [bitwise.py](bitwise.py). Preserve register order, feedback slots and caller-owned
workspace. From the repository root, run the focused scenario tests.

```bash
python -m pytest tests/unit/test_controlled_bitwise.py -q
```

The [register package guide](../README.md) provides executable consumer examples,
source-parity requirements, resource-counting boundaries and the relocation map.

`bitwise_width.py` reuses the XOR/constant-XOR emitters and the existing OnEach
boundary to bind arithmetic resource identities to automatic connection widths.
All words remain caller-owned and no workspace is allocated. Run
`python -m pytest tests/unit/test_bitwise_resource_width.py -q` for frozen-source
cache/order, symbolic fallback, signed range and source-width DOT comparisons.
[BITWISE_RESOURCE_WIDTH.md](../../../../docs/development/BITWISE_RESOURCE_WIDTH.md)
records native identity normalization and execution/resource evidence.

`bitwise_component.py` stores XOR, encoded constant XOR and complement in the
existing typed Program. It reuses the emitters and `on_each_word_program`.
The dtype remains attached to each through word, and binding leaves every word
with its caller. Constants are unsigned encoded bit patterns. For example,
the three-bit two's-complement encoding of -1 is pattern 7.

```python
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.bits.bitwise_component import bitwise_word_program
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.registers import RegisterType

program = bitwise_word_program(RegisterType("signed", 3), "xor_constant", pattern=7)
bound = bind_register_program(program, ((2, 0, 1),))
operation = FieldOperation(3, bound.circuit.instructions)
circuit = Circuit(3)
circuit.append(operation)
assert circuit.state().flatten().abs().argmax().item() == 7
```

Run `python -m pytest tests/unit/test_bitwise_component.py -q` for stored call
order, native binding, frozen-source gates, source tensors and caller ownership.

`controlled_constant_width.py` reuses controlled XorK emission and the shared
single-zero-control lifetime wrapper. Both directions and polarities keep the
caller control/target live without workspace. Run
`python -m pytest tests/unit/test_controlled_constant_resource_width.py tests/unit/test_controlled_bitwise.py -q`.
[The interface and evidence](../../../../docs/development/CONTROLLED_CONSTANT_RESOURCE_WIDTH.md)
record source self-adjoint child identities and symbolic fallback limits.

`controlled_xor_width.py` binds the existing controlled register-XOR emitter's
atomic Toffolis to through-wire connections. Zero control reuses the shared
control-flip wrapper with both words live. See
[the interface and evidence](../../../../docs/development/CONTROLLED_XOR_RESOURCE_WIDTH.md).

`hamming_weight_width.py` binds count-tree logical junk/output creation and
adjoint consumption. The gate emitter and width graph share the tree and six-step
compression sequence. See [the interface and evidence](../../../../docs/development/HAMMING_WEIGHT_RESOURCE_WIDTH.md).
Run `python -m pytest tests/unit/test_hamming_weight_resource_width.py tests/unit/test_hamming_weight.py -q`.

`bitwise_component_control.py` owns control propagation through actual XorK and
BitwiseNot child calls. It specializes X to CNOT and threads controls unchanged
through Split/Join. Conditional arithmetic and controlled constant loading share
this owner. Arithmetic constants arrive as already encoded word patterns.

`fanout_component.py` stores the existing `fanout_ir` tree as typed CNOT calls.
It keeps a through QBit control and opaque target word, supports dirty targets,
and owns no workspace. Signed subtraction prefixes share this owner. Source QEC
charges one logical Clifford for the entire fanout, whereas the expanded
primitive count is `2 * width - 1` CNOTs. Run
`python -m pytest tests/unit/test_fanout_component.py -q` for source gate order,
complex dirty-target amplitudes, restoration, ownership and separate costs.
