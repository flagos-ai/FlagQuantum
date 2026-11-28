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
It keeps a through QBit control and target storage, supports dirty targets,
and owns no workspace. Signed subtraction prefixes share this owner. Source QEC
charges one logical Clifford for the entire fanout, whereas the expanded
primitive count is `2 * width - 1` CNOTs. Run
`python -m pytest tests/unit/test_fanout_component.py -q` for source gate order,
complex dirty-target amplitudes, restoration, ownership and separate costs.

`sign_extension_component.py` consumes one signed word and returns its widened
view with an owned redundant sign prefix. Its inverse consumes the correct
extended image, clears/frees the prefix and returns the narrow word. Both
arguments describe the forward widths. Two's-complement and ones-complement
words reuse the same fanout owner. Fanout uses a QBit target at width one and an
opaque word at larger widths, following the frozen source signature. Run
`python -m pytest tests/unit/test_sign_extension_component.py -q` for both
encodings, source call order, complex states, spectator coherence and cleanup.

`signed_encoding_component.py` consumes LEFT one's-complement `x` and returns
RIGHT two's-complement `y` over the same MSB-first word. Its inverse exchanges
the two types. It shares the existing increment sequence with
`signed_encoding.py`, calls the shared And owner and erases/frees each carry.
Bind with `width - 2` clean wires, then append the bound operation as above.
Run `python -m pytest tests/unit/test_signed_encoding_component.py tests/unit/test_signed_encoding.py -q`
for exact native streams, source And leaves, complete complex amplitudes,
measurement histories and clean work. The frozen conversion root supplies a
signature and Toffoli model but no executable decomposition. See
[the complete source audit](../../../../docs/development/SIGNED_CONVERSION_FILE_AUDIT.md).

`weighted_columns_component.py` stores weighted polynomial reduction over
MSB-first bit arrays. `bit_compression.py` owns the shared product/compression
sequence. The typed graph allocates its work block once, retains each shared And
compute/erase call and frees the restored block. The work block keeps physical
operand bindings stable across independent product scheduling. Arithmetic square
and sorting index compose this owner. Numerical execution and domain-specific
term selection remain in their respective owners. Run
`python -m pytest tests/unit/test_weighted_columns_component.py -q` for original
And leaves, feedback, per-wire dependency order, resource and lifetime checks.

`column_steps_component.py` owns only typed column CX/And/erase leaves and SSA
updates. Weighted-column XOR and live product accumulation both use it. Consumers
retain their own term layout, allocation lifetimes and arithmetic action. The
pre-extraction XOR Program and complete binding are checked for exact equality.

`hamming_weight_component.py` stores the population-count tree through the
existing Hamming traversal, compression sequence, CX and And owners. It returns
source, opaque junk and unsigned count in that order. The inverse consumes the
computed junk/count and restores source. Width one retains an empty junk word.
Run `python -m examples.algorithms.register_hamming_weight` for coherent count
copying and complete work erasure through `circuit.append(PopulationCount())`.
Run `python -m pytest tests/unit/test_hamming_weight_component.py -q` to check
stored calls, gate dependencies, branch amplitudes, ownership and source models.

`on_each_declaration.py` owns repeated-child constructor metadata, the default
opaque target signature and direct child multiplicity. Its finite body delegates
to `on_each_word_program`, preserving MSB-first Split/call/Join and caller-owned
storage. Symbolic signatures and models remain available without physical binding.
Source-admitted shaped or directional children retain metadata but fail finite
construction. Run `python -m pytest tests/unit/test_on_each_foreign_file.py -q`
for actual foreign-library resolution, typed SDK consumers and original failures.

`bitwise_component_control.py` also propagates one positive control through
two-word XOR: each CNOT becomes the existing typed Toffoli, while Split/Join
retain the control unchanged. No work or measurement is introduced. Controlled
equality reuses this owner; its source primitive checks are in
`tests/unit/test_controlled_equality_component.py`.

`nested_controlled_xor_constant_program` preserves two separate positive scalar
controls (`ctrl2`, `ctrl`) on an MSB-first encoded pattern. Each source-loaded
CNOT becomes one atomic Toffoli; no shared predicate flag or measurement is
introduced. This differs from grouped-control synthesis even when the truth
table agrees. Run `tests/unit/test_nested_controlled_bitwise_component.py` for
complete source call/operand, gate and full complex-operator comparisons.
