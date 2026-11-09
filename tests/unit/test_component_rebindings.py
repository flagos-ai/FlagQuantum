"""Replace original split/join builder witnesses with real Program functions."""

from dataclasses import replace

import numpy as np
import pytest

from flagquantum.compiler.registers.component_bindings import RegisterBindingError
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.register_programs import lower_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_register_component_calls import port

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


class NativeRebindings:
    """Normalize observations from existing Program operations, without a graph clone."""

    error_type = RegisterBindingError

    def __init__(self, signature):
        self.signature = signature
        self.builder = ComponentProgramBuilder(signature)
        self.inputs = self.builder.inputs

    def identity(self, binding):
        value = binding.value
        producer = (
            "entry"
            if value in self.builder._arguments
            else next(
                index
                for index, op in enumerate(self.builder._operations)
                if value in op.results
            )
        )
        return producer, binding.register.name, binding.index

    def identities(self, collection):
        return tuple(self.identity(binding) for binding in np.asarray(collection).flat)

    @property
    def available(self):
        return {self.identity(binding) for binding in self.builder._available}

    @property
    def instances(self):
        return tuple(range(len(self.builder._operations)))

    def _known_bindings(self):
        # Recover port metadata from definitions and results, using the original
        # register shapes. The adapter never assigns or stores value identities.
        from flagquantum.compiler.registers.component_bindings import (
            create_register_bindings,
        )

        known = {
            binding.value: binding
            for value in self.inputs.values()
            for binding in np.asarray(value).flat
        }
        for operation in self.builder._operations:
            function = self.builder._functions[operation.callee]
            values = iter(operation.results)
            for register in Signature(function.registers).rights():
                if not operation.results:
                    break
                collection = create_register_bindings(values.__next__, register)
                known.update(
                    (binding.value, binding) for binding in np.asarray(collection).flat
                )
        return known

    @property
    def connections(self):
        known = self._known_bindings()
        connections = []
        for instance, operation in enumerate(self.builder._operations):
            function = self.builder._functions[operation.callee]
            targets = [
                (register.name, index)
                for register in Signature(function.registers).lefts()
                for index in register.all_idxs()
            ]
            connections.extend(
                (self.identity(known[value]), (instance, name, index))
                for value, (name, index) in zip(
                    operation.operands, targets, strict=False
                )
            )
        targets = [
            (register.name, index)
            for register in self.signature.rights()
            for index in register.all_idxs()
        ]
        connections.extend(
            (self.identity(known[value]), ("return", name, index))
            for value, (name, index) in zip(
                self.builder._returned, targets, strict=False
            )
        )
        return tuple(connections)

    def split(self, binding):
        return self.builder.split(binding)

    def join(self, bindings, dtype):
        return self.builder.join(bindings, dtype)

    def finalize(self, bindings):
        self.program = self.builder.finalize(bindings)
        self._final_bindings = bindings
        return self

    def final_bindings(self):
        # Source final_soqs omits declared empty-shaped outputs with no dangle.
        return {
            name: value
            for name, value in self._final_bindings.items()
            if np.asarray(value).size
        }


@pytest.mark.parametrize("width", [0, 1, 3])
@pytest.mark.parametrize("shape", [(), (2,), (2, 2), (0, 2)])
def test_original_owned_word_scenarios_accept_native_builder(width, shape):
    from tests.unit.test_register_rebinding_contract import (
        test_every_word_has_ownership_even_when_it_has_no_physical_bits,
    )

    test_every_word_has_ownership_even_when_it_has_no_physical_bits(
        NativeRebindings, width, shape
    )
    register = port("word", width, shape)
    native = NativeRebindings(Signature([register]))
    words = np.asarray(native.inputs["word"], dtype=object)
    returned = np.empty(shape, dtype=object)
    for index in register.all_idxs():
        returned[index] = native.join(native.split(words[index]), register.dtype)
    module = native.finalize({"word": returned[()] if not shape else returned}).program
    assert verify_program_record(module) == ()
    block = lower_register_program(module).functions[0].blocks[0]
    assert not block.operations
    assert block.arguments == block.return_values


@pytest.mark.parametrize("failure", ["duplicate", "short", "rank"])
def test_original_failed_join_scenarios_accept_native_builder(failure):
    from tests.unit.test_register_rebinding_contract import (
        test_failed_join_retains_exact_consumption_prefix,
    )

    test_failed_join_retains_exact_consumption_prefix(NativeRebindings, failure)


@pytest.mark.parametrize("encoding", ["opaque", "uint"])
@pytest.mark.parametrize("permutation", [(0, 1, 2), (2, 0, 1)])
def test_native_rebindings_preserve_source_connections_and_zero_primitive_cost(
    encoding, permutation
):
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from tests.unit.test_register_rebinding_contract import SourceRebindings

    signature = Signature([Register("word", RegisterType(encoding, 3))])
    source, native = SourceRebindings(signature), NativeRebindings(signature)
    for builder in (source, native):
        bits = builder.split(builder.inputs["word"])
        result = builder.join(bits[list(permutation)], signature.get_left("word").dtype)
        builder.finalize({"word": result})
    assert native.connections == source.connections
    assert native.instances == source.instances == (0, 1)
    assert get_cost_value(source.program, QECGatesCost()).asdict() == {}
    block = lower_register_program(native.program).functions[0].blocks[0]
    assert block.operations == ()
    assert block.return_values == tuple(block.arguments[index] for index in permutation)


@pytest.mark.parametrize("permutation", [(0, 1, 2), (2, 0, 1)])
def test_split_gate_join_matches_source_complex_tensor_with_spectator(permutation):
    import torch
    from qualtran import BloqBuilder, QAny
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    import flagquantum as fq
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.core.ir import CircuitIR, Instruction
    from tests.unit.test_register_program_lowering import packed_circuit

    upstream, inputs = BloqBuilder.from_signature(
        SourceSignature([SourceRegister("word", QAny(3))])
    )
    bits = upstream.split(inputs["word"])
    bits[1] = upstream.add(XGate(), q=bits[1])
    source = upstream.finalize(
        word=upstream.join(bits[list(permutation)], dtype=QAny(3))
    )
    tensor = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        8, 8
    )
    assert get_cost_value(source, QECGatesCost()).asdict() == {"clifford": 1}

    leaf = packed_circuit(CircuitIR(1, (Instruction("x", (0,)),)), (1,))
    functions = tuple(f for f in leaf.functions if f.name != "main")
    signature = Signature([port()])
    builder = ComponentProgramBuilder(signature, functions)
    bits = builder.split(builder.inputs["word"])
    bits[1] = builder.call("packed_leaf", {"word0": bits[1]})
    result = builder.join(bits[list(permutation)], signature.get_left("word").dtype)
    module = builder.finalize({"word": result})
    body = lower_register_program(module).functions[0].blocks[0]
    assert [op.opcode for op in body.operations] == ["x"]
    physical = (3, 1, 0)
    locations = {value: index for index, value in enumerate(body.arguments)}
    instructions = []
    for operation in body.operations:
        wires = tuple(locations[value] for value in operation.operands)
        instructions.append(
            Instruction(operation.opcode, tuple(physical[wire] for wire in wires))
        )
        locations.update(
            (result, wire)
            for result, wire in zip(operation.results, wires, strict=True)
        )
    output = tuple(physical[locations[value]] for value in body.return_values)
    rows = torch.arange(1, 17, dtype=torch.float64)
    state = torch.complex(rows, rows.remainder(7) - 3)
    state /= state.norm()
    circuit = fq.Circuit(4, inputs=state, dtype=state.dtype)
    circuit.append(FieldOperation(4, tuple(instructions)))
    actual = circuit.state().reshape((2,) * 4).permute(2, *output).reshape(2, 8)
    expected = state.reshape((2,) * 4).permute(2, *physical).reshape(2, 8) @ tensor.T
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=0)
    assert [gate.name for gate in circuit.to_ir().instructions] == ["x"]


@pytest.mark.parametrize("consumer", ["add", "xor_qrom"])
@pytest.mark.parametrize("factory_created", [False, True])
def test_rebindings_extend_unchanged_append_consumer_and_all_measurement_histories(
    monkeypatch, consumer, factory_created
):
    from tests.unit import test_component_program_builder as scenario

    seen = []

    class ReboundBuilder(ComponentProgramBuilder):
        def call_dict(self, callee, bindings):
            outputs = super().call_dict(callee, bindings)
            if callee == "packed_leaf":
                outputs = {
                    name: self.join(self.split(value), value.register.dtype)
                    for name, value in outputs.items()
                }
                seen.append(self)
            return outputs

    monkeypatch.setattr(scenario, "ComponentProgramBuilder", ReboundBuilder)
    scenario.test_builder_replaces_existing_append_consumer_with_exact_instructions_and_all_branches(
        monkeypatch, consumer, factory_created
    )
    assert len(seen) == 1
    assert len(seen[0]._operations) == 9


def test_join_defaults_and_helper_names_avoid_entry_and_definition_collisions():
    from tests.unit.test_component_program_builder import identity

    signature = Signature([port()])
    functions = [identity("__fq_rebinding_1", []), identity("__fq_rebinding_3", [])]
    builder = ComponentProgramBuilder(signature, functions, name="__fq_rebinding_2")
    bits = builder.split(builder.inputs["word"])
    word = builder.join(bits)
    assert word.register.dtype == RegisterType("opaque", 3)
    module = builder.finalize({"word": word})
    assert len({function.name for function in module.functions}) == len(
        module.functions
    )
    assert (
        not lower_register_program(module, entry="__fq_rebinding_2")
        .functions[0]
        .blocks[0]
        .operations
    )


@pytest.mark.parametrize("failure", ["not_scalar", "not_bit", "bad_type", "bad_rank"])
def test_rebinding_guards_fail_before_reserving_calls(failure):
    builder = ComponentProgramBuilder(Signature([port()]))
    word = builder.inputs["word"]
    with pytest.raises(ValueError):
        if failure == "not_scalar":
            builder.split(np.asarray([word]))
        elif failure == "not_bit":
            builder.join([word])
        elif failure == "bad_type":
            builder.split(
                replace(word, register=Register("word", RegisterType("signed", 3)))
            )
        else:
            builder.join([[word]])
    assert not builder._operations
    assert len(builder._available) == 1
