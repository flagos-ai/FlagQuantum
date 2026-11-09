"""Real finite functions replace pinned source call construction observations."""

from dataclasses import replace

import numpy as np
import pytest

from flagquantum.compiler.registers.component_bindings import (
    RegisterBindingError,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.register_programs import lower_register_program
from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    ValueRef,
    program_to_circuit,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_register_component_calls import SourceCalls, port

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def identity(name, signature):
    """A real identity body, with one independently owned value per element."""
    signature = Signature(signature)
    values = tuple(
        ValueRef("entry", index, "register")
        for index, _ in enumerate(
            (reg, idx) for reg in signature.lefts() for idx in reg.all_idxs()
        )
    )
    return FunctionRecord(
        name,
        (BlockRecord("entry", values, return_values=values),),
        registers=tuple(signature),
    )


def binding_values(values):
    return tuple(
        binding.value for value in values.values() for binding in np.asarray(value).flat
    )


@pytest.mark.parametrize("width", [0, 1, 3])
@pytest.mark.parametrize("shape", [(), (2,), (2, 3), (0, 2)])
def test_real_identity_calls_match_original_order_ownership_and_zero_gate_lowering(
    width, shape
):
    signature = Signature([port("z", width, shape), port("a", 2)])
    functions = [identity("first", signature), identity("second", signature)]
    native = ComponentProgramBuilder(signature, functions)
    source = SourceCalls(signature)
    inputs = native.inputs
    original = source.inputs
    prior = set(binding_values(inputs))
    assert len(prior) == sum(len(tuple(reg.all_idxs())) for reg in signature.lefts())
    for name in ("first", "second"):
        inputs = native.call_dict(name, dict(reversed(inputs.items())))
        original = source.call(name, signature, dict(reversed(original.items())))
        assert list(inputs) == list(original) == ["z", "a"]
        current = set(binding_values(inputs))
        assert not prior & current
        assert current == {binding.value for binding in native._available}
        prior |= current
        for reg in signature.rights():
            ours = np.asarray(inputs[reg.name])
            theirs = np.asarray(original[reg.name])
            assert ours.shape == theirs.shape == reg.shape
            assert [binding.index for binding in ours.flat] == [
                index for _, _, index in source.identities(theirs)
            ]
    module = native.finalize(inputs)
    source.finalize(original)
    assert not native._available and not source.available
    assert verify_program_record(module) == ()
    block = lower_register_program(module).functions[0].blocks[0]
    assert not block.operations
    assert block.arguments == block.return_values
    assert len(block.arguments) == sum(
        reg.bitsize * len(tuple(reg.all_idxs())) for reg in signature.lefts()
    )
    assert [len(op.operands) for op in native._operations] == [
        len(module.functions[0].blocks[0].arguments)
    ] * 2


@pytest.mark.parametrize("failure", ["missing", "extra", "dtype", "duplicate", "short"])
def test_failed_real_call_matches_source_consumption_prefix_and_reserves_instance(
    failure,
):
    signature = Signature([port("first"), port("second")])
    callee = signature
    if failure == "dtype":
        callee = Signature([port("first", 4), port("second")])
    elif failure == "short":
        callee = Signature([port("first"), port("second", shape=(2,))])
    native = ComponentProgramBuilder(
        signature, [identity("failing", callee), identity("empty", [])]
    )
    source = SourceCalls(signature)
    native_args, source_args = dict(native.inputs), dict(source.inputs)
    for args in (native_args, source_args):
        if failure == "missing":
            args.pop("second")
        elif failure == "extra":
            args["extra"] = args["first"]
        elif failure == "duplicate":
            args["second"] = args["first"]
        elif failure == "short":
            args["second"] = [args["second"]]
    exception = IndexError if failure == "short" else RegisterBindingError
    with pytest.raises(exception):
        native.call_dict("failing", native_args)
    with pytest.raises(IndexError if failure == "short" else source.error_type):
        source.call("failing", callee, source_args)
    consumed = len(source.connections)
    assert len(native._operations[0].operands) == consumed
    assert len(native._available) == len(source.available)
    assert native._operations[0].results == ()
    assert (
        native.call_dict("empty", {}) == source.call("empty", Signature([]), {}) == {}
    )
    assert len(native._operations) == len(source.instances) == 2


@pytest.mark.parametrize(
    "failure", ["missing", "extra", "dtype", "duplicate", "unused"]
)
def test_failed_return_matches_source_prefix_and_retry(failure):
    from tests.unit.test_register_component_calls import (
        test_failed_return_preserves_prefix_and_rejects_unconsumed_values,
    )

    class NativeReturns:
        error_type = RegisterBindingError

        def __init__(self, signature):
            self.builder = ComponentProgramBuilder(signature)
            self.inputs = self.builder.inputs
            self.signature = signature

        @property
        def available(self):
            return {
                ("entry", b.register.name, b.index) for b in self.builder._available
            }

        @property
        def instances(self):
            return ()

        @property
        def connections(self):
            right = [
                (reg.name, index)
                for reg in self.signature.rights()
                for index in reg.all_idxs()
            ]
            names = {
                b.value: b
                for value in self.inputs.values()
                for b in np.asarray(value).flat
            }
            return tuple(
                (
                    ("entry", names[value].register.name, names[value].index),
                    ("return", name, index),
                )
                for value, (name, index) in zip(
                    self.builder._returned, right, strict=False
                )
            )

        def finalize(self, bindings):
            return self.builder.finalize(bindings)

    test_failed_return_preserves_prefix_and_rejects_unconsumed_values(
        NativeReturns, failure
    )


@pytest.mark.parametrize("form", ["call_dict", "call_tuple", "call"])
@pytest.mark.parametrize("ports", [0, 1, 2])
def test_call_result_forms_use_real_identity_bodies(form, ports):
    signature = Signature([port(f"word{i}") for i in range(ports)])
    builder = ComponentProgramBuilder(signature, [identity("identity", signature)])
    result = getattr(builder, form)("identity", builder.inputs)
    if form == "call_dict":
        returned = result
    elif form == "call":
        if ports == 0:
            assert result is None
            returned = {}
        else:
            returned = dict(
                zip(builder.inputs, (result,) if ports == 1 else result, strict=True)
            )
    else:
        returned = dict(zip(builder.inputs, result, strict=True))
    assert verify_program_record(builder.finalize(returned)) == ()


def test_failed_no_port_call_retains_a_valid_empty_instance_as_in_the_source():
    signature = Signature([])
    builder = ComponentProgramBuilder(signature, [identity("empty", [])])
    source = SourceCalls(signature)
    with pytest.raises(RegisterBindingError):
        builder.call_dict("empty", {"extra": object()})
    with pytest.raises(source.error_type):
        source.call("empty", signature, {"extra": object()})
    assert verify_program_record(builder.finalize({})) == ()
    source.finalize({})
    assert len(builder._operations) == len(source.instances) == 1


def test_finalization_verifies_supplied_body_instead_of_trusting_signature():
    signature = Signature([port()])
    body = identity("bad", signature)
    body = replace(body, blocks=(replace(body.blocks[0], return_values=()),))
    builder = ComponentProgramBuilder(signature, [body])
    result = builder.call_dict("bad", builder.inputs)
    with pytest.raises(ValueError, match="invalid constructed"):
        builder.finalize(result)


def test_definition_names_and_execution_profile_fail_explicitly():
    signature = Signature([port()])
    function = identity("leaf", signature)
    with pytest.raises(ValueError, match="conflicting"):
        ComponentProgramBuilder(signature, [function, function])
    with pytest.raises(ValueError, match="conflicting"):
        ComponentProgramBuilder(signature, [function], name="leaf")
    builder = ComponentProgramBuilder(signature, [function])
    with pytest.raises(ValueError, match="unknown"):
        builder.call_dict("absent", builder.inputs)
    assert not builder._operations
    with pytest.raises(ValueError, match="unsupported executable"):
        ComponentProgramBuilder(
            Signature([Register("x", RegisterType("classical_bit", 0, 1))])
        )
    gate = replace(function, registers=None)
    builder = ComponentProgramBuilder(signature, [gate])
    with pytest.raises(ValueError, match="immutable declared ports"):
        builder.call_dict("leaf", builder.inputs)
    assert not builder._operations


@pytest.mark.parametrize("consumer", ["add", "xor_qrom"])
@pytest.mark.parametrize("factory_created", [False, True])
def test_builder_replaces_existing_append_consumer_with_exact_instructions_and_all_branches(
    monkeypatch, consumer, factory_created
):
    from flagquantum.compiler.registers.arithmetic import addition
    from flagquantum.compiler.registers.lookup.qrom_table import QromTable
    from flagquantum.core.ir import CircuitIR
    from tests.unit.test_register_bindings import (
        NativeContract,
        test_bindings_feed_append_with_exact_measured_channels_and_resources,
    )
    from tests.unit.test_register_program_lowering import packed_circuit

    original = addition.add_ir if consumer == "add" else QromTable.bind_packed
    seen = []

    def through_builder(*args, **kwargs):
        gates = original(*args, **kwargs)
        width = 1 + max(wire for gate in gates for wire in gate.wires)
        circuit = CircuitIR(width, gates)
        widths = (2, 3, 2, 0) if consumer == "add" else (2, 2, 1, 0)
        existing = packed_circuit(circuit, widths)
        functions = tuple(f for f in existing.functions if f.name != "main")
        leaf = next(f for f in functions if f.name == "packed_leaf")
        builder = ComponentProgramBuilder(Signature(leaf.registers), functions)
        outputs = builder.call_dict("packed_leaf", builder.inputs)
        module = replace(builder.finalize(outputs), canonical_payload=circuit.to_dict())
        restored = program_to_circuit(lower_register_program(module))
        assert restored.state == "supported_exact", restored.diagnostics
        result = CircuitIR.from_dict(restored.canonical_payload)
        assert result.instructions == gates
        seen.append(module)
        return result.instructions

    if consumer == "add":
        monkeypatch.setattr(addition, "add_ir", through_builder)
    else:
        monkeypatch.setattr(QromTable, "bind_packed", through_builder)
    test_bindings_feed_append_with_exact_measured_channels_and_resources(
        NativeContract, consumer, factory_created
    )
    assert len(seen) == 1


def test_stale_handle_failure_reserves_its_call_and_cannot_create_successors():
    signature = Signature([port()])
    builder = ComponentProgramBuilder(signature, [identity("identity", signature)])
    first = builder.call_dict("identity", builder.inputs)
    second = builder.call_dict("identity", first)
    before = set(builder._available)
    with pytest.raises(RegisterBindingError, match="not an available"):
        builder.call_dict("identity", first)
    assert builder._available == before
    assert len(builder._operations) == 3
    assert builder._operations[-1].operands == builder._operations[-1].results == ()
    with pytest.raises(ValueError, match="invalid constructed"):
        builder.finalize(second)


def test_builder_snapshots_declarations_and_final_modules_are_immutable_snapshots():
    signature = Signature([port()])
    builder = ComponentProgramBuilder(signature, [identity("empty", [])], name="outer")
    signature._registers = ()
    module = builder.finalize(builder.inputs)
    assert module.functions[0].name == "outer"
    assert len(module.functions[0].registers) == 1
    builder.call_dict("empty", {})
    assert module.functions[0].blocks[0].operations == ()
    assert (
        lower_register_program(module, entry="outer").functions[0].blocks[0].operations
        == ()
    )
