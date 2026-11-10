"""Real function bodies cover remaining directional call ownership scenarios."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_bindings import RegisterBindingError
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.register_programs import lower_register_program
from flagquantum.core.ir.program import FunctionRecord, verify_program_record
from flagquantum.core.ir.program.register_types import classical_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_program_classical_registers import identity
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def original(signature: Signature) -> Any:
    # The pinned test observer exposes opaque upstream Soquets.
    factory: Any = SourceCalls
    return factory(signature)


def replacement(signature: Signature) -> tuple[FunctionRecord, ...]:
    """Use actual Free/Allocate bodies for directional quantum ports."""
    builder = ComponentProgramBuilder(signature, name="replace")
    for register in signature.lefts():
        for binding in np.asarray(builder.inputs[register.name]).flat:
            builder.free(binding)
    outputs = {
        register.name: builder.allocate(dtype=register.dtype)
        for register in signature.rights()
    }
    module = builder.finalize(outputs)
    assert verify_program_record(module) == ()
    return module.functions


@pytest.mark.parametrize("form", ["dict", "tuple", "auto"])
@pytest.mark.parametrize("output_count", [0, 1, 2])
def test_directional_output_forms_consume_real_lefts_and_allocate_rights(
    form: str, output_count: int
) -> None:
    registers = (
        Register("input", RegisterType("uint", 3), side=RegisterSide.LEFT),
        *(
            Register(f"out{i}", RegisterType("uint", 3), side=RegisterSide.RIGHT)
            for i in range(output_count)
        ),
    )
    signature = Signature(registers)
    native = ComponentProgramBuilder(signature, replacement(signature))
    source = original(signature)
    call = {"dict": native.call_dict, "tuple": native.call_tuple, "auto": native.call}[
        form
    ]
    result = call("replace", native.inputs)
    expected = source.call("replace", signature, source.inputs, form)
    names = tuple(register.name for register in signature.rights())
    if form == "dict":
        assert isinstance(result, Mapping)
        assert tuple(result) == tuple(expected) == names
        returned = dict(result)
        source_returned = expected
    elif form == "auto" and output_count == 0:
        assert result is expected is None
        returned, source_returned = {}, {}
    else:
        values = (result,) if form == "auto" and output_count == 1 else result
        source_values = (
            (expected,) if form == "auto" and output_count == 1 else expected
        )
        assert isinstance(values, tuple)
        assert len(values) == len(source_values) == output_count
        returned = dict(zip(names, values, strict=True))
        source_returned = dict(zip(names, source_values, strict=True))
    module = native.finalize(returned)
    source.finalize(source_returned)
    assert verify_program_record(module) == ()
    assert not native._available and not source.available
    call_operation = native._operations[0]
    assert len(call_operation.operands) == 1
    assert len(call_operation.results) == output_count
    assert len(source.connections) == 1 + output_count
    leaf = module.functions[1]
    assert len(leaf.blocks[0].operations) == 1 + output_count


def test_same_name_opposite_sides_use_real_distinct_width_lifetimes() -> None:
    signature = Signature(
        (
            Register("word", RegisterType("uint", 2), side=RegisterSide.LEFT),
            Register("word", RegisterType("uint", 5), side=RegisterSide.RIGHT),
        )
    )
    native = ComponentProgramBuilder(signature, replacement(signature))
    source = original(signature)
    result = native.call_dict("replace", native.inputs)
    expected = source.call("replace", signature, source.inputs)
    before = np.asarray(native.inputs["word"]).item()
    after = np.asarray(result["word"]).item()
    assert before.register.bitsize == 2 and after.register.bitsize == 5
    assert before.value != after.value
    module = native.finalize(result)
    source.finalize(expected)
    assert verify_program_record(module) == ()
    assert source.connections == (
        (("entry", "word", ()), (0, "word", ())),
        ((0, "word", ()), ("return", "word", ())),
    )


def test_array_alias_retains_same_row_major_consumption_prefix() -> None:
    registers = (Register("word", RegisterType("uint", 3), (2, 2)),)
    signature = Signature(registers)
    native = ComponentProgramBuilder(signature, (identity("leaf", registers),))
    source = original(signature)
    for builder, call, error in (
        (native, lambda values: native.call_dict("leaf", values), RegisterBindingError),
        (
            source,
            lambda values: source.call("leaf", signature, values),
            source.error_type,
        ),
    ):
        values = np.asarray(builder.inputs["word"]).copy()
        values[1, 0] = values[0, 0]
        with pytest.raises(error, match="not an available"):
            call({"word": values})
    assert len(native._operations) == len(source.instances) == 1
    assert len(native._operations[0].operands) == len(source.connections) == 2
    assert native._operations[0].results == ()
    assert {binding.index for binding in native._available} == {(1, 0), (1, 1)}
    assert [target[2] for _, target in source.connections] == [(0, 0), (0, 1)]


@pytest.mark.parametrize("consume_left", [False, True])
def test_explicit_empty_return_verifies_real_consumed_left_body(
    consume_left: bool,
) -> None:
    registers = (
        (Register("word", RegisterType("uint", 3), side=RegisterSide.LEFT),)
        if consume_left
        else ()
    )
    signature = Signature(registers)
    native = ComponentProgramBuilder(signature, replacement(signature))
    source = original(signature)
    if consume_left:
        assert native.call_dict("replace", native.inputs) == {}
        assert source.call("replace", signature, source.inputs) == {}
    assert verify_program_record(native.finalize({})) == ()
    source.finalize({})
    assert source.final_bindings() == {}
    assert not native._available and not source.available


def test_empty_shaped_return_keeps_native_declaration_and_original_omission() -> None:
    registers = (Register("word", RegisterType("uint", 3), (0, 2)),)
    signature = Signature(registers)
    native = ComponentProgramBuilder(signature, (identity("leaf", registers),))
    source = original(signature)
    result = native.call_dict("leaf", native.inputs)
    expected = source.call("leaf", signature, source.inputs)
    assert np.asarray(result["word"]).shape == expected["word"].shape == (0, 2)
    module = native.finalize(result)
    source.finalize(expected)
    assert verify_program_record(module) == ()
    assert module.functions[0].registers == registers
    assert source.final_bindings() == {}
    assert source.connections == ()
    lowered = lower_register_program(module).functions[0].blocks[0]
    assert lowered.arguments == lowered.return_values == ()
    assert lowered.operations == ()


def test_actual_cbit_descriptor_keeps_source_single_consumption_and_stale_prefix() -> (
    None
):
    registers = (Register("bit", classical_bit_type()),)
    signature = Signature(registers)
    native = ComponentProgramBuilder(signature, (identity("leaf", registers),))
    source = original(signature)
    result = native.call_dict("leaf", native.inputs)
    expected = source.call("leaf", signature, source.inputs)
    with pytest.raises(RegisterBindingError, match="not an available"):
        native.call_dict("leaf", native.inputs)
    with pytest.raises(source.error_type, match="not an available"):
        source.call("leaf", signature, source.inputs)
    assert len(native._operations) == len(source.instances) == 2
    assert native._operations[-1].operands == native._operations[-1].results == ()
    assert len(source.connections) == 1
    # The source keeps a signature-only failed instance. Core requires a real
    # callee body to have its declared operands and results before execution.
    with pytest.raises(ValueError, match="invalid constructed"):
        native.finalize(result)
    source.finalize(expected)
    assert not native._available and not source.available


@pytest.mark.parametrize("output_width", [0, 1, 5])
def test_directional_lifetimes_preserve_source_complex_spectator_and_zero_work(
    output_width: int,
) -> None:
    input_register = Register("word", RegisterType("uint", 2), side=RegisterSide.LEFT)
    output_register = Register(
        "word", RegisterType("uint", output_width), side=RegisterSide.RIGHT
    )
    leaf_signature = Signature((input_register, output_register))
    signature = Signature(
        (*leaf_signature, Register("spectator", RegisterType("uint", 1)))
    )
    native = ComponentProgramBuilder(signature, replacement(leaf_signature))
    outputs = native.call_dict("replace", {"word": native.inputs["word"]})
    module = native.finalize(
        {"word": outputs["word"], "spectator": native.inputs["spectator"]}
    )
    bound = bind_register_program(
        module,
        ((0, 1), (2,)),
        clean=(3, 4, 5),
        input_ownership=((0, False), (1, False)),
        allow_workspace_outputs=True,
    )
    assert bound.circuit.instructions == ()
    assert bound.outputs[1] == (2,)
    assert len(bound.outputs[0]) == output_width
    assert sorted(event.kind for event in bound.events) == ["allocate", "free"]

    source = original(signature)
    source.builder.free(source.inputs["word"])
    dtype = source.signature(Signature((output_register,)))[0].dtype
    word = source.builder.allocate(dtype=dtype)
    composite = source.builder.finalize(word=word, spectator=source.inputs["spectator"])
    source_matrix = np.asarray(composite.tensor_contract())
    initial_logical = np.zeros(8, dtype=np.complex128)
    initial_logical[:2] = (1, 1j)
    initial_logical /= np.sqrt(2)
    expected = np.einsum("ij,j->i", source_matrix, initial_logical, optimize=False)
    initial = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
    stride = 1 << (bound.circuit.n_wires - 1 - 2)
    initial[0], initial[stride] = 1 / np.sqrt(2), 1j / np.sqrt(2)
    result = forced_branch(initial, bound.circuit.instructions, ())
    torch.testing.assert_close(result, initial, rtol=0, atol=1e-12)
    actual = np.zeros(1 << (output_width + 1), dtype=np.complex128)
    actual[:2] = result[[0, stride]].numpy()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    assert torch.count_nonzero(result).item() == 2
