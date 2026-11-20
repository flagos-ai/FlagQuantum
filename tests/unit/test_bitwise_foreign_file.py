"""Roundtrip complete bitwise factories through native interchange owners."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Literal

import pytest
import torch
from attrs import evolve

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.registers.bits.bitwise_component import bitwise_word_program
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "factory_name", ("_xork", "_xor", "_xor_symb", "_bitwise_not", "_bitwise_not_symb")
)
def test_all_factory_objects_preserve_original_wire_metadata(factory_name: str) -> None:
    from qualtran.bloqs.arithmetic import bitwise
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto
    from qualtran.testing import assert_bloq_example_serializes

    factory = getattr(bitwise, factory_name)
    source = factory.make()
    assert_bloq_example_serializes(factory)
    for depth in (0, 1, 2):
        original = bloqs_to_proto(source, name="bitwise", max_depth=depth)
        wire = original.SerializeToString(deterministic=True)
        native = deserialize_component_library(wire)
        assert serialize_component_library(native) == wire
        assert (
            deserialize_component_library(serialize_component_library(native)) == native
        )
        assert source in bloqs_from_proto(original)
        root = native.entries[0].declaration
        assert root.name.endswith("." + type(source).__name__)
        assert root.registers is not None
        arguments = {argument.name: argument.value for argument in root.arguments}
        assert arguments["dtype"] == root.registers[0].dtype
        if factory_name == "_xork":
            assert arguments["k"] == source.k


@pytest.mark.parametrize("factory_name", ("_xor_symb", "_bitwise_not_symb"))
def test_symbolic_factory_declaration_then_explicit_concrete_construction(
    factory_name: str,
) -> None:
    from qualtran.bloqs.arithmetic import bitwise
    from qualtran.serialization.bloq import bloqs_to_proto

    source = getattr(bitwise, factory_name).make()
    library = deserialize_component_library(
        bloqs_to_proto(source, max_depth=0).SerializeToString()
    )
    row = library.entries[0]
    declaration = row.declaration
    assert declaration.registers is not None
    symbolic_dtype = declaration.registers[0].dtype
    operation: Literal["xor", "bitwise_not"] = (
        "xor" if factory_name == "_xor_symb" else "bitwise_not"
    )

    def declared(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord:
        assert registers is not None and arguments == {"dtype": symbolic_dtype}
        return FunctionRecord(name, (), registers=registers)

    program = resolve_component_program(library, 0, {declaration.name: declared})
    assert program.functions[0].registers == declaration.registers
    with pytest.raises(ValueError, match="unsupported executable register descriptor"):
        bitwise_word_program(symbolic_dtype, operation)
    # Width assignment is explicit caller construction, not symbolic execution.
    for width in (1, 3, 5):
        dtype = replace(symbolic_dtype, quantum_bits=width)
        concrete_source = evolve(source, dtype=evolve(source.dtype, bitsize=width))
        registers = tuple(
            Register(reg.name, dtype, reg.shape, reg.side)
            for reg in declaration.registers
        )
        concrete_row = replace(
            row,
            declaration=replace(
                declaration,
                registers=registers,
                arguments=tuple(
                    replace(arg, value=dtype) for arg in declaration.arguments
                ),
            ),
        )
        concrete = replace(library, entries=(concrete_row,))

        def executable(
            name: str,
            ports: tuple[Register, ...] | None,
            arguments: Mapping[str, ResolvedArgument],
        ) -> ProgramRecord:
            assert name == "main" and ports is not None
            value = arguments["dtype"]
            assert isinstance(value, RegisterType)
            assert all(port.dtype == value for port in ports)
            return bitwise_word_program(value, operation)

        resolved = resolve_component_program(
            concrete, 0, {declaration.name: executable}
        )
        total = width * len(registers)
        wires = tuple(reversed(range(total)))
        groups = tuple(wires[i : i + width] for i in range(0, total, width))
        bound = bind_register_program(resolved, groups)
        emitter = import_register_emitter(concrete_source)
        expected = emitter(
            dict(zip((reg.name for reg in registers), groups, strict=True)),
            (),
            (),
            False,
        )
        assert bound.circuit.instructions == expected
        assert bound.outputs == groups and not bound.events
        initial = torch.arange(1, (1 << (total + 1)) + 1, dtype=torch.float64).to(
            torch.complex128
        )
        initial += 1j * initial.flip(0)
        initial /= initial.norm()
        actual = Circuit(total + 1, inputs=initial, dtype=torch.complex128)
        actual.append(FieldOperation(total + 1, bound.circuit.instructions))
        reference = Circuit(total + 1, inputs=initial, dtype=torch.complex128)
        reference.append(FieldOperation(total + 1, expected))
        torch.testing.assert_close(
            actual.state(), reference.state(), rtol=0, atol=2e-12
        )
        actual.append(FieldOperation(total + 1, bound.circuit.instructions))
        torch.testing.assert_close(
            actual.state().flatten(), initial, rtol=0, atol=2e-12
        )
        assert len(expected) == width
        assert all(
            gate.name == ("cx" if operation == "xor" else "x") for gate in expected
        )
