"""Actual CAdd libraries resolve through existing typed execution owners."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.registers.arithmetic.controlled_addition_component import (
    controlled_addition_program,
)
from flagquantum.compiler.registers.component_graph import _namespace_helpers
from flagquantum.compiler.registers.control.specialized_component_control import (
    multi_controlled_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register, RegisterType
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")


@pytest.mark.parametrize("large", (False, True))
def test_actual_factories_preserve_complete_library(large: bool) -> None:
    from qualtran.bloqs.arithmetic.controlled_addition import _cadd_large, _cadd_small
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto
    from qualtran.testing import assert_bloq_example_serializes

    factory = _cadd_large if large else _cadd_small
    source = factory.make()
    assert_bloq_example_serializes(factory)
    for depth in ((0, 1) if large else (0, 1, 2)):
        original = bloqs_to_proto(source, max_depth=depth)
        wire = original.SerializeToString(deterministic=True)
        library = deserialize_component_library(wire)
        assert serialize_component_library(library) == wire
        assert (
            deserialize_component_library(serialize_component_library(library))
            == library
        )
        assert source in bloqs_from_proto(original)
        root = library.entries[0].declaration
        assert root.registers is not None
        assert tuple(reg.bitsize for reg in root.registers) == (
            1,
            1000 if large else 3,
            1000 if large else 3,
        )

    if large:
        with pytest.raises(RecursionError):
            bloqs_to_proto(source, max_depth=2)


@pytest.mark.parametrize("signed", (False, True))
@pytest.mark.parametrize("cv", (0, 1))
def test_symbolic_declarations_resolve_without_physical_specialization(
    signed: bool, cv: int
) -> None:
    from qualtran import DecomposeTypeError, QInt, QUInt
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd
    from qualtran.serialization.bloq import bloqs_to_proto
    from sympy import Symbol

    source = CAdd(
        (QInt if signed else QUInt)(Symbol("n", positive=True, integer=True)), cv=cv
    )
    wire = bloqs_to_proto(source, max_depth=0).SerializeToString(deterministic=True)
    library = deserialize_component_library(wire)
    declaration = library.entries[0].declaration
    assert serialize_component_library(library) == wire

    def declared(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord:
        assert registers is not None
        assert arguments["a_dtype"] == arguments["b_dtype"] == registers[1].dtype
        assert arguments["cv"] == cv
        return FunctionRecord(name, (), registers=registers)

    resolved = resolve_component_program(library, 0, {declaration.name: declared})
    assert resolved.functions[0].registers == declaration.registers
    with pytest.raises(DecomposeTypeError, match="symbolic"):
        source.decompose_bloq()
    assert declaration.registers is not None
    with pytest.raises(ValueError, match="unsupported executable"):
        controlled_addition_program(declaration.registers[1].dtype, control_value=cv)


@pytest.mark.parametrize("old,new", tuple(product((0, 1), repeat=2)))
@pytest.mark.parametrize("signed", (False, True))
def test_outer_object_references_resolve_to_exact_executable_calls(
    old: int, new: int, signed: bool
) -> None:
    from qualtran import CtrlSpec, QInt, QUInt
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd
    from qualtran.serialization.bloq import bloqs_to_proto

    dtype = QInt if signed else QUInt
    source: Any = CAdd(dtype(1), dtype(2), cv=old).controlled(CtrlSpec(cvs=new))
    wire = bloqs_to_proto(source, max_depth=0).SerializeToString(deterministic=True)
    library = deserialize_component_library(wire)
    assert len(library.entries) == 2
    assert serialize_component_library(library) == wire
    modules: dict[str, ProgramRecord] = {}

    def child_factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        a, b = arguments["a_dtype"], arguments["b_dtype"]
        assert isinstance(a, RegisterType) and isinstance(b, RegisterType)
        assert arguments["cv"] == 1
        record = controlled_addition_program(a, b)
        record = replace(
            record,
            functions=(replace(record.functions[0], name=name), *record.functions[1:]),
        )
        record = _namespace_helpers(record, name)
        assert record.functions[0].registers == registers
        modules[name] = record
        return record

    def outer_factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        child = arguments["ctrl_bloq"]
        assert isinstance(child, FunctionRecord)
        assert arguments["ctrl_reg_name"] == "ctrl"
        np.testing.assert_array_equal(arguments["cvs"], (new, old))
        base = modules[child.name]
        record = multi_controlled_component_program(
            base, "ctrl", (new, old), entry=child.name
        )
        inherited = {function.name for function in base.functions}
        record = replace(
            record,
            functions=(
                replace(record.functions[0], name=name),
                *(
                    function
                    for function in record.functions[1:]
                    if function.name not in inherited
                ),
            ),
        )
        assert record.functions[0].registers == registers
        return record

    factories = {
        entry.declaration.name: (
            child_factory if entry.declaration.name.endswith(".CAdd") else outer_factory
        )
        for entry in library.entries
    }
    resolved = resolve_component_program(library, 0, factories)
    child = controlled_addition_program(
        RegisterType("signed" if signed else "uint", 1),
        RegisterType("signed" if signed else "uint", 2),
    )
    direct = multi_controlled_component_program(child, "ctrl", (new, old))
    ports = {"ctrl": (0, 1), "a": (2,), "b": (3, 4)}
    operands = ((0,), (1,), (2,), (3, 4))
    work = (6, 7, 8)
    bound = bind_register_program(resolved, operands, clean=work)
    expected = bind_register_program(direct, operands, clean=work)
    assert bound.circuit == expected.circuit
    assert bound.outputs == expected.outputs
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit
    slots = {
        gate.wires[0]: gate.metadata["classical_bit"]
        for gate in bound.circuit.instructions
        if gate.name == "measure"
    }
    assert_source(
        bound.circuit.instructions,
        reference_arithmetic_gates(source, ports, work, slots),
    )
    counts = Counter(gate.name for gate in bound.circuit.instructions)
    assert counts["t"] + counts["tdg"] == 12
    assert counts["measure"] == counts["reset"] == 3
    assert "any" not in counts
    raw = torch.arange(1, 65, dtype=torch.float64)
    amplitudes = torch.complex(raw, 2 - 3 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(512, dtype=torch.complex128)
    initial[::8] = amplitudes
    target = torch.zeros_like(initial)
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        32, 32
    )
    target[::8] = (matrix @ amplitudes.reshape(32, 2)).flatten()
    for history in product((0, 1), repeat=3):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(actual, target / 8**0.5, rtol=0, atol=3e-12)
        assert torch.count_nonzero(actual.reshape(-1, 8)[:, 1:]) == 0


@pytest.mark.parametrize("left,right", ((1, 1), (1, 2), (2, 3), (3, 3)))
@pytest.mark.parametrize("cv", (0, 1))
@pytest.mark.parametrize("adjoint", (False, True))
def test_original_typed_sdk_consumer_preserves_complete_stream(
    left: int, right: int, cv: int, adjoint: bool
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd

    from flagquantum.ecosystem.cirq._register_import import import_cirq_register_gate
    from tests.unit.test_register_tensor import native_signature

    cirq = pytest.importorskip("cirq")
    source: Any = CAdd(QUInt(left), QUInt(right), cv=cv)
    if adjoint:
        source = source.adjoint()
    imported = import_cirq_register_gate(
        source.on(*reversed(cirq.LineQubit.range(1 + left + right))),
        signature=native_signature(source.signature),
    )
    assert imported is not None
    dtype_a, dtype_b = RegisterType("uint", left), RegisterType("uint", right)
    native = controlled_addition_program(
        dtype_a, dtype_b, control_value=cv, adjoint=adjoint
    )
    words = ((0,), tuple(range(1, left + 1)), tuple(range(left + 1, left + right + 1)))
    work = tuple(range(left + right + 1, 2 * (left + right)))
    actual = bind_register_program(imported, words, clean=work)
    expected = bind_register_program(native, words, clean=work)
    assert actual.circuit.instructions == expected.circuit.instructions
    assert actual.outputs == expected.outputs
    assert CircuitIR.from_json(actual.circuit.to_json()) == actual.circuit
