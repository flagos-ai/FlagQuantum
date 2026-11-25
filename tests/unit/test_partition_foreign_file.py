"""Resolve original partition factories without moving SDK objects into runtime."""

from collections.abc import Mapping
from math import prod
from typing import Any

import pytest
import torch

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
from flagquantum.compiler.registers.layout.component_partition import (
    partition_component_program,
    partition_signature,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import Register
from tests.unit.test_subtraction_foreign_file import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def resolve_original(source: Any, depth: int) -> ProgramRecord:
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    proto = bloqs_to_proto(source, max_depth=depth)
    wire = proto.SerializeToString(deterministic=True)
    library = deserialize_component_library(wire)
    assert (
        serialize_component_library(library) == wire
        and bloqs_from_proto(proto)[0] == source
    )

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        assert registers is not None
        if "regs" in arguments:
            assert set(arguments) == {"regs", "n", "partition"}
            raw = arguments["regs"]
            assert isinstance(raw, tuple) and all(
                isinstance(reg, Register) for reg in raw
            )
            parts = tuple(reg for reg in raw if isinstance(reg, Register))
            assert sum(reg.total_qubits() for reg in parts) == arguments["n"]
        else:
            assert set(arguments) == {"n1", "n2"}
            parts = tuple(reg for reg in registers if reg.name != "x")
        reverse = registers[0].side.name == "RIGHT"
        assert tuple(partition_signature(parts, reverse=reverse)) == registers
        if any(type(reg.bitsize) is not int for reg in parts):
            return FunctionRecord(name, (), registers=registers)
        return partition_component_program(parts, name, reverse)

    program = resolve_component_program(
        library, 0, {entry.declaration.name: factory for entry in library.entries}
    )
    assert verify_program_record(program) == ()
    assert program.functions[0].registers == tuple(native_signature(source.signature))
    return program


@pytest.mark.parametrize("example", ("_partition", "_split2", "_join2"))
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_all_actual_factories_preserve_native_foreign_signature_and_sdk_disposition(
    example: str, depth: int
) -> None:
    np = pytest.importorskip("numpy")
    from qualtran.bloqs.bookkeeping import partition
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = getattr(partition, example).make()
    program = resolve_original(source, depth)
    assert get_cost_value(source, QECGatesCost()).asdict() == {}
    if example != "_partition":
        assert program.functions[0].blocks == ()
        with pytest.raises(ValueError, match="concrete|bodies"):
            bind_register_program(program, ((0, 1),))
        if example == "_split2":
            with pytest.raises((TypeError, ValueError)):
                source.as_cirq_op(None, x=np.arange(3))
        else:
            from flagquantum.compiler.registers.layout.partition import (
                join_register_pair,
            )

            operation, outputs = source.as_cirq_op(
                None, y1=np.asarray([2]), y2=np.asarray([1, 0])
            )
            assert operation is None
            assert tuple(outputs["x"]) == join_register_pair((2,), (1, 0))
        assert source.as_pl_op(object()) is None
        return
    wires = tuple(reversed(range(49)))
    bound = bind_register_program(program, (wires,))
    operation, views = source.as_cirq_op(None, x=np.asarray(wires))
    assert operation is None
    assert bound.outputs == tuple(
        tuple(int(wire) for wire in word)
        for value in views.values()
        for word in value.reshape(-1, value.shape[-1])
    )
    assert bound.events == () and bound.circuit.instructions == ()
    assert get_cost_value(source, QubitCount()) == 49
    assert source.as_pl_op(object()) is None


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize(
    "profile", ("opaque", "signed", "signed_negative", "empty", "empty_negative")
)
@pytest.mark.parametrize("shape", ((2,), (1, 2), (0, 2)))
def test_shaped_concrete_foreign_partition_and_inverse_match_every_complex_amplitude(
    reverse: bool,
    profile: str,
    shape: tuple[int, ...],
) -> None:
    np = pytest.importorskip("numpy")
    from qualtran import QAny, QFxp
    from qualtran import Register as OriginalRegister
    from qualtran.bloqs.bookkeeping import Partition
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    dtype = {
        "opaque": QAny(2),
        "signed": QFxp(2, 1, signed=True),
        "signed_negative": QFxp(2, -2, signed=True),
        "empty": QFxp(0, 0),
        "empty_negative": QFxp(0, -2),
    }[profile]
    width = int(dtype.num_qubits)
    packed_width = prod(shape) * width + 2
    source = Partition(
        packed_width,
        (OriginalRegister("words", dtype, shape), OriginalRegister("tail", QAny(2))),
        partition=not reverse,
    )
    program = resolve_original(source, 1)
    wires = tuple(reversed(range(packed_width)))
    ports = {
        "words": np.asarray(wires[:-2]).reshape((*shape, width)),
        "tail": np.asarray(wires[-2:]),
    }
    inputs = (
        tuple(
            tuple(int(wire) for wire in port)
            for reg in source.regs
            for index in reg.all_idxs()
            for port in (ports[reg.name][index],)
        )
        if reverse
        else (wires,)
    )
    operation, views = (
        source.as_cirq_op(None, **ports)
        if reverse
        else source.as_cirq_op(None, x=np.asarray(wires))
    )
    assert operation is None
    expected_outputs = (
        (wires,)
        if reverse
        else tuple(
            tuple(word)
            for reg in source.regs
            for index in reg.all_idxs()
            for word in (views[reg.name][index],)
        )
    )
    bound = bind_register_program(program, inputs)
    assert bound.outputs == expected_outputs
    assert bound.events == () and bound.circuit.instructions == ()
    state = torch.randn(
        1 << (packed_width + 1),
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(816),
    )
    state /= state.norm()
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        1 << packed_width, 1 << packed_width
    )
    order = (*wires, packed_width)
    expected = (
        (
            matrix
            @ state.reshape((2,) * (packed_width + 1))
            .permute(order)
            .reshape(1 << packed_width, 2)
        )
        .reshape((2,) * (packed_width + 1))
        .permute(order)
        .flatten()
    )
    circuit = Circuit(packed_width + 1, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(packed_width + 1, bound.circuit.instructions))
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)
    assert get_cost_value(source, QECGatesCost()).asdict() == {}
    assert get_cost_value(source, QubitCount()) == packed_width
