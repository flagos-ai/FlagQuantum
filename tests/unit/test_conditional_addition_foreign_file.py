"""Complete conditional arithmetic objects reuse typed interchange and execution."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from itertools import product

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
from flagquantum.compiler.registers.arithmetic.conditional_addition_component import (
    conditional_addition_program,
)
from flagquantum.compiler.registers.component_graph import _namespace_helpers
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.ecosystem.cirq._register_import import import_cirq_register_gate
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_tensor import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")
cirq = pytest.importorskip("cirq")
FACTORIES = (
    "_ctrl_add_or_sub_unsigned",
    "_ctrl_add_or_sub_signed",
    "_ctrl_add_or_sub_signed_symb",
)


@pytest.mark.parametrize("factory_name", FACTORIES)
def test_actual_factories_preserve_full_library_and_resolve_native_owner(
    factory_name: str,
) -> None:
    from qualtran.bloqs.arithmetic import controlled_add_or_subtract as original
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto
    from qualtran.testing import assert_bloq_example_serializes

    factory = getattr(original, factory_name)
    source = factory.make()
    assert_bloq_example_serializes(factory)
    for depth in (0, 1, 2):
        proto = bloqs_to_proto(source, max_depth=depth)
        wire = proto.SerializeToString(deterministic=True)
        library = deserialize_component_library(wire)
        assert serialize_component_library(library) == wire
        assert (
            deserialize_component_library(serialize_component_library(library))
            == library
        )
        assert source in bloqs_from_proto(proto)
    root_library = deserialize_component_library(
        bloqs_to_proto(source, max_depth=0).SerializeToString(deterministic=True)
    )
    declaration = root_library.entries[0].declaration

    def factory_owner(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        assert registers is not None
        a, b = arguments["a_dtype"], arguments["b_dtype"]
        assert a == registers[1].dtype and b == registers[2].dtype
        assert isinstance(a, RegisterType) and isinstance(b, RegisterType)
        assert arguments["add_when_ctrl_is_on"] == 1
        if factory_name.endswith("symb"):
            return FunctionRecord(name, (), registers=registers)
        record = conditional_addition_program(a, b)
        record = replace(
            record,
            functions=(replace(record.functions[0], name=name), *record.functions[1:]),
        )
        return _namespace_helpers(record, name)

    resolved = resolve_component_program(
        root_library, 0, {declaration.name: factory_owner}
    )
    assert resolved.functions[0].registers == declaration.registers
    if factory_name.endswith("symb"):
        assert declaration.registers is not None
        with pytest.raises(ValueError, match="unsupported executable"):
            conditional_addition_program(declaration.registers[1].dtype)
    else:
        words = ((0,), tuple(range(1, 9)), tuple(range(9, 17)))
        work = tuple(range(17, 24))
        dtype = RegisterType(
            ("signed" if factory_name == "_ctrl_add_or_sub_signed" else "uint"),
            8,
        )
        actual = bind_register_program(resolved, words, clean=work)
        expected = bind_register_program(
            conditional_addition_program(dtype), words, clean=work
        )
        assert actual.circuit == expected.circuit
        assert actual.outputs == expected.outputs


@pytest.mark.parametrize("signed", (False, True))
@pytest.mark.parametrize("left,right", ((1, 1), (1, 2), (2, 3), (3, 3)))
@pytest.mark.parametrize("add_on", (False, True))
def test_original_typed_sdk_stream_and_small_complex_branches(
    signed: bool, left: int, right: int, add_on: bool
) -> None:
    from qualtran import QInt, QUInt
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        ControlledAddOrSubtract,
    )

    dtype = QInt if signed else QUInt
    source = ControlledAddOrSubtract(
        dtype(left), dtype(right), add_when_ctrl_is_on=add_on
    )
    imported = import_cirq_register_gate(
        source.on(*reversed(cirq.LineQubit.range(1 + left + right))),
        signature=native_signature(source.signature),
    )
    assert imported is not None
    encoding = "signed" if signed else "uint"
    native = conditional_addition_program(
        RegisterType(encoding, left),
        RegisterType(encoding, right),
        add_when_control_is_on=add_on,
    )
    words = ((0,), tuple(range(1, left + 1)), tuple(range(left + 1, left + right + 1)))
    logical = 1 + left + right
    work = tuple(range(logical + 1, logical + right))
    actual = bind_register_program(imported, words, clean=work)
    expected = bind_register_program(native, words, clean=work)
    assert actual.circuit == expected.circuit
    assert actual.outputs == expected.outputs
    assert CircuitIR.from_json(actual.circuit.to_json()) == actual.circuit
    if left <= 1:
        raw = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
        amplitudes = torch.complex(raw, 1 - 3 * raw)
        amplitudes /= amplitudes.norm()
        initial = torch.zeros(1 << (logical + right), dtype=torch.complex128)
        initial[:: 1 << len(work)] = amplitudes
        target = torch.zeros_like(initial)
        matrix = torch.as_tensor(
            source.tensor_contract(), dtype=torch.complex128
        ).reshape(1 << logical, 1 << logical)
        target[:: 1 << len(work)] = (
            matrix @ amplitudes.reshape(1 << logical, 2)
        ).flatten()
        for history in product((0, 1), repeat=len(work)):
            branch = forced_branch(initial, actual.circuit.instructions, history)
            torch.testing.assert_close(
                branch, target / 2 ** (len(work) / 2), rtol=0, atol=3e-12
            )
            if work:
                assert (
                    torch.count_nonzero(branch.reshape(-1, 1 << len(work))[:, 1:]) == 0
                )


@pytest.mark.parametrize("add_on", (False, True))
@pytest.mark.parametrize("outer", (0, 1))
def test_actual_outer_sdk_execution_and_original_negative_control_boundary(
    add_on: bool, outer: int
) -> None:
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        ControlledAddOrSubtract,
    )
    from qualtran.bloqs.mcmt import And

    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    source = ControlledAddOrSubtract(
        QUInt(1), QUInt(2), add_when_ctrl_is_on=add_on
    ).controlled(CtrlSpec(cvs=outer))
    signature = native_signature(source.signature)
    operation = source.on(*cirq.LineQubit.range(5))
    if outer == 0:
        # The source's missing controlled erase decomposition recurses in unitary.
        with pytest.raises(RecursionError):
            cirq.unitary(And(uncompute=True).controlled())
        with pytest.raises(RecursionError):
            import_cirq_register_gate(operation, signature=signature)
        return
    imported = import_cirq_register_gate(operation, signature=signature)
    assert imported is not None
    bound = bind_register_program(imported, ((0,), (1,), (2,), (3, 4)), clean=(6, 7))
    assert bound.circuit.n_wires == 8
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    counts = Counter(gate.name for gate in bound.circuit.instructions)
    assert counts["t"] + counts["tdg"] == 8
    assert counts["ccx"] == 4
    assert counts["measure"] == counts["reset"] == 2
    model = get_cost_value(source, QECGatesCost())
    assert model.toffoli == 4 and model.and_bloq == 2 and model.measurement == 2
    assert model.clifford == (7 if add_on else 5)
    assert get_cost_value(source, QubitCount()) == 7
    assert not any(gate.name == "any" for gate in bound.circuit.instructions)
    ports = {"ctrl2": (0,), "ctrl": (1,), "a": (2,), "b": (3, 4)}
    slots = {
        gate.wires[0]: gate.metadata["classical_bit"]
        for gate in bound.circuit.instructions
        if gate.name == "measure"
    }
    assert_source(
        bound.circuit.instructions,
        reference_arithmetic_gates(
            source, ports, (6, 7), slots, keep_exchange_leaves=True
        ),
    )
    raw = torch.arange(1, 65, dtype=torch.float64)
    amplitudes = torch.complex(raw, 3 - 2 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[::4] = amplitudes
    expected = torch.zeros_like(initial)
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        32, 32
    )
    expected[::4] = (matrix @ amplitudes.reshape(32, 2)).flatten()
    for history in product((0, 1), repeat=2):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(actual, expected / 2, rtol=0, atol=1e-11)
        assert torch.count_nonzero(actual.reshape(-1, 4)[:, 1:]) == 0
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit
