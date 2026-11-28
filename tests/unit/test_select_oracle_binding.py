"""Replace grouped external SELECT objects through the packed native consumer."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha1
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
import torch
from attrs import evolve, frozen
from sympy import Symbol

from flagquantum import Circuit
from flagquantum.algorithms.primitives.black_box_select import PackedSelectDeclaration
from flagquantum.algorithms.primitives.select_oracle import SelectOracle
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.value_math import SymbolicInt
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from flagquantum.simulation.unitary import get_unitary
from tests._reference_dependencies import unavailable_reference

try:
    import cirq
    import numpy as np

    import_module("qualtran")
    import_module("kickmix")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_empty_system_packed_select_keeps_selected_relative_phase(
    source_module: tuple[Any, Any],
) -> None:
    from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU

    @dataclass(frozen=True)
    class EmptySystem(SelectOracle):
        @property
        def control_registers(self) -> tuple[Register, ...]:
            return ()

        @property
        def selection_registers(self) -> tuple[Register, ...]:
            return (Register("selection", opaque_register_type(1)),)

        @property
        def target_registers(self) -> tuple[Register, ...]:
            return (Register("target", opaque_register_type(0)),)

    child = SelectPauliLCU(
        1,
        0,
        tuple(cirq.DensePauliString("", coefficient=sign) for sign in (1, -1)),
    )
    original = source_module[0].BlackBoxSelect(child)
    expected = torch.as_tensor(original.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(
        expected, torch.diag(torch.tensor([1, -1], dtype=torch.complex128))
    )
    emitter = import_register_emitter(child)
    assert emitter.requirements() == (0, 0)
    packed = PackedSelectDeclaration(EmptySystem()).bind(emitter)
    assert packed.system_bitsize == 0
    for adjoint in (False, True):
        circuit = Circuit(1, dtype=torch.complex128).append(
            packed.make_on(selection=(0,), target=(), adjoint=adjoint)
        )
        torch.testing.assert_close(get_unitary(circuit), expected, atol=2e-12, rtol=0)
        ir = circuit.to_ir()
        assert CircuitIR.from_json(ir.to_json()) == ir
        assert Counter(gate.name for gate in ir.instructions) == {"x": 2, "z": 1}


@dataclass(frozen=True)
class GroupedDeclaration(SelectOracle):
    address_width: SymbolicInt
    target_width: SymbolicInt

    @property
    def control_registers(self) -> tuple[Register, ...]:
        return ()

    @property
    def selection_registers(self) -> tuple[Register, ...]:
        return (
            Register("row", quantum_bit_type()),
            Register("column", opaque_register_type(self.address_width)),
        )

    @property
    def target_registers(self) -> tuple[Register, ...]:
        return (Register("data", opaque_register_type(self.target_width)),)


@pytest.fixture(scope="module")
def source_module() -> tuple[Any, Any]:
    import kickmix
    from qualtran.bloqs.multiplexers import black_box_select, select_base

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(black_box_select.__file__).parent
    for name, digest in (
        ("black_box_select.py", "8cb8d402b17f17952aa3f06b3fca8ed4cc048239"),
        ("black_box_select_test.py", "6bd3dd68d3adef743a4ae6990180951c40d928d0"),
        ("select_base.py", "dc72bbc1a41b7cf499b1273f1269984289c48672"),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return black_box_select, select_base


@pytest.fixture(scope="module")
def external_select() -> Any:
    from qualtran import QAny, QBit, Register
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.multiplexers.select_base import SelectOracle as SourceOracle

    @frozen
    class GroupedSelect(SourceOracle):  # type: ignore[misc]
        address_width: Any
        target_width: Any

        @property
        def control_registers(self) -> tuple[Any, ...]:
            return ()

        @property
        def selection_registers(self) -> tuple[Any, ...]:
            return (
                Register("row", QBit()),
                Register("column", QAny(self.address_width)),
            )

        @property
        def target_registers(self) -> tuple[Any, ...]:
            return (Register("data", QAny(self.target_width)),)

        def build_composite_bloq(
            self, bb: Any, row: Any, column: Any, data: Any
        ) -> dict[str, Any]:
            column_bits = bb.split(column)
            targets = list(bb.split(data))
            row, targets[0] = bb.add(CNOT(), ctrl=row, target=targets[0])
            for index, control in enumerate(column_bits):
                target = index % len(targets)
                column_bits[index], targets[target] = bb.add(
                    CNOT(), ctrl=control, target=targets[target]
                )
            return {
                "row": row,
                "column": bb.join(column_bits),
                "data": bb.join(targets),
            }

    return GroupedSelect


@pytest.mark.parametrize("widths", [(1, 1), (1, 2), (2, 2)])
def test_symbol_resolved_grouped_oracle_replaces_black_box_select(
    source_module: tuple[Any, Any], external_select: Any, widths: tuple[int, int]
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    address_width = Symbol("n", integer=True, positive=True)
    target_width = Symbol("m", integer=True, positive=True)
    declaration = external_select(address_width, target_width)
    packed_declaration = PackedSelectDeclaration(
        GroupedDeclaration(address_width, target_width)
    )
    assert tuple(reg.name for reg in packed_declaration.signature) == (
        "selection",
        "system",
    )
    with pytest.raises(ValueError, match="concrete positive shapes"):
        import_register_emitter(declaration)

    source = evolve(declaration, address_width=widths[1], target_width=widths[0])
    packed = PackedSelectDeclaration(GroupedDeclaration(widths[1], widths[0]))
    source_emitter = import_register_emitter(source)
    native = packed.bind(source_emitter)
    assert source_emitter.requirements() == (0, 0)
    source_box = source_module[0].BlackBoxSelect(source)
    boxed_emitter = import_register_emitter(source_box)
    logical_width = widths[1] + 1 + widths[0]
    assert [
        (register.name, register.total_bits(), register.side.name)
        for register in packed.signature
    ] == [
        (register.name, register.total_bits(), register.side.name)
        for register in source_box.signature
    ]
    assert boxed_emitter.requirements() == (0, 0)

    for adjoint in (False, True):
        direction = int(adjoint)
        expected = boxed_emitter.snapshots[direction]
        circuit = Circuit(logical_width).append(
            native.make_on(
                selection=tuple(range(widths[1] + 1)),
                target=tuple(range(widths[1] + 1, logical_width)),
                adjoint=adjoint,
            )
        )
        actual = circuit.to_ir()
        assert isinstance(actual, CircuitIR)
        assert CircuitIR.from_json(actual.to_json()).to_json() == actual.to_json()
        assert Counter(g.name for g in actual.instructions) == Counter(
            g.name for g in expected.instructions
        )
        assert actual.instructions == expected.instructions
        torch.testing.assert_close(
            get_unitary(actual, dtype=torch.complex128),
            get_unitary(expected, dtype=torch.complex128),
            atol=1e-12,
            rtol=0,
        )
        source_cost = get_cost_value(source_box, QECGatesCost())
        assert (
            source_cost.measurement == 0
            and source_cost.toffoli == 0
            and source_cost.clifford == widths[1] + 1
        )
        assert actual.n_wires == get_cost_value(source_box, QubitCount())
        assert sum(g.name == "cx" for g in actual.instructions) == widths[1] + 1

        for basis in range(1 << logical_width):
            state = torch.zeros(1 << logical_width, dtype=torch.complex128)
            state[basis] = 1
            result = get_unitary(actual, dtype=torch.complex128) @ state
            address = basis >> widths[0]
            data = basis & ((1 << widths[0]) - 1)
            row = address >> widths[1]
            column = address & ((1 << widths[1]) - 1)
            changed = data ^ (row << (widths[0] - 1))
            for index in range(widths[1]):
                control = column >> (widths[1] - index - 1) & 1
                changed ^= control << (widths[0] - index % widths[0] - 1)
            output = (address << widths[0]) | changed
            assert abs(result[output] - 1) < 1e-12
            assert torch.count_nonzero(result).item() == 1


def test_control_metadata_is_preserved_but_locked_packing_defect_is_explicit(
    source_module: tuple[Any, Any],
) -> None:
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
        SelectHubbard,
    )

    source = source_module[0].BlackBoxSelect(SelectHubbard(2, 2, control_val=1))

    @dataclass(frozen=True)
    class ControlledDeclaration(GroupedDeclaration):
        @property
        def control_registers(self) -> tuple[Register, ...]:
            return (Register("control", quantum_bit_type()),)

    declaration = PackedSelectDeclaration(ControlledDeclaration(1, 1))
    assert tuple(reg.name for reg in declaration.control_registers) == ("control",)
    assert tuple(reg.name for reg in source.signature) == ("selection", "system")
    with pytest.raises(ValueError, match="uncontrolled child"):
        declaration.bind(import_register_emitter(source.select))


def test_original_select_oracle_abstract_protocol_and_symbolic_metadata(
    source_module: tuple[Any, Any], external_select: Any
) -> None:
    from qualtran import QAny, QBit, Register

    source_base = source_module[1]
    assert source_base._SELECT_ORACLE_DOC.bloq_cls is source_base.SelectOracle
    assert source_base._SELECT_ORACLE_DOC.import_line == (
        "from qualtran.bloqs.multiplexers.select_base import SelectOracle"
    )
    assert source_base.SelectOracle.__abstractmethods__ == frozenset(
        {"control_registers", "selection_registers", "target_registers"}
    )
    symbolic = external_select(Symbol("n", integer=True, positive=True), 2)
    assert tuple(symbolic.signature) == (
        *symbolic.control_registers,
        *symbolic.selection_registers,
        *symbolic.target_registers,
    )
    assert symbolic.selection_registers == (
        Register("row", QBit()),
        Register("column", QAny(Symbol("n", integer=True, positive=True))),
    )
    assert tuple(register.name for register in symbolic.signature) == (
        "row",
        "column",
        "data",
    )


def test_native_array_declaration_binds_non_self_adjoint_external_child(
    source_module: tuple[Any, Any],
) -> None:
    from qualtran import QAny
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.multiplexers.select_base import SelectOracle as SourceOracle

    @frozen
    class ArraySelect(SourceOracle):  # type: ignore[misc]
        @property
        def control_registers(self) -> tuple[Any, ...]:
            return ()

        @property
        def selection_registers(self) -> tuple[Any, ...]:
            return (SourceRegister("address", QAny(1), shape=(2,)),)

        @property
        def target_registers(self) -> tuple[Any, ...]:
            return (SourceRegister("data", QAny(1), shape=(2,)),)

        def decompose_from_registers(self, context: Any, **quregs: Any) -> Any:
            address = np.asarray(quregs["address"]).reshape(-1)
            data = np.asarray(quregs["data"]).reshape(-1)
            yield cirq.CNOT(address[0], data[0])
            yield cirq.S(data[1])

    @dataclass(frozen=True)
    class ArrayDeclaration(SelectOracle):
        @property
        def control_registers(self) -> tuple[Register, ...]:
            return ()

        @property
        def selection_registers(self) -> tuple[Register, ...]:
            return (Register("address", quantum_bit_type(), shape=(2,)),)

        @property
        def target_registers(self) -> tuple[Register, ...]:
            return (Register("data", quantum_bit_type(), shape=(2,)),)

    source = ArraySelect()
    declaration = ArrayDeclaration()
    assert declaration.selection_registers[0].shape == (2,)
    packed = PackedSelectDeclaration(declaration).bind(import_register_emitter(source))
    expected = import_register_emitter(source_module[0].BlackBoxSelect(source))
    matrices = []
    for adjoint in (False, True):
        ir = (
            Circuit(4)
            .append(packed.make_on(selection=(0, 1), target=(2, 3), adjoint=adjoint))
            .to_ir()
        )
        restored = CircuitIR.from_json(ir.to_json())
        assert restored.instructions == expected.snapshots[int(adjoint)].instructions
        matrix = get_unitary(restored, dtype=torch.complex128)
        torch.testing.assert_close(
            matrix,
            get_unitary(expected.snapshots[int(adjoint)], dtype=torch.complex128),
            atol=1e-12,
            rtol=0,
        )
        matrices.append(matrix)
    assert not torch.equal(matrices[0], matrices[1])
    torch.testing.assert_close(matrices[1], matrices[0].mH, atol=1e-12, rtol=0)


def test_native_packing_rejects_directed_and_classical_ports() -> None:
    from flagquantum.core.ir.program.registers import RegisterSide, RegisterType

    @dataclass(frozen=True)
    class DirectedDeclaration(GroupedDeclaration):
        side: RegisterSide = RegisterSide.RIGHT
        classical: bool = False

        @property
        def target_registers(self) -> tuple[Register, ...]:
            dtype = (
                RegisterType("classical", 0, 1)
                if self.classical
                else quantum_bit_type()
            )
            return (Register("data", dtype, side=self.side),)

    for side, classical in ((RegisterSide.RIGHT, False), (RegisterSide.THRU, True)):
        with pytest.raises(ValueError, match="quantum THRU"):
            PackedSelectDeclaration(DirectedDeclaration(1, 1, side, classical)).bind(
                lambda *_: ()
            )
    with pytest.raises(ValueError, match="concrete width for column"):
        PackedSelectDeclaration(GroupedDeclaration(Symbol("n", positive=True), 1)).bind(
            lambda *_: ()
        )
