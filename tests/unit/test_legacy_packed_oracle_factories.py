"""Packed constructor examples reuse native graphs and retain source refusals."""

from typing import Any

import pytest

from flagquantum.algorithms.primitives.black_box_prepare import PackedPrepareDeclaration
from flagquantum.algorithms.primitives.hubbard_prepare import PrepareHubbard
from flagquantum.algorithms.primitives.hubbard_prepare_component import (
    hubbard_prepare_program,
)
from flagquantum.algorithms.primitives.packed_prepare_component import (
    packed_prepare_program,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.ecosystem._packed_oracle_component_library import (
    packed_oracle_library_factories,
)
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_hubbard_stored_graph import source_terminals
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "kind,inverse",
    tuple(
        (kind, inverse) for kind in ("prepare", "select") for inverse in (False, True)
    ),
)
def test_original_packed_examples_and_source_autopartition_boundary(
    kind: str, inverse: bool
) -> None:
    from qualtran.bloqs.multiplexers.black_box_select import _black_box_select
    from qualtran.bloqs.state_preparation.black_box_prepare import _black_box_prepare

    source = (_black_box_prepare if kind == "prepare" else _black_box_select).make()
    if inverse:
        source = source.adjoint()
    record = resolve_component_program(
        _library(source, 0), 0, packed_oracle_library_factories()
    )
    assert not verify_program_record(record)
    assert (
        record.functions[0].registers
        == _library(source, 0).entries[0].declaration.registers
    )
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    if kind == "prepare":
        compare_layers(record, source, source_terminals(record))
        oracle = PrepareHubbard(2, 1.0, 4.0)
        reference = packed_prepare_program(
            PackedPrepareDeclaration(oracle),
            hubbard_prepare_program(oracle),
            inverse=hubbard_prepare_program(oracle, adjoint=True),
            adjoint=inverse,
        )
        # Full numerical/channel evidence already exists for this exact graph.
        # Prove the constructor selects that same emitted graph and lifetime.
        a = bind_register_program(
            record, (tuple(range(8)), (8, 9)), clean=tuple(range(10, 18))
        )
        b = bind_register_program(
            reference, (tuple(range(8)), (8, 9)), clean=tuple(range(10, 18))
        )
        assert a.circuit.to_json() == b.circuit.to_json()
        assert a.events == b.events
    else:
        from flagquantum.algorithms.primitives.hubbard_select import SelectHubbard
        from flagquantum.algorithms.primitives.hubbard_select_component import (
            hubbard_select_program,
        )

        compare_layers(
            record,
            source,
            source_terminals(record)
            | {f.name for f in record.functions if f.registers == ()},
        )
        select = SelectHubbard(2)
        ports = []
        cursor = 0
        for _, width in select.layout:
            ports.append(tuple(range(cursor, cursor + width)))
            cursor += width
        actual = bind_register_program(
            record, (tuple(range(8)), tuple(range(8, 16))), clean=(16, 17, 18, 19)
        )
        expected = bind_register_program(
            hubbard_select_program(select, adjoint=inverse),
            tuple(ports),
            clean=(16, 17, 18, 19),
        )
        assert actual.circuit == expected.circuit
        # Packing adds logical view operations. Physical lease offsets and
        # ownership remain identical beneath that explicit wrapper.
        assert [
            (e.kind, e.wires, e.dirty, e.instruction_offset) for e in actual.events
        ] == [(e.kind, e.wires, e.dirty, e.instruction_offset) for e in expected.events]
    # Original max_depth=1 fails before native decoding. Preserve that cause.
    from qualtran.serialization.bloq import bloqs_to_proto

    with pytest.raises(ValueError, match="inhomogeneous|sequence"):
        bloqs_to_proto(source, max_depth=1)


@pytest.mark.parametrize("inverse", (False, True))
def test_source_free_packed_prepare_resolution_and_append(
    inverse: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from qualtran.bloqs.state_preparation.black_box_prepare import _black_box_prepare

    from flagquantum import Circuit
    from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation

    source = _black_box_prepare.make()
    if inverse:
        source = source.adjoint()
    library = _library(source, 0)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        record = resolve_component_program(
            library, 0, packed_oracle_library_factories()
        )
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        bound = bind_register_program(
            record, (tuple(range(8)), (8, 9)), clean=tuple(range(10, 18))
        )
        circuit = Circuit(bound.circuit.n_wires)
        circuit.append(
            _BoundBlockOperation(bound.circuit.instructions, bound.circuit.n_wires)
        )
        assert (
            circuit.to_ir().to_dict()["instructions"]
            == bound.circuit.to_dict()["instructions"]
        )


@pytest.mark.parametrize("inverse", (False, True))
def test_select_explicit_native_owner_full_complex_operator_and_array_roles(
    inverse: bool,
) -> None:
    from collections.abc import Mapping

    import attrs

    np = pytest.importorskip("numpy")
    import torch
    from qualtran import QBit
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.basic_gates import CNOT, ZPowGate
    from qualtran.bloqs.multiplexers.black_box_select import BlackBoxSelect
    from qualtran.bloqs.multiplexers.select_base import SelectOracle

    from flagquantum.compiler.formats.legacy_component_resolution import (
        ResolvedArgument,
    )
    from flagquantum.compiler.formats.legacy_word_factories import _rename_program
    from flagquantum.compiler.registers.component_adjoint import adjoint_component_calls
    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.compiler.registers.component_gate_function import bit_gate_function
    from flagquantum.compiler.registers.primitives.pauli_rotation_component import (
        z_power_function,
    )
    from flagquantum.core.ir import CircuitIR, Instruction
    from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
    from flagquantum.core.ir.program.register_types import quantum_bit_type
    from flagquantum.core.ir.program.registers import Register
    from flagquantum.core.ir.program.signatures import Signature
    from tests.unit.test_multi_control_component import atomic_leaves
    from tests.unit.test_multiplexed_source_binding import logical_operator

    class ArraySelect:
        @property
        def control_registers(self) -> tuple[Any, ...]:
            return ()

        @property
        def selection_registers(self) -> tuple[Any, ...]:
            return (SourceRegister("address", QBit()),)

        @property
        def target_registers(self) -> tuple[Any, ...]:
            return (SourceRegister("data", QBit(), shape=(2,)),)

        def build_composite_bloq(self, bb: Any, **soqs: Any) -> dict[str, Any]:
            data = np.asarray(soqs["data"]).copy()
            address, data[0] = bb.add(CNOT(), ctrl=soqs["address"], target=data[0])
            data[1] = bb.add(ZPowGate(0.37), q=data[1])
            return {"address": address, "data": data}

    bit = quantum_bit_type()
    selection = (Register("address", bit),)
    target = (Register("data", bit, (2,)),)
    cnot = ProgramRecord(
        (
            bit_gate_function(
                "cx",
                (Register("ctrl", bit), Register("target", bit)),
                CircuitIR(2, (Instruction("cx", (0, 1)),)),
            ),
        )
    )
    phase = ProgramRecord((z_power_function("phase", 0.37),))
    reverse_phase = ProgramRecord((z_power_function("phase", -0.37),))
    builder = ComponentProgramBuilder(Signature((*selection, *target)))
    data = np.asarray(builder.inputs["data"]).copy()
    changed = builder.call_program(
        cnot, {"ctrl": builder.inputs["address"], "target": data[0]}, entry="cx"
    )
    data[0] = changed["target"]
    data[1] = builder.call_program(phase, {"q": data[1]}, entry="phase")["q"]
    forward = builder.finalize({"address": changed["ctrl"], "data": data})
    backward = adjoint_component_calls(forward, ((cnot, cnot), (phase, reverse_phase)))
    known: set[FunctionRecord] = set()

    def child_factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        assert not arguments and registers == (*selection, *target)
        record = _rename_program(forward, name)
        known.add(record.functions[0])
        return record

    def owner(
        child: FunctionRecord,
    ) -> tuple[
        tuple[Register, ...], tuple[Register, ...], tuple[ProgramRecord, ProgramRecord]
    ]:
        assert child in known
        return selection, target, (forward, backward)

    # The optional SDK base stays at the foreign test boundary.
    source_type = attrs.frozen(
        type("ArraySelect", (ArraySelect, SelectOracle), {"__module__": __name__})
    )
    source = BlackBoxSelect(source_type())
    if inverse:
        source = source.adjoint()
    factories = packed_oracle_library_factories(select_context=owner)
    factories[__name__ + ".ArraySelect"] = child_factory
    library = _library(source, 0)
    # The source codec records the local class's name without its local scope.
    child_identifier = next(
        entry.declaration.name
        for entry in library.entries
        if entry.declaration.name.endswith(".ArraySelect")
    )
    factories[child_identifier] = child_factory
    record = resolve_component_program(library, 0, factories)
    assert not verify_program_record(record)
    compare_layers(record, source, atomic_leaves(record))
    bound = bind_register_program(record, ((0,), (1, 2)))
    torch.testing.assert_close(
        logical_operator(bound.circuit, 3),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=8e-12,
        rtol=0,
    )
    assert not bound.events


def test_original_controlled_select_partition_failure_is_not_repaired() -> None:
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
        SelectHubbard,
    )
    from qualtran.bloqs.multiplexers.black_box_select import BlackBoxSelect

    from flagquantum.errors import SerializationError

    source = BlackBoxSelect(SelectHubbard(2, 2, control_val=1))
    with pytest.raises(ValueError, match="partitions"):
        source.decompose_bloq()
    with pytest.raises(SerializationError, match="child input names"):
        resolve_component_program(
            _library(source, 0), 0, packed_oracle_library_factories()
        )
    with pytest.raises(NotImplementedError, match="control_val=0"):
        SelectHubbard(2, 2, control_val=0)
