"""Complete packed PREPARE source wrappers retain original dependency calls."""

from typing import Any

import pytest

from flagquantum.algorithms.primitives.alias_preparation import (
    alias_preparation_program,
)
from flagquantum.algorithms.primitives.black_box_prepare import PackedPrepareDeclaration
from flagquantum.algorithms.primitives.hubbard_prepare import PrepareHubbard
from flagquantum.algorithms.primitives.hubbard_prepare_component import (
    hubbard_prepare_program,
)
from flagquantum.algorithms.primitives.packed_prepare_component import (
    packed_prepare_program,
)
from flagquantum.algorithms.primitives.symbolic_alias_preparation import (
    AliasPreparationShape,
)
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_alias_stored_graph import ising_source, native_alias
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_hubbard_stored_graph import source_terminals

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "kind",
    [
        "hubbard2",
        "hubbard4",
        "dense",
        "sparse",
        "ising",
        "symbolic",
        "sparse_list",
        "sparse_symbolic",
    ],
)
@pytest.mark.parametrize("adjoint", [False, True])
def test_complete_packed_wrapper_and_recursive_children(
    kind: str, adjoint: bool
) -> None:
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.prepare_hubbard import (
        PrepareHubbard as SourceHubbard,
    )
    from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
    from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
        _sparse_state_prep_alias,
        _sparse_state_prep_alias_from_list,
        _sparse_state_prep_alias_symb,
        _state_prep_alias,
        _state_prep_alias_symb,
    )

    oracle: Any
    if kind.startswith("hubbard"):
        dimension = int(kind[-1])
        source = SourceHubbard(dimension, dimension, 1.0, 4.0)
        oracle = PrepareHubbard(dimension, 1.0, 4.0)
        forward = hubbard_prepare_program(oracle)
        inverse = hubbard_prepare_program(oracle, adjoint=True)
    else:
        source = {
            "dense": _state_prep_alias.make,
            "sparse": _sparse_state_prep_alias.make,
            "ising": ising_source,
            "symbolic": _state_prep_alias_symb.make,
            "sparse_list": _sparse_state_prep_alias_from_list.make,
            "sparse_symbolic": _sparse_state_prep_alias_symb.make,
        }[kind]()
        table = native_alias(source)
        oracle = (
            table
            if isinstance(table, AliasPreparationShape)
            else AliasPreparationShape.from_table(table)
        )
        forward = alias_preparation_program(table)
        inverse = alias_preparation_program(table, adjoint=True)
    record = packed_prepare_program(
        PackedPrepareDeclaration(oracle), forward, inverse=inverse, adjoint=adjoint
    )
    assert verify_program_record(record) == ()
    assert len(record.functions[0].blocks[0].operations) == 1
    expected = BlackBoxPrepare(source)
    compare_layers(
        record, expected.adjoint() if adjoint else expected, source_terminals(record)
    )


@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("kind", ["hubbard", "dense", "sparse", "ising"])
def test_packed_program_copy_append_and_serialization_preserve_child_execution(
    kind: str, adjoint: bool
) -> None:
    from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
        _sparse_state_prep_alias,
        _state_prep_alias,
    )

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum import Circuit
    from flagquantum.algorithms.primitives.alias_preparation import (
        alias_preparation_requirements,
    )
    from flagquantum.algorithms.primitives.alias_sampling import AliasSamplingTable
    from flagquantum.compiler.registers.component_graph import copy_component_program
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.core.ir import CircuitIR

    if kind == "hubbard":
        oracle: Any = PrepareHubbard(2, 1.0, 4.0)
        forward = hubbard_prepare_program(oracle)
        inverse = hubbard_prepare_program(oracle, adjoint=True)
        work, _ = oracle.requirements()
    else:
        source = {
            "dense": _state_prep_alias.make,
            "sparse": _sparse_state_prep_alias.make,
            "ising": ising_source,
        }[kind]()
        table = native_alias(source)
        assert isinstance(table, AliasSamplingTable)
        oracle = AliasPreparationShape.from_table(table)
        forward = alias_preparation_program(table)
        inverse = alias_preparation_program(table, adjoint=True)
        work, _ = alias_preparation_requirements(table)
    declaration = PackedPrepareDeclaration(oracle)
    graph = packed_prepare_program(
        declaration, forward, inverse=inverse, adjoint=adjoint
    )
    logical = int(declaration.selection_width + declaration.junk_width)
    clean = tuple(range(logical, logical + work))
    packed_inputs = (
        tuple(range(int(declaration.selection_width))),
        tuple(range(int(declaration.selection_width), logical)),
    )
    unpacked = []
    cursor = 0
    for reg in oracle.signature:
        size = int(reg.total_qubits())
        unpacked.append(tuple(range(cursor, cursor + size)))
        cursor += size
    expected = bind_register_program(
        inverse if adjoint else forward, tuple(unpacked), clean=clean
    ).circuit
    for candidate in (graph, copy_component_program(graph)):
        bound = bind_register_program(candidate, packed_inputs, clean=clean)
        assert bound.circuit == expected
        restored = CircuitIR.from_json(bound.circuit.to_json())
        assert restored == expected
        appended = (
            Circuit(expected.n_wires)
            .append(FieldOperation(expected.n_wires, restored.instructions))
            .to_ir()
        )
        assert appended.instructions == expected.instructions


@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("widths", [(0, 0), (0, 5), (1, 0), (1, 5)])
def test_zero_groups_and_original_dummy_keep_declared_children(
    widths: tuple[int, int], adjoint: bool
) -> None:
    from flagquantum.algorithms.primitives.prepare_oracle import PrepareOracle
    from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
    from flagquantum.core.ir.program.register_types import (
        opaque_register_type,
        quantum_bit_type,
    )
    from flagquantum.core.ir.program.registers import Register
    from flagquantum.core.ir.program.signatures import Signature
    from tests.unit.test_component_graph import _native_trace

    class Oracle(PrepareOracle):
        @property
        def selection_registers(self) -> tuple[Register, ...]:
            return (
                Register(
                    "z",
                    (
                        quantum_bit_type()
                        if widths[0] == 1
                        else opaque_register_type(widths[0])
                    ),
                ),
            )

        @property
        def junk_registers(self) -> tuple[Register, ...]:
            return (Register("a", opaque_register_type(widths[1])),)

        @property
        def signature(self) -> Signature:
            return Signature.build(z=widths[0], a=widths[1])

    oracle = Oracle()
    child = ProgramRecord(
        (FunctionRecord("main", (), registers=tuple(oracle.signature)),)
    )
    record = packed_prepare_program(
        PackedPrepareDeclaration(oracle), child, inverse=child, adjoint=adjoint
    )
    assert verify_program_record(record) == ()
    assert len(_native_trace(record)) == int(bool(widths[0]))
    if widths[0]:
        with pytest.raises(ValueError, match="exact inverse"):
            packed_prepare_program(
                PackedPrepareDeclaration(oracle), child, adjoint=True
            )


def test_body_signature_mismatch_rejects_before_binding() -> None:
    from flagquantum.core.ir.program import FunctionRecord, ProgramRecord

    prepare = PrepareHubbard(2, 1.0, 4.0)
    with pytest.raises(ValueError, match="declared signature"):
        packed_prepare_program(
            PackedPrepareDeclaration(prepare),
            ProgramRecord((FunctionRecord("main", (), registers=()),)),
        )
