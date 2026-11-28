"""Retain complete actual Hubbard walk dependencies without a cleanup claim."""

from importlib import import_module
from typing import Any

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    import_module("numpy")
    import_module("cirq")
    import_module("qualtran")
    import_module("kickmix")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

from flagquantum.core.ir.program import verify_program_record
from flagquantum.ecosystem._hubbard_walk_component_library import (
    capture_hubbard_walk_context,
)
from flagquantum.errors import SerializationError
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_hubbard_stored_graph import source_terminals
from tests.unit.test_legacy_phased_block_factories import scalar_terminals
from tests.unit.test_qubitization_walk_file import (
    complete_frozen_sources as complete_frozen_sources,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("symbolic", (False, True))
def test_actual_hamiltonian_examples_keep_all_four_walk_dependency_graphs(
    symbolic: bool,
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.hamiltonian_simulation import hamiltonian_simulation_by_gqsp

    example = getattr(
        hamiltonian_simulation_by_gqsp,
        "_symbolic_hamsim_by_gqsp" if symbolic else "_hubbard_time_evolution_by_gqsp",
    ).make()
    source = example.walk_operator
    context = capture_hubbard_walk_context(source)
    for controlled in (False, True):
        original = source.get_ctrl_system(CtrlSpec())[0] if controlled else source
        for record, expected in zip(
            context.pair(controlled=controlled),
            (original, original.adjoint()),
            strict=True,
        ):
            assert verify_program_record(record) == ()
            assert all(function.blocks for function in record.functions)
            pairs = compare_layers(
                record, expected, source_terminals(record) | scalar_terminals(record)
            )
            assert len(pairs) == 118
            # The captured source signature excludes reflection temp. Its
            # internal release is literal source behavior, not a clean-work proof.
            assert "temp" not in {
                register.name for register in record.functions[0].registers or ()
            }


@pytest.mark.parametrize("source", (None, object(), 3))
def test_non_hubbard_source_cannot_supply_a_walk_context(source: Any) -> None:
    with pytest.raises(SerializationError, match="original PREPARE and SELECT"):
        capture_hubbard_walk_context(source)


def test_actual_degree_127_hamiltonian_root_and_all_rotation_definitions() -> None:
    from qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp import (
        _hubbard_time_evolution_by_gqsp,
    )

    from flagquantum.compiler.formats.legacy_su2_factories import _su2_factory
    from flagquantum.compiler.registers.component_definition_match import (
        matches_component_definition,
    )
    from flagquantum.compiler.registers.phase.hamiltonian_gqsp_component import (
        hamiltonian_gqsp_component_program,
    )
    from flagquantum.compiler.registers.preparation.hubbard_prepare_component import (
        hubbard_prepare_program,
    )
    from flagquantum.core.ir.program import ProgramRecord
    from flagquantum.ecosystem._generalized_qsp_component_library import (
        generalized_qsp_library_factories,
    )
    from flagquantum.ecosystem._packed_oracle_component_library import _prepare_angles

    source = _hubbard_time_evolution_by_gqsp.make()
    context = capture_hubbard_walk_context(source.walk_operator)
    ordinary = context.pair()
    positive = context.pair(controlled=True, source_control_recipes=True)
    factory = next(
        iter(
            generalized_qsp_library_factories(
                child_owner=lambda _: ordinary, positive_child_owner=lambda _: positive
            ).values()
        )
    )
    # This interoperability scenario captures coefficient payloads at the SDK
    # boundary. Native Simulation derives phases; independent native coefficient
    # numerics retain their existing Hamiltonian model acceptance evidence.
    processor = factory(
        "processor",
        None,
        {
            "U": ordinary[0].functions[0],
            "P": tuple(map(complex, source.gqsp.P)),
            "Q": tuple(map(complex, source.gqsp.Q)),
            "negative_power": source.gqsp.negative_power,
            "precision": source.gqsp.precision,
        },
    )
    assert isinstance(processor, ProgramRecord)
    prepare = source.walk_operator.prepare
    angles = _prepare_angles(prepare.x_dim, prepare.t, prepare.u)
    assert angles is not None
    pair = (
        hubbard_prepare_program(prepare.x_dim, *angles),
        hubbard_prepare_program(prepare.x_dim, *angles, adjoint=True),
    )
    record = hamiltonian_gqsp_component_program(
        processor, *pair, junk_names=tuple(r.name for r in prepare.junk_registers)
    )
    assert verify_program_record(record) == ()
    functions = {function.name: function for function in record.functions}
    processor_name = record.functions[0].blocks[0].operations[2].callee
    assert processor_name is not None
    rotations = {
        op.callee
        for op in functions[processor_name].blocks[0].operations[::2]
        if op.callee is not None
    }
    assert source.degree == 127 and len(rotations) == 255
    pairs = compare_layers(
        record, source, source_terminals(record) | scalar_terminals(record) | rotations
    )
    assert len(pairs) == 563
    for name in rotations:
        original = pairs[name]
        expected = _su2_factory(
            "main",
            None,
            {
                key: float(getattr(original, key))
                for key in ("theta", "phi", "lambd", "global_shift", "eps")
            },
        )
        # Check only this reachable small subtree; verifying the unrelated
        # 2921-function module for each of 255 rotations obscures real failures.
        reachable = {name: functions[name]}
        pending = [name]
        for child in pending:
            for block in functions[child].blocks:
                for op in block.operations:
                    if op.callee is not None and op.callee not in reachable:
                        reachable[op.callee] = functions[op.callee]
                        pending.append(op.callee)
        assert matches_component_definition(
            ProgramRecord(tuple(reachable.values())), name, expected
        )
