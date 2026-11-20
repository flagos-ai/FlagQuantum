"""Resolve complete compound fixtures while preserving source protocol failures."""

from collections.abc import Mapping
from typing import Any

import pytest

from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register
from flagquantum.simulation.classical_bit_binding import (
    allocate_classical_value,
    free_classical_value,
    split_classical_value,
)
from flagquantum.simulation.component_classical_state import ComponentClassicalState
from flagquantum.simulation.integer_encoding import integer_from_bits, integer_to_bits
from tests.unit.composition_fixtures import Mode, composition_fixture
from tests.unit.test_composition_fixtures import (
    assert_composition_port_actions_and_projection,
)
from tests.unit.test_subtraction_foreign_file import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def original_fixture(mode: Mode) -> Any:
    from qualtran.bloqs.for_testing.with_decomposition import (
        TestIndependentParallelCombo,
        TestParallelCombo,
        TestSerialCombo,
    )

    return {
        "serial": TestSerialCombo,
        "parallel": TestParallelCombo,
        "independent": TestIndependentParallelCombo,
    }[mode]()


@pytest.mark.parametrize("mode", ("serial", "parallel", "independent"))
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_all_compound_source_libraries_preserve_typed_calls_and_execution(
    mode: Mode, depth: int
) -> None:
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    source = original_fixture(mode)
    proto = bloqs_to_proto(source, max_depth=depth)
    wire = proto.SerializeToString(deterministic=True)
    library = deserialize_component_library(wire)
    assert serialize_component_library(library) == wire
    assert bloqs_from_proto(proto)[0] == source

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        assert name == "main" and not arguments
        result = composition_fixture(mode)
        assert result.functions[0].registers == registers
        return result

    program = resolve_component_program(
        library, 0, {library.entries[0].declaration.name: factory}
    )
    assert program.functions[0].registers == tuple(native_signature(source.signature))
    assert_composition_port_actions_and_projection(mode, program)


@pytest.mark.parametrize("mode", ("serial", "parallel", "independent"))
def test_atomic_classical_refusal_preserves_the_available_assignment_prefix(
    mode: Mode,
) -> None:
    source = original_fixture(mode)
    inputs: dict[str, object] = {} if mode == "independent" else {"reg": 0}
    with pytest.raises(NotImplementedError, match="TestAtom.*classically simulable"):
        source.call_classically(**inputs)
    record = composition_fixture(mode)
    order = tuple(position for position, _, _ in iter_component_calls(record))

    def action(function: FunctionRecord, values: dict[str, object]) -> object:
        if function.blocks[0].operations[0].opcode == "allocate_register":
            return {
                "reg": allocate_classical_value(
                    1, lambda bits: integer_from_bits(bits, 1)
                )
            }
        if not function.name.startswith("atom"):
            assert function.blocks[0].operations[0].opcode == "split_register"
            return {
                "reg": split_classical_value(
                    values["reg"], lambda value: integer_to_bits(value, 3)
                )
            }
        return NotImplemented

    state = ComponentClassicalState(record, order, inputs, action)
    with pytest.raises(NotImplementedError, match="classical decomposition.*atom"):
        state.simulate()
    assert state.last_position == (1 if mode != "serial" else 0)
    assert inputs == ({} if mode == "independent" else {"reg": 0})


def test_independent_fixture_is_outside_clean_free_precondition_in_both_protocols() -> (
    None
):
    from qualtran import QBit
    from qualtran.bloqs.bookkeeping import Free

    source = original_fixture("independent")
    assert source.tensor_contract().item() == 0
    with pytest.raises(ValueError, match="non-zero register"):
        Free(QBit()).on_classical_vals(reg=1)
    with pytest.raises(ValueError, match="nonzero classical register"):
        free_classical_value(1)
    assert Free(QBit()).on_classical_vals(reg=0) == free_classical_value(0) == {}
