"""Literal definition matching retains phases, feedback and helper sharing."""

from dataclasses import replace

import pytest

from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.component_graph import _namespace_helpers
from flagquantum.compiler.registers.control.conjunction_program import (
    and_component_program,
    conjunction_program,
)
from flagquantum.compiler.registers.primitives.relative_swap_component import (
    relative_swap_program,
)
from flagquantum.core.ir.program import ProgramRecord

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("erase", [False, True])
@pytest.mark.parametrize("values", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_exact_and_template_ignores_names_but_retains_polarities(
    values: tuple[int, int], erase: bool
) -> None:
    template = and_component_program(values, name="and", uncompute=erase)
    imported = _namespace_helpers(template, "parent")
    imported = replace(imported, functions=tuple(reversed(imported.functions)))
    assert matches_component_definition(imported, "parent_and", template)
    for other in ((0, 0), (0, 1), (1, 0), (1, 1)):
        assert matches_component_definition(
            imported,
            "parent_and",
            and_component_program(other, name="different", uncompute=erase),
        ) == (values == other)


def test_reachable_calls_preserve_sharing_ports_operands_and_phase() -> None:
    template = relative_swap_program(2)
    imported = _namespace_helpers(template, "lookup")
    assert matches_component_definition(imported, "lookup_main", template)
    primitive = next(
        function for function in imported.functions if function.name == "lookup_t"
    )
    block = primitive.blocks[0]
    operations = tuple(
        replace(operation, opcode="tdg") if operation.opcode == "t" else operation
        for operation in block.operations
    )
    changed = replace(primitive, blocks=(replace(block, operations=operations),))
    mutated = replace(
        imported,
        functions=tuple(
            changed if function == primitive else function
            for function in imported.functions
        ),
    )
    assert not matches_component_definition(mutated, "lookup_main", template)
    assert not matches_component_definition(relative_swap_program(3), "main", template)
    unrelated = conjunction_program((1, 1, 1), name="unused")
    assert matches_component_definition(
        replace(imported, functions=(*imported.functions, *unrelated.functions)),
        "lookup_main",
        template,
    )


def test_missing_and_invalid_definitions_are_refused() -> None:
    template = conjunction_program((1, 0, 1))
    with pytest.raises(ValueError, match="Unknown component"):
        matches_component_definition(template, "missing", template)
    with pytest.raises(ValueError, match="entry definition"):
        matches_component_definition(template, "multi_and", ProgramRecord(()))
    with pytest.raises(ValueError, match="Invalid component"):
        matches_component_definition(
            replace(template, functions=(*template.functions, template.functions[0])),
            "multi_and",
            template,
        )
