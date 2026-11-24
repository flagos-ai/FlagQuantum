"""Serialized blocked XOR constructors retain bank ownership and inverse calls."""

import sys
from typing import Any

import pytest

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_select_swap_factories import (
    select_swap_factories,
)
from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.registers.lookup.select_swap_qrom_component import (
    select_swap_qrom_program,
)
from flagquantum.compiler.registers.lookup.source_qroam_clean import QroamCleanTable
from flagquantum.core.ir.program import FunctionRecord, verify_program_record
from flagquantum.errors import SerializationError
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_select_swap_qrom_component import (
    pinned_select_swap as pinned_select_swap,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.ftqc_reference,
    pytest.mark.usefixtures("pinned_select_swap"),
]


def _source(example: str, dirty: bool) -> Any:
    import attrs
    from qualtran.bloqs.data_loading import select_swap_qrom

    return attrs.evolve(
        getattr(select_swap_qrom, example).make(), use_dirty_ancilla=dirty
    )


@pytest.mark.parametrize("example", ("_qroam_multi_data", "_qroam_multi_dim"))
@pytest.mark.parametrize("dirty", (False, True))
@pytest.mark.parametrize("inverse", (False, True))
def test_original_concrete_constructor_metadata_and_exact_adjoint_dependencies(
    example: str, dirty: bool, inverse: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flagquantum.compiler.registers.lookup.qroam_resource_model import (
        source_select_swap_qrom_cost,
    )
    from tests.unit.test_first_quantization_resource_model import assert_source_model

    source = _source(example, dirty)
    root = source.adjoint() if inverse else source
    libraries = (_library(root, 0), _library(root, 1))
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        records = tuple(
            resolve_component_program(library, 0, select_swap_factories())
            for library in libraries
        )
    record = records[0]
    from flagquantum.compiler.registers.component_definition_match import (
        matches_component_definition,
    )

    # Original Adjoint serialization assigns different IDs to its unused
    # forward recipe as depth changes. Compare every reachable body literally.
    assert verify_program_record(records[1]) == ()
    assert matches_component_definition(records[1], "main", record)
    assert verify_program_record(record) == ()
    table = QroamCleanTable(
        tuple(source.data_shape),
        tuple(source.log_block_sizes),
        tuple(tuple(int(v) for v in data.flat) for data in source.data),
    )
    expected = _rename_program(
        select_swap_qrom_program(
            table,
            tuple(source.target_bitsizes),
            selection_widths=tuple(source.selection_bitsizes),
            num_controls=int(source.num_controls),
            dirty=dirty,
            adjoint=inverse,
        ),
        "main",
    )
    # A resolved Adjoint retains the separately declared forward recipe module.
    assert record.functions[0] == expected.functions[0]
    assert set(expected.functions) <= set(record.functions)
    assert_source_model(
        source_select_swap_qrom_cost(
            table,
            int(source.num_controls),
            tuple(source.selection_bitsizes),
            tuple(source.target_bitsizes),
            dirty=dirty,
        ),
        root,
    )


@pytest.mark.parametrize(
    "example",
    (
        "_qroam_symb_dirty_1d",
        "_qroam_symb_dirty_2d",
        "_qroam_symb_clean_1d",
        "_qroam_symb_clean_2d",
    ),
)
def test_four_original_symbolic_serialization_boundaries_remain_unmodified(
    example: str,
) -> None:
    from qualtran.bloqs.data_loading import select_swap_qrom

    source = getattr(select_swap_qrom, example).make()
    with pytest.raises(ValueError, match="allow_pickle=False"):
        _library(source, 0)


def test_unknown_adjoint_context_delegates_to_existing_owner() -> None:
    calls: list[tuple[str, Any, Any]] = []

    def owner(name: str, registers: Any, arguments: Any) -> FunctionRecord:
        calls.append((name, registers, arguments))
        return FunctionRecord(name, (), registers=registers)

    factory = select_swap_factories(adjoint_factory=owner)[
        "qualtran._infra.adjoint.Adjoint"
    ]
    result = factory(
        "foreign", (), {"subbloq": FunctionRecord("unowned", (), registers=())}
    )
    assert isinstance(result, FunctionRecord)
    assert result.name == "foreign" and len(calls) == 1
    with pytest.raises(SerializationError, match="context"):
        select_swap_factories()["qualtran._infra.adjoint.Adjoint"](
            "foreign", (), {"subbloq": FunctionRecord("unowned", (), registers=())}
        )
