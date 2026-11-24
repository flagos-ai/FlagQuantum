"""Clean lookup constructors preserve executable forward and declared erasure."""

from dataclasses import replace
from typing import Any

import pytest

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_qroam_factories import qroam_factories
from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.registers.lookup.qroam_clean_component import (
    qroam_clean_program,
)
from flagquantum.compiler.registers.lookup.qroam_resource_model import (
    source_qroam_measurement_erase_cost,
)
from flagquantum.compiler.registers.lookup.source_qroam_clean import QroamCleanTable
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.errors import SerializationError
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_qroam_clean_component import pinned_source as pinned_source

pytestmark = [
    pytest.mark.unit,
    pytest.mark.ftqc_reference,
    pytest.mark.usefixtures("pinned_source"),
]


def _source(example: str) -> Any:
    from qualtran.bloqs.data_loading import qroam_clean

    return getattr(qroam_clean, example).make()


def _table(source: Any) -> QroamCleanTable:
    return QroamCleanTable(
        tuple(source.data_shape),
        tuple(source.log_block_sizes),
        tuple(tuple(int(value) for value in data.flat) for data in source.data),
    )


@pytest.mark.parametrize(
    "example", ("_qroam_clean_multi_data", "_qroam_clean_multi_dim")
)
def test_original_forward_examples_resolve_complete_native_dependency_module(
    example: str,
) -> None:
    source = _source(example)
    record = resolve_component_program(_library(source, 0), 0, qroam_factories())
    assert verify_program_record(record) == ()
    assert record == resolve_component_program(
        _library(source, 1), 0, qroam_factories()
    )
    expected = _rename_program(
        qroam_clean_program(
            _table(source),
            tuple(source.target_bitsizes),
            selection_widths=tuple(source.selection_bitsizes),
            num_controls=int(source.num_controls),
        ),
        "main",
    )
    assert record == expected
    from flagquantum.compiler.registers.lookup.qroam_resource_model import (
        source_qroam_clean_cost,
    )
    from tests.unit.test_first_quantization_resource_model import assert_source_model

    assert_source_model(
        source_qroam_clean_cost(
            _table(source),
            int(source.num_controls),
            tuple(source.selection_bitsizes),
            tuple(source.target_bitsizes),
        ),
        source,
    )


@pytest.mark.parametrize(
    "example", ("_qroam_clean_multi_data", "_qroam_clean_multi_dim")
)
def test_original_wrapper_source_call_and_bodyless_leaf_remain_distinct(
    example: str,
) -> None:
    from qualtran import DecomposeNotImplementedError
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.compiler._component_trace.resource_model import (
        estimate_component_resources,
    )

    forward = _source(example)
    source = forward.adjoint()
    record = resolve_component_program(_library(source, 0), 0, qroam_factories())
    assert verify_program_record(record) == ()
    root = record.functions[0]
    assert root.registers is not None
    terminal = {f.name for f in record.functions if f.name != root.name}
    compare_layers(record, source, terminal)
    leaf = source.qroam_clean_adjoint_bloq
    declaration = resolve_component_program(_library(leaf, 0), 0, qroam_factories())
    assert not declaration.functions[0].blocks
    with pytest.raises(DecomposeNotImplementedError, match="does not declare"):
        leaf.decompose_bloq()
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(declaration, ())
    models = source_qroam_measurement_erase_cost(
        _table(forward),
        int(forward.num_controls),
        tuple(forward.target_bitsizes),
        erase_logs=tuple(source.log_block_sizes),
    )
    native = (
        estimate_component_resources(models.wrapper).gates.toffoli,
        estimate_component_resources(models.padded_leaf).gates.toffoli,
    )
    original = (
        get_cost_value(source, QECGatesCost()).toffoli,
        get_cost_value(leaf, QECGatesCost()).toffoli,
    )
    assert native == original == ((1, 1) if example.endswith("multi_data") else (7, 9))


def test_wrapper_requires_owned_lookup_recipe_and_exact_signature() -> None:
    factory = qroam_factories()[
        "qualtran.bloqs.data_loading.qroam_clean.QROAMCleanAdjointWrapper"
    ]
    source = _source("_qroam_clean_multi_data")
    function = resolve_component_program(
        _library(source, 0), 0, qroam_factories()
    ).functions[0]
    np = pytest.importorskip("numpy")

    with pytest.raises(SerializationError, match="context"):
        factory(
            "main", None, {"qroam_clean": function, "log_block_sizes": np.array([1])}
        )
    library = _library(source, 0)
    first = library.entries[0]
    wrong = replace(first.declaration, registers=())
    library = replace(
        library, entries=(replace(first, declaration=wrong), *library.entries[1:])
    )
    with pytest.raises(SerializationError, match="signature"):
        resolve_component_program(library, 0, qroam_factories())


def test_serialized_lookup_and_wrapper_compile_without_source_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    source = _source("_qroam_clean_multi_data")
    libraries = (_library(source, 0), _library(source.adjoint(), 0))
    expected = tuple(
        resolve_component_program(library, 0, qroam_factories())
        for library in libraries
    )
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        actual = tuple(
            resolve_component_program(library, 0, qroam_factories())
            for library in libraries
        )
    assert actual == expected
