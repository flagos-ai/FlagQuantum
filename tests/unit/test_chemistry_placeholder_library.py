"""Original analytical chemistry placeholders resolve without false gate bodies."""

import sys
from importlib import import_module

import pytest
import sympy

from flagquantum.algorithms.primitives.chemistry_qroam_model import (
    source_chemistry_qroam_cost,
    source_chemistry_qroam_two_axis_cost,
)
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_thc_selection_factories import (
    thc_selection_factories,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.ecosystem._chemistry_black_boxes_library import (
    chemistry_black_boxes_factories,
)
from flagquantum.errors import SerializationError
from tests.unit.test_cirq_decomposition_file import locked_source as locked_source
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_BASE = "qualtran.bloqs.chemistry.black_boxes."


@pytest.mark.parametrize(
    "kind,values",
    (
        ("QROAM", (51, 20, None)),
        ("QROAM", (17, 3, 1)),
        ("QROAM", (17, 3, 3)),
        ("QROAM", (1, 3, 1)),
        ("QROAM", (1, 1, 1)),
        ("QROAMTwoRegs", (5, 10, 3, 2, 4)),
        ("QROAMTwoRegs", (1, 7, 1, 3, 2)),
        ("QROAMTwoRegs", (1, 1, 1, 1, 1)),
    ),
)
@pytest.mark.parametrize("adjoint", (False, True))
def test_actual_constructor_libraries_and_independent_models(
    kind: str,
    values: tuple[int | None, ...],
    adjoint: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("qualtran.bloqs.chemistry.black_boxes")
    if kind == "QROAM":
        rows, width, block = values
        assert rows is not None and width is not None
        source = module.QROAM(rows, width, is_adjoint=adjoint, qroam_block_size=block)
        model = source_chemistry_qroam_cost(
            rows, width, block_size=block, adjoint=adjoint
        )
    else:
        first, second, block1, block2, width = values
        assert (
            first is not None
            and second is not None
            and block1 is not None
            and block2 is not None
            and width is not None
        )
        rows = first
        source = module.QROAMTwoRegs(
            first, second, block1, block2, width, is_adjoint=adjoint
        )
        model = source_chemistry_qroam_two_axis_cost(
            first, second, block1, block2, width, adjoint=adjoint
        )
    assert_source_model(model, source)
    baseline = None
    for depth in (0, 1, 2):
        library = _library(source, depth)
        encoded = serialize_component_library(library)
        assert (
            serialize_component_library(deserialize_component_library(encoded))
            == encoded
        )
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "qualtran", None)
            patch.setitem(sys.modules, "cirq", None)
            program = resolve_component_program(
                library, 0, chemistry_black_boxes_factories()
            )
        assert verify_program_record(program) == ()
        root = program.functions[0]
        assert root.registers == library.entries[0].declaration.registers
        assert not root.blocks
        assert tuple(r.name for r in root.registers or ()) == (
            ("trg",) if rows == 1 else ("sel", "trg")
        )
        with pytest.raises(
            ValueError, match="physical lowering requires component bodies"
        ):
            bind_register_program(program, (), clean=tuple(range(50)))
        if baseline is None:
            baseline = program
        else:
            assert program == baseline


@pytest.mark.parametrize(
    "field,value",
    (
        ("data_size", 0),
        ("target_bitsize", 0),
        ("data_size", sympy.Symbol("N")),
        ("qroam_block_size", -1),
        ("qroam_block_size", sympy.Symbol("k")),
        ("is_adjoint", 2),
        ("extra", 1),
    ),
)
def test_invalid_or_outside_positive_concrete_admission(
    field: str, value: int | sympy.Expr
) -> None:
    fields = {"data_size": 8, "target_bitsize": 3, field: value}
    with pytest.raises(SerializationError):
        chemistry_black_boxes_factories()[_BASE + "QROAM"]("main", None, fields)


@pytest.mark.parametrize("second,block1,block2", ((0, 1, 1), (2, 0, 1), (2, 1, -1)))
def test_second_axis_and_blocks_are_validated_despite_not_adding_ports(
    second: int, block1: int, block2: int
) -> None:
    with pytest.raises(SerializationError, match="positive concrete integer"):
        chemistry_black_boxes_factories()[_BASE + "QROAMTwoRegs"](
            "main",
            None,
            {
                "data_a_size": 5,
                "data_b_size": second,
                "data_a_block_size": block1,
                "data_b_block_size": block2,
                "target_bitsize": 4,
            },
        )


def test_automatic_forward_model_refusal_remains_distinct_from_declaration() -> None:
    module = import_module("qualtran.bloqs.chemistry.black_boxes")
    source = module.QROAM(2, 3)
    library = _library(source, 0)
    program = resolve_component_program(library, 0, chemistry_black_boxes_factories())
    assert not program.functions[0].blocks
    with pytest.raises(AssertionError):
        module.get_qroam_cost_clean_ancilla(2, 3)
    with pytest.raises(ValueError, match="rows >= target width"):
        source_chemistry_qroam_cost(2, 3)


def test_optional_none_block_and_existing_phase_provider_are_reused() -> None:
    provider = chemistry_black_boxes_factories()
    one = provider[_BASE + "QROAM"]
    assert one("main", None, {"data_size": 8, "target_bitsize": 3}) == one(
        "main", None, {"data_size": 8, "target_bitsize": 3, "qroam_block_size": None}
    )
    source = import_module("qualtran.bloqs.chemistry.black_boxes").ApplyControlledZs(
        (0, 1), 2
    )
    library = _library(source, 0)
    actual = resolve_component_program(library, 0, provider)
    existing = resolve_component_program(library, 0, thc_selection_factories())
    assert matches_component_definition(actual, "main", existing)
