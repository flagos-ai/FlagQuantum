"""Complete addition source file call graphs retain actual upstream drawings."""

from collections import Counter
from typing import Any

import pytest

from examples.algorithms import arithmetic_ui_export as native
from examples.algorithms.addition_decomposition import addition_program, addition_score
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_arithmetic_ui_export import originals as originals
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_register_gate_ui_export import _canonical_score

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
CONCRETE = (
    "add_small",
    "add_large",
    "add_diff_size_regs",
    "add_oop_small",
    "add_oop_large",
    "add_k_small",
    "add_k_large",
)


@pytest.mark.parametrize("identifier", CONCRETE)
def test_original_ordered_calls_operands_and_fresh_score(
    originals: tuple[Any, dict[str, Any]], identifier: str
) -> None:
    from qualtran.drawing import get_musical_score_data

    ports = native.signature(identifier)
    record = addition_program(
        native.name(identifier), ports, native.attributes(identifier)
    )
    assert record is not None and verify_program_record(record) == ()
    source = originals[1][identifier].decompose_bloq()
    assert _native_trace(record) == _source_trace(_source(ports), source)
    actual = addition_score(
        native.name(identifier), ports, native.attributes(identifier)
    )
    assert actual is not None
    if identifier in {"add_large", "add_oop_large"}:
        with pytest.raises(IndexError):
            get_musical_score_data(source)
        return
    fresh = get_musical_score_data(source)
    expected = {
        "max_x": fresh.max_x,
        "max_y": fresh.max_y,
        **{
            name: [item.json_dict() for item in getattr(fresh, name)]
            for name in ("soqs", "hlines", "vlines")
        },
    }
    assert _canonical_score(dict(actual)) == _canonical_score(expected)


@pytest.mark.parametrize("identifier", ("add_symb", "add_oop_symb", "add_k"))
def test_original_symbolic_decomposition_boundary(
    originals: tuple[Any, dict[str, Any]], identifier: str
) -> None:
    from qualtran import DecomposeTypeError

    with pytest.raises(DecomposeTypeError):
        originals[1][identifier].decompose_bloq()
    assert (
        addition_program(
            native.name(identifier),
            native.signature(identifier),
            native.attributes(identifier),
        )
        is None
    )


@pytest.mark.parametrize("width", (1, 2, 3))
@pytest.mark.parametrize("include_msb", (False, True))
def test_source_ordered_copy_full_isometry_and_all_erasure_histories(
    monkeypatch: pytest.MonkeyPatch, width: int, include_msb: bool
) -> None:
    from flagquantum.compiler.registers.arithmetic.out_of_place_addition_component import (
        out_of_place_addition_program,
    )
    from flagquantum.compiler.registers.component_graph import copy_component_program
    from flagquantum.core.ir.program import ProgramRecord
    from tests.unit import test_out_of_place_addition_component as checks

    def make(
        width: int, *, include_msb: bool = True, adjoint: bool = False
    ) -> ProgramRecord:
        record = out_of_place_addition_program(
            width, include_msb=include_msb, adjoint=adjoint
        )
        return record if adjoint else copy_component_program(record)

    from flagquantum.compiler.registers.register_binding import bind_register_program

    inputs = (tuple(range(width)), tuple(range(width, 2 * width)))
    clean = tuple(range(3 * width + int(include_msb) - 1, 2 * width - 1, -1))
    primitive_counts = []
    for record in (
        out_of_place_addition_program(width, include_msb=include_msb),
        make(width, include_msb=include_msb),
    ):
        bound = bind_register_program(
            record, inputs, clean=clean, allow_workspace_outputs=True
        )
        primitive_counts.append(Counter(g.name for g in bound.circuit.instructions))
    assert primitive_counts[0] == primitive_counts[1]

    # Substitute only the native fixture. The locked reference stays immutable.
    monkeypatch.setattr(checks, "out_of_place_addition_program", make)
    checks.test_source_full_isometry_entangled_spectator_and_all_erasure_histories(
        width, include_msb
    )
