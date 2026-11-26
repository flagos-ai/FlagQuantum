"""Packed Program ownership is independent from Hubbard numerical conversion."""

import sys

import pytest

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_hubbard_packed_context import (
    hubbard_packed_oracle_context,
)
from flagquantum.core.value_math import SymbolicReal
from flagquantum.ecosystem._packed_oracle_component_library import (
    _prepare_angles,
    packed_oracle_library_factories,
)
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("kind", ("prepare", "select"))
@pytest.mark.parametrize("inverse", (False, True))
def test_actual_packed_constructor_uses_replaceable_numeric_owner_without_sdk(
    kind: str, inverse: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran.bloqs.multiplexers.black_box_select import _black_box_select
    from qualtran.bloqs.state_preparation.black_box_prepare import _black_box_prepare

    source = (_black_box_prepare if kind == "prepare" else _black_box_select).make()
    if inverse:
        source = source.adjoint()
    library = _library(source, 0)
    expected = resolve_component_program(library, 0, packed_oracle_library_factories())
    angles = _prepare_angles(2, 1.0, 4.0)
    seen: list[tuple[int, SymbolicReal, SymbolicReal]] = []

    def replacement(
        size: int, t: SymbolicReal, u: SymbolicReal
    ) -> tuple[float, float] | None:
        seen.append((size, t, u))
        return angles

    with monkeypatch.context() as patch:
        for module in (
            "qualtran",
            "cirq",
            "flagquantum.algorithms",
            "flagquantum.simulation",
        ):
            patch.setitem(sys.modules, module, None)
        factories, _, _, _ = hubbard_packed_oracle_context(angles=replacement)
        actual = resolve_component_program(library, 0, factories)
    assert actual == expected
    assert bool(seen) == (kind == "prepare")
    assert all(values == (2, 1, 4) for values in seen)
