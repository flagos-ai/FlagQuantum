"""Complete triangular-index source declaration, glyphs and consumer boundary."""

from hashlib import sha1
from pathlib import Path

import pytest

from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.sorting.contiguous_index import (
    source_contiguous_index_cost,
)
from flagquantum.drawer.chemistry_symbols import contiguous_index_wire_symbol
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import register_score_data
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.test_drawer_component_score import assert_source_geometry
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_register_tensor import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.arithmetic.conversions import contiguous_index

    for name, blob in (
        ("contiguous_index.py", "79f552805b99702e0ee732b8be8fd46f107c0a13"),
        ("contiguous_index_test.py", "01d5618ecd38083defdca5b2c8975e5bfbc2025f"),
    ):
        raw = (Path(contiguous_index.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


def test_actual_factory_signature_model_and_source_execution_boundaries() -> None:
    from qualtran import DecomposeNotImplementedError, QUInt, Side
    from qualtran.bloqs.arithmetic.conversions.contiguous_index import (
        _TO_CONTG_INDX,
        _to_contg_index,
    )

    source = _to_contg_index.make()
    assert _TO_CONTG_INDX.examples == (_to_contg_index,)
    assert [(r.name, r.dtype, r.side) for r in source.signature] == [
        ("mu", QUInt(4), Side.THRU),
        ("nu", QUInt(4), Side.THRU),
        ("s", QUInt(8), Side.THRU),
    ]
    assert_source_model(source_contiguous_index_cost(4, 8), source)
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    with pytest.raises(TypeError, match="unexpected keyword argument 's'"):
        source.call_classically(mu=1, nu=2, s=7)
    with pytest.raises(
        ValueError, match="Foreign oracle primitive cannot be decomposed"
    ):
        import_register_emitter(source)
    assert source.on_classical_vals(mu=1, nu=2)["s"] == 4


def test_actual_factory_complete_score_preserves_source_printed_polynomial() -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.arithmetic.conversions.contiguous_index import _to_contg_index
    from qualtran.drawing import get_musical_score_data

    source = _to_contg_index.make()
    registers = tuple(native_signature(source.signature))
    names = tuple(r.name for r in registers)
    step = RegisterStep(
        str(source), names, names, input_registers=registers, output_registers=registers
    )
    score = layout_register_score(
        {r.name: int(r.bitsize) for r in registers}, names, (step,)
    )
    builder, inputs = BloqBuilder.from_signature(source.signature)
    composite = builder.finalize(**builder.add_d(source, **inputs))
    assert_source_geometry(
        register_score_data(score, register_wire_symbol=contiguous_index_wire_symbol),
        get_musical_score_data(composite),
        ordered_ports=True,
    )
    assert contiguous_index_wire_symbol(step, registers[2], ()) == (
        "TextBox",
        {"text": "⊕ν(ν-1)/2+μ"},
    )
    # The source glyph's minus differs from its actual plus-polynomial value.
    assert source.on_classical_vals(mu=1, nu=2)["s"] != 1 + 2 * (2 - 1) // 2


def test_single_factorization_uses_replaced_index_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.algorithms.primitives import (
        single_factorization_resource_model as native,
    )

    seen: list[tuple[int, int]] = []

    def replacement(width: int, output_width: int) -> ComponentCost:
        seen.append((width, output_width))
        return ComponentCost(
            "index probe", 2 * width + output_width, GateCost(toffoli=17)
        )

    monkeypatch.setattr(native, "source_contiguous_index_cost", replacement)
    model = native.source_single_factorization_inner_prepare_cost(3, 8, 4)
    assert seen == [(2, 4)]
    assert model.calls[1].component == ComponentCost(
        "index probe", 8, GateCost(toffoli=17)
    )
    assert model.calls[1].repetitions == 1
