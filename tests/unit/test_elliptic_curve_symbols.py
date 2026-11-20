"""Compare point-addition glyph callbacks and complete register-score layouts."""

from functools import partial

import pytest
from attrs import asdict

from flagquantum.algorithms.primitives.elliptic_curve_fixed_point import (
    fixed_point_addition_signature,
)
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.drawer.elliptic_curve_symbols import (
    elliptic_curve_addition_wire_symbol,
)
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import register_score_data
from tests.unit.test_drawer_component_score import assert_source_geometry

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("window", [None, 1, 3])
@pytest.mark.parametrize("symbolic", [False, True])
def test_complete_callbacks_and_original_unknown_port_failure(
    window: int | None, symbolic: bool
) -> None:
    import sympy
    from qualtran import QBit
    from qualtran import Register as OriginalRegister
    from qualtran.bloqs.cryptography.ecc import ECAddR, ECPoint, ECWindowAddR

    n, x, y = sympy.symbols("n x y") if symbolic else (5, 15, 13)
    point = ECPoint(x, y, 17)
    source = ECAddR(n, point) if window is None else ECWindowAddR(n, point, window)
    callback = partial(
        elliptic_curve_addition_wire_symbol,
        x=x,
        y=y,
        windowed_width=None if window is None else n,
    )
    signature = fixed_point_addition_signature(n, window)
    step = RegisterStep(str(source))
    for name in (None, "ctrl", "x", "y"):
        original_register = None if name is None else source.signature.get_left(name)
        register = None if name is None else signature.get_left(name)
        for index in ((), (0,), (2,)):
            symbol = source.wire_symbol(original_register, index)
            assert callback(step, register, index) == (
                type(symbol).__name__,
                asdict(symbol),
            )
    with pytest.raises(ValueError, match="Unrecognized register name other"):
        source.wire_symbol(OriginalRegister("other", QBit()))
    with pytest.raises(ValueError, match="Unrecognized register name other"):
        callback(step, Register("other", quantum_bit_type()), ())


@pytest.mark.parametrize("window", [None, 1, 3])
def test_complete_original_score_geometry_and_native_renderer(
    window: int | None,
) -> None:
    from matplotlib import pyplot as plt
    from qualtran import BloqBuilder
    from qualtran.bloqs.cryptography.ecc import ECAddR, ECPoint, ECWindowAddR
    from qualtran.drawing import get_musical_score_data

    from flagquantum.drawer.register_score_data import export_register_score_json
    from flagquantum.drawer.register_score_mpl import draw_register_score

    width = 3
    source = (
        ECAddR(width, ECPoint(0, 1, 3, 1))
        if window is None
        else ECWindowAddR(width, ECPoint(0, 1, 3, 1), window)
    )
    builder, inputs = BloqBuilder.from_signature(source.signature)
    original = builder.finalize(**builder.add_d(source, **inputs))
    signature = fixed_point_addition_signature(width, window)
    names = tuple(
        reg.name if not reg.shape else f"{reg.name}[{i}]"
        for reg in signature
        for i in range(1 if not reg.shape else reg.shape[0])
    )
    widths = {name: (1 if name.startswith("ctrl") else width) for name in names}
    score = layout_register_score(
        widths,
        names,
        (
            RegisterStep(
                str(source),
                names,
                names,
                input_registers=tuple(signature),
                output_registers=tuple(signature),
            ),
        ),
        outputs=names,
        input_registers=tuple(signature),
        output_registers=tuple(signature),
    )
    observed = register_score_data(
        score,
        register_wire_symbol=partial(
            elliptic_curve_addition_wire_symbol,
            x=0,
            y=1,
            windowed_width=None if window is None else width,
        ),
    )
    assert_source_geometry(
        observed, get_musical_score_data(original), ordered_ports=True
    )
    before = export_register_score_json(observed)
    figure, _ = draw_register_score(observed)
    try:
        figure.canvas.draw()
        assert export_register_score_json(observed) == before
    finally:
        plt.close(figure)
