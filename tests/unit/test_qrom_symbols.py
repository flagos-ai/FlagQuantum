"""Compare native score glyphs with the frozen lookup source."""

from math import prod

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_source_qrom_wire_glyphs() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("qualtran")
    from qualtran.bloqs.data_loading import QROM

    from flagquantum.drawer.qrom_symbols import qrom_wire_symbol
    from flagquantum.drawer.register_score import RegisterStep

    step = RegisterStep("QROM", (), (), gate="qrom")
    for shape in ((4,), (2, 2), (2, 2, 2)):
        data = np.arange(prod(shape)).reshape(shape)
        source = QROM.build_from_data(data, data, data, num_controls=1)
        for register in (None, *source.signature):
            symbol = source.wire_symbol(register)
            native = qrom_wire_symbol(step, register.name if register else None, "THRU")
            expected = {"text": symbol.text} if hasattr(symbol, "text") else {}
            assert native == (type(symbol).__name__, expected)
