"""Compare complete native physical diagrams with locked source text."""

import json
from pathlib import Path

import pytest

from flagquantum.compiler.formats.kmb import decode_kmb, encode_kmb
from flagquantum.drawer.reversible_phase_diagram import render_reversible_phase_text

_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "kickmix_circuit_text.json").read_text()
)


@pytest.mark.parametrize("name", tuple(_CASES))
@pytest.mark.parametrize("use_unicode", [False, True])
def test_locked_text_diagram(name: str, use_unicode: bool) -> None:
    case = _CASES[name]
    kmx: str = case["kmx"]
    expected: str = case["unicode" if use_unicode else "ascii"]
    assert render_reversible_phase_text(kmx, use_unicode=use_unicode) == expected
    assert (
        render_reversible_phase_text(
            decode_kmb(encode_kmb(kmx)), use_unicode=use_unicode
        )
        == expected
    )
