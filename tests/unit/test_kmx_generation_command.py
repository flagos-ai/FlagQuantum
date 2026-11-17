"""Exercise the physical generator against an immutable locked CLI capture."""

import base64
import json
from pathlib import Path
from typing import TypedDict
from xml.etree.ElementTree import Element, fromstring

import pytest

from flagquantum.drawer.kmx_generate_command import main


class _Case(TypedDict):
    argv: list[str]
    kmx: str
    kmb: str


_SOURCE = json.loads(
    (Path(__file__).parent / "fixtures" / "kickmix_generation_command.json").read_text()
)
_CASES: list[_Case] = _SOURCE["cases"]


def _svg(node: Element) -> tuple[str, dict[str, str], str, list[object]]:
    return node.tag, node.attrib, (node.text or "").strip(), [_svg(x) for x in node]


@pytest.mark.parametrize("case", _CASES, ids=lambda case: " ".join(case["argv"]))
def test_generated_bytes(case: _Case, tmp_path: Path) -> None:
    out = tmp_path / "circuit"
    argv = ["--kind", *case["argv"], "--out", str(out), "--validate"]
    assert main(argv) == 0
    assert out.read_bytes() == case["kmx"].encode()
    assert main([*argv, "--kmb"]) == 0
    assert out.read_bytes() == base64.b64decode(case["kmb"])


def test_diagram_precedes_kmb_and_analysis_survives_skip(tmp_path: Path) -> None:
    out, analysis = tmp_path / "circuit", tmp_path / "analysis"
    argv = ["--kind", "iadd", "--n", "3", "--out", str(out), "--kmb", "--diagram"]
    assert main(argv) == 0
    expected = _SOURCE["diagram"].encode()
    assert out.read_bytes() == expected
    assert main([*argv, "--skip_output", "--flame_out", str(analysis)]) == 0
    assert out.read_bytes() == expected
    assert _svg(fromstring(analysis.read_bytes())) == _svg(
        fromstring(_SOURCE["analysis"])
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["--kind", "iadd"],
        ["--kind", "iadd", "--n", "1000001"],
        ["--kind", "lookup", "--clean", "0"],
        ["--kind", "unlookup", "--controlled"],
        ["--kind", "lookup", "--n", "31"],
    ],
)
def test_rejection_preserves_output(tmp_path: Path, argv: list[str]) -> None:
    out = tmp_path / "existing"
    out.write_bytes(b"previous")
    assert main([*argv, "--out", str(out)]) == 1
    assert out.read_bytes() == b"previous"
