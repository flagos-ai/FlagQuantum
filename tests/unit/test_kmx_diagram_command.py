"""Match locked diagram command output across physical input formats."""

import json
import re
import subprocess
import sys
from pathlib import Path
from xml.etree.ElementTree import Element, fromstring

import pytest

from flagquantum.compiler.formats.kmb import encode_kmb
from flagquantum.drawer.kmx_diagram_command import main

_SOURCE = json.loads(
    (Path(__file__).parent / "fixtures" / "kickmix_diagram_command.json").read_text()
)
_INPUTS = {"incrementer": "CCX q0 q1 q2\nCX q0 q1\nX q0", "empty": ""}


def _xml(node: Element) -> tuple[str, dict[str, str], str | None, list[object]]:
    return node.tag, node.attrib, node.text, [_xml(child) for child in node]


@pytest.mark.parametrize("name", tuple(_INPUTS))
@pytest.mark.parametrize("kind", ["text", "html", "svg"])
@pytest.mark.parametrize("binary", [False, True])
def test_source_diagram_file_output(
    name: str, kind: str, binary: bool, tmp_path: Path
) -> None:
    source, out = tmp_path / "input", tmp_path / "diagram"
    text = _INPUTS[name]
    source.write_bytes(encode_kmb(text) if binary else text.encode())
    assert main(["--in", str(source), "--out", str(out), "--kind", kind]) == 0
    raw = out.read_bytes()
    assert raw.endswith(b"\n")
    expected: str = _SOURCE[name][kind]
    if kind == "text":
        assert raw == expected.encode()
    elif kind == "svg":
        assert _xml(fromstring(raw)) == _xml(fromstring(expected))
    else:
        html = raw.decode()
        stats = re.compile(r"\(Circuit stats: (.*?)\)</div>")
        local, locked = stats.search(html), stats.search(expected)
        assert local is not None and locked is not None and local[1] == locked[1]
        payload = re.search(r'id="circuit-data">(.*?)</script>', html)
        assert payload is not None
        boxes = json.loads(payload[1])
        for dimension, axis in (("width", "x"), ("height", "y")):
            boundary = re.search(rf"const max_{axis} = (\d+);", expected)
            assert boundary is not None
            assert boxes[dimension] == int(boundary[1])
        assert len(boxes["objects"]) == expected.count("new BoundedObject(")


def test_default_stdin_stdout_matches_locked_text() -> None:
    output = subprocess.run(
        [sys.executable, "-m", "flagquantum.drawer.kmx_diagram_command"],
        input=_INPUTS["incrementer"].encode(),
        capture_output=True,
        timeout=20,
    )
    assert (output.returncode, output.stdout, output.stderr) == (
        0,
        _SOURCE["incrementer"]["text"].encode(),
        b"",
    )


def test_invalid_input_or_kind_preserves_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src, out = tmp_path / "corrupt", tmp_path / "rendering"
    src.write_bytes(b"\xd7")
    out.write_bytes(b"previous")
    assert main(["--in", str(src), "--out", str(out), "--kind", "bad"]) == 1
    assert main(["--in", str(src), "--out", str(out)]) == 1
    assert "Invalid KMB magic" in capsys.readouterr().err
    assert out.read_bytes() == b"previous"
