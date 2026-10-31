"""Static input families preserve Python requirements and native guards."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("valid", [True, False])
def test_source_overloads_distinguish_file_text_and_python(
    tmp_path, monkeypatch, valid
):
    api = pytest.importorskip("mypy.api")
    monkeypatch.setenv("MYPYPATH", str(ROOT))
    code = """from pathlib import Path
from flagquantum import synth
def polynomial(x: int) -> int:
    return x*x+1
"""
    if valid:
        code += """synth.synthesize(polynomial, inputs={"x":synth.UInt(2)})
synth.synthesize(function=polynomial, inputs={"x":synth.UInt(2)}, memory_limit_bytes=512)
synth.synthesize(Path("source.qmod"))
synth.synthesize("qfunc main(output q:qbit) { allocate(q); }", source_format="qmod")
"""
    else:
        code += """synth.synthesize(polynomial)
synth.synthesize("text")
synth.synthesize("text", source_format="qmod", inputs={"x":synth.UInt(1)})
synth.synthesize(Path("source.qmod"), source_name="override")
synth.synthesize(polynomial, inputs={"x":synth.UInt(1)}, source_format="qmod")
"""
    path = tmp_path / "typing_case.py"
    path.write_text(code)
    stdout, stderr, result = api.run(
        [
            "--strict",
            "--python-version",
            "3.12",
            "--ignore-missing-imports",
            "--follow-imports=silent",
            "--cache-dir",
            str(tmp_path / "mypy-cache"),
            str(path),
        ]
    )
    if valid:
        assert result == 0, stdout + stderr
    else:
        assert result == 1, stdout + stderr
        assert stdout.count(": error:") == 5, stdout
