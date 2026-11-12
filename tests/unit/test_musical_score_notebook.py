"""Compare the complete frozen score notebook with the native exported asset."""

import builtins
import hashlib
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]


def execute(notebook: dict[str, Any]) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"score-cell-{i}", "exec"), namespace)
    return namespace


def test_complete_score_notebook_and_exports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import IPython.display
    import matplotlib.pyplot as plt

    agg = pytest.importorskip("matplotlib.backends.backend_agg")

    raw = (
        ROOT.parent / "ftqc-reference-env/qualtran/drawing/musical_score.ipynb"
    ).read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "7205fa53b168a8b0de5429345dc25d2081979446"
    )
    source = json.loads(raw)
    native_path = ROOT / "examples/notebooks/register_musical_score.ipynb"
    native_raw = native_path.read_bytes()
    native = json.loads(native_raw)
    for notebook in (source, native):
        assert len(notebook["cells"]) == 7
        assert sum(c["cell_type"] == "code" for c in notebook["cells"]) == 3
        assert all(
            not c["metadata"] and not c.get("outputs") and not c.get("attachments")
            for c in notebook["cells"]
        )
    assert len({c["id"] for c in native["cells"]}) == 7
    assert native["metadata"]["kernelspec"]["name"] == "python3"
    source_dir, native_dir = tmp_path / "source", tmp_path / "native"
    source_dir.mkdir()
    native_dir.mkdir()
    renders: list[Any] = []
    imports = builtins.__import__

    def native_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "quimb", "cirq"}:
            raise AssertionError(f"Native score notebook imported {name}")
        return imports(name, *args, **kwargs)

    with plt.rc_context({"figure.dpi": 100, "font.family": "DejaVu Sans"}):
        with monkeypatch.context() as patch:
            patch.chdir(source_dir)
            patch.setattr(IPython.display, "display", lambda *args: None)
            expected = execute(source)
        try:
            with monkeypatch.context() as patch:
                patch.chdir(native_dir)
                patch.setattr(builtins, "__import__", native_import)
                patch.setattr(IPython.display, "display", renders.append)
                actual = execute(native)
            source_data = json.loads(
                (source_dir / "musical_score_example.json").read_text()
            )
            native_data = json.loads(
                (native_dir / "musical_score_example.json").read_text()
            )
            assert native_data == actual["msd"]
            for key in ("max_x", "max_y", "hlines", "vlines"):
                assert native_data[key] == source_data[key]

            # Port IDs use native logical lanes. All geometric and symbol data
            # must agree without altering either reference or native records.
            def ports(data: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
                return {
                    (p["seq_x"], p["y"]): {k: v for k, v in p.items() if k != "ident"}
                    for p in data["soqs"]
                }

            assert ports(native_data) == ports(source_data)
            assert len({p["ident"] for p in native_data["soqs"]}) == len(
                source_data["soqs"]
            )
            assert len(renders) == 2
            ElementTree.fromstring(renders[0].data)
            (tmp_path / "native-graph.svg").write_text(renders[0].data)
            expected_canvas = agg.FigureCanvasAgg(expected["fig"])
            actual_canvas = agg.FigureCanvasAgg(actual["fig"])
            expected_canvas.draw()
            actual_canvas.draw()
            assert np.array_equal(
                expected_canvas.buffer_rgba(), actual_canvas.buffer_rgba()
            )
            assert actual["ax"].get_xlim() == expected["ax"].get_xlim()
            assert actual["ax"].get_ylim() == expected["ax"].get_ylim()
            actual_canvas.print_png(tmp_path / "native-score.png")
            html = (native_dir / "musical_score_example.html").read_text()
            payload = html.split('type="application/json">', 1)[1].split(
                "</script>", 1
            )[0]
            assert json.loads(payload)["sequential"] == native_data
            assert "<script src=" not in html
            assert not plt.fignum_exists(actual["fig"].number)
        finally:
            plt.close(expected["fig"])
            if "actual" in locals():
                plt.close(actual["fig"])
    assert native_path.read_bytes() == native_raw
