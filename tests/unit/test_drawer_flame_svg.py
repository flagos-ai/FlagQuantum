"""Verify local FlameGraph execution and owned temporary-file lifecycle."""

import subprocess
import tempfile
from collections.abc import Iterator
from hashlib import sha1
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pytest

from flagquantum.compiler._component_trace.resource_generalization import (
    FixedTCall,
    ResourceCall,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.drawer import flame_graph
from flagquantum.drawer.flame_graph import (
    iter_resource_flame_graph_data,
    render_flame_graph_svg,
)

pytestmark = pytest.mark.unit
REFERENCE_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "Qualtran-linear-adjoint-fix/qualtran/third_party/flamegraph/flamegraph.pl"
)


@pytest.fixture
def temporary_paths(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    paths: list[Path] = []

    def create(**kwargs: Any) -> Any:
        stream = tempfile.TemporaryDirectory(**kwargs)
        paths.append(Path(stream.name) / "folded.txt")
        return stream

    monkeypatch.setattr(flame_graph, "TemporaryDirectory", create)
    return paths


def test_streamed_utf8_input_and_success_cleanup(
    monkeypatch: pytest.MonkeyPatch, temporary_paths: list[Path]
) -> None:
    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert args[1:3] == ["--countname", "TCounts"]
        assert (
            Path(args[3]).read_bytes()
            == "Rootλ(T:7);Leaf(T:7)\t7\nRootλ(T:1)\t1".encode()
        )
        assert kwargs == {"capture_output": True, "check": True, "timeout": 2.5}
        return subprocess.CompletedProcess(args, 0, stdout=b"<svg/>")

    monkeypatch.setattr(subprocess, "run", run)
    rows = iter(["Rootλ(T:7);Leaf(T:7)\t7", "Rootλ(T:1)\t1"])
    assert (
        render_flame_graph_svg(rows, executable="/usr/bin/perl", timeout=2.5)
        == b"<svg/>"
    )
    assert len(temporary_paths) == 1 and not temporary_paths[0].exists()


@pytest.mark.parametrize("failure", ["timeout", "process", "os"])
def test_process_errors_retain_cause_and_cleanup(
    failure: str, monkeypatch: pytest.MonkeyPatch, temporary_paths: list[Path]
) -> None:
    errors: dict[str, Exception] = {
        "timeout": subprocess.TimeoutExpired("renderer", 0.01),
        "process": subprocess.CalledProcessError(
            2, "renderer", stderr=b"ignored-prefix" + b"x" * 5000 + b"\xfffailure"
        ),
        "os": OSError("execution denied"),
    }

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert Path(args[-1]).exists()
        raise errors[failure]

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError) as raised:
        render_flame_graph_svg(["Root\t1"], executable="/usr/bin/perl", timeout=0.01)
    assert raised.value.__cause__ is errors[failure]
    if failure == "process":
        assert "ignored-prefix" not in str(raised.value)
        assert "�failure" in str(raised.value)
        assert len(str(raised.value)) < 4100
    assert len(temporary_paths) == 1 and not temporary_paths[0].exists()


def test_generator_failure_cleans_input_before_process(
    temporary_paths: list[Path],
) -> None:
    def rows() -> Iterator[str]:
        yield "Root\t1"
        raise ValueError("input generation failed")

    with pytest.raises(ValueError, match="input generation failed"):
        render_flame_graph_svg(rows(), executable="/usr/bin/perl")
    assert len(temporary_paths) == 1 and not temporary_paths[0].exists()


def test_missing_executable_does_not_consume_input(temporary_paths: list[Path]) -> None:
    def rows() -> Iterator[str]:
        raise AssertionError("Input must remain untouched")
        yield "unreachable"

    with pytest.raises(ImportError, match="requires an executable"):
        render_flame_graph_svg(rows(), executable="/nonexistent-flagquantum-flamegraph")
    assert temporary_paths == []


def canonical_svg(raw: bytes) -> str:
    """Perl hash iteration changes only the order of nonoverlapping frames."""
    root = ET.fromstring(raw)
    frames = root.find(".//*[@id='frames']")
    assert frames is not None
    frames[:] = sorted(frames, key=ET.tostring)
    return ET.canonicalize(ET.tostring(root, encoding="unicode"))


@pytest.mark.ftqc_reference
def test_actual_svg_matches_untouched_source_and_frozen_asset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from qualtran.bloqs.basic_gates.swap import CSwap
    from qualtran.drawing import flame_graph as source

    for name, expected in (
        ("flamegraph.pl", "8c917ecf323e13c1ab3c1107c1963f33dec0123e"),
        ("LICENSE", "2230598ebad9e470736e12119db740007c1a764b"),
        ("METADATA", "8c6696d1a9b9c36c929d0a01a00dfe46f4de1cbb"),
    ):
        raw = (REFERENCE_SCRIPT.parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == expected
    asset = tmp_path / "qualtran/third_party/flamegraph/flamegraph.pl"
    asset.parent.mkdir(parents=True)
    asset.symlink_to(REFERENCE_SCRIPT)
    monkeypatch.setattr(
        source, "__file__", str(tmp_path / "qualtran/drawing/flame_graph.py")
    )
    expected = source.get_flame_graph_svg_data(CSwap(10))
    assert isinstance(expected, str)
    root, fredkin = ComponentCost("CSwap[10]", 21), FixedTCall("fredkin")
    names: dict[ResourceCall, str] = {root: root.name, fredkin: "TwoBitCSwap"}
    rows = iter_resource_flame_graph_data(
        root, declared_calls={root: {fredkin: 10}}, title=names.__getitem__
    )
    actual = render_flame_graph_svg(rows, executable=str(REFERENCE_SCRIPT))
    assert canonical_svg(actual) == canonical_svg(expected.encode())
    xml = ET.fromstring(actual)
    assert "70 TCounts" in "".join(xml.itertext())
    assert "function zoom" in "".join(xml.itertext())
    assert "function search" in "".join(xml.itertext())
    (tmp_path / "flame.svg").write_bytes(actual)


@pytest.mark.ftqc_reference
def test_actual_renderer_empty_data_and_timeout_cleanup(
    temporary_paths: list[Path], tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError, match="No stack counts found"):
        render_flame_graph_svg([], executable=str(REFERENCE_SCRIPT))
    sleeper = tmp_path / "sleep-renderer"
    sleeper.write_text("#!/usr/bin/perl\nsleep 5;\n")
    sleeper.chmod(0o700)
    with pytest.raises(RuntimeError, match="timed out"):
        render_flame_graph_svg(["Root\t1"], executable=str(sleeper), timeout=0.02)
    assert len(temporary_paths) == 2 and all(
        not path.exists() for path in temporary_paths
    )
