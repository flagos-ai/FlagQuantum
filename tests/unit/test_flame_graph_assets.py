"""Require the complete frozen renderer asset triple in native distributions."""

import os
from hashlib import sha1
from pathlib import Path
from xml.etree import ElementTree

import pytest

from tools.verify_distribution_artifacts import (
    REQUIRED_MEMBER_SUFFIXES,
    _missing_required_members,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
ASSETS = "flagquantum/drawer/_assets/"


def test_complete_frozen_asset_bytes_and_distribution_requirements() -> None:
    blobs = {
        "resource_flame_graph.pl": "8c917ecf323e13c1ab3c1107c1963f33dec0123e",
        "resource_flame_graph.metadata": "8c6696d1a9b9c36c929d0a01a00dfe46f4de1cbb",
        "licenses/cddl-1.0.txt": "2230598ebad9e470736e12119db740007c1a764b",
    }
    for name, expected_blob in blobs.items():
        path = ASSETS + name
        raw = (ROOT / path).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == expected_blob
        assert path in REQUIRED_MEMBER_SUFFIXES
        assert _missing_required_members(
            tuple(member for member in REQUIRED_MEMBER_SUFFIXES if member != path)
        ) == (path,)


@pytest.mark.skipif(os.name == "nt", reason="Direct Perl script execution is Unix-only")
def test_native_packaged_renderer_and_empty_input() -> None:
    pytest.importorskip("numpy")
    from flagquantum.drawer.flame_graph import render_flame_graph_svg

    executable = ROOT / ASSETS / "resource_flame_graph.pl"
    assert executable.stat().st_mode & 0o111
    svg = render_flame_graph_svg(
        ["Algorithm(T:4);Select(T:4)\t4"], executable=str(executable)
    )
    tree = ElementTree.fromstring(svg)
    text = "".join(tree.itertext())
    assert "4 TCounts" in text
    assert "Algorithm" in text and "Select" in text
    assert "function zoom" in text and "function search" in text
    with pytest.raises(RuntimeError, match="No stack counts found"):
        render_flame_graph_svg([], executable=str(executable))
