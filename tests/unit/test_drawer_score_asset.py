"""Verify the original browser asset and preserve it through native export."""

from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha1
from pathlib import Path

import pytest

from flagquantum.drawer.register_score_data import export_register_score_json
from flagquantum.drawer.register_score_html import export_register_score_html

pytestmark = pytest.mark.unit
ASSET = Path(__file__).resolve().parents[1] / "fixtures/register_score/example.json"


def test_original_asset_identity_and_records() -> None:
    raw = ASSET.read_bytes()
    assert len(raw) == 4681
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "b5ed08cad72ae69d275fc89b90337b7dc4daf3a8"
    )
    data = json.loads(raw)
    assert (data["max_x"], data["max_y"]) == (3, 6)
    assert [len(data[key]) for key in ("soqs", "hlines", "vlines")] == [20, 7, 3]
    assert all("flavor" not in line for line in data["hlines"])
    assert [line["label"] for line in data["vlines"]] == ["And"] * 3
    assert [
        port["symb_attributes"]["filled"]
        for port in data["soqs"]
        if port["symb_cls"] == "Circle"
    ] == [True, True, True, False, True, True]


def test_original_asset_offline_and_json_exports_preserve_records() -> None:
    raw = ASSET.read_bytes()
    data = json.loads(raw)
    assert json.loads(export_register_score_json(data)) == data
    page = export_register_score_html(data)
    payload = page.split('type="application/json">', 1)[1].split("</script>", 1)[0]
    views = json.loads(payload)
    assert views["sequential"] == data
    projected = deepcopy(data)
    projected["max_x"] = 4
    columns = {-1: -1, 0: 1, 1: 2, 2: 3, 3: 4}
    for port in projected["soqs"]:
        port["seq_x"] = columns[port["seq_x"]]
    for line in projected["hlines"]:
        line["seq_x_start"] = columns[line["seq_x_start"]]
        line["seq_x_end"] = columns[line["seq_x_end"]]
    for connector in projected["vlines"]:
        connector["x"] = columns[connector["x"]]
    assert views["dependency"] == projected
    assert "<script src=" not in page
    assert data == json.loads(raw)
    assert ASSET.read_bytes() == raw
