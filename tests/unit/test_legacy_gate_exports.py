"""Compatibility coverage for historical class-based gate exports."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_gate_classes_remain_compatibility_exports_outside_stable_root() -> None:
    import flagquantum as fq

    assert "CX" in fq.ops.__all__
    assert "CX" not in fq.__all__
    assert fq.CX is fq.ops.CX
    assert fq.CX(wires=[0, 1]).wires == [0, 1]
