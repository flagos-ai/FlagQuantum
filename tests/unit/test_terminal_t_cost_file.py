"""Audit the locked terminal estimator and its boundaries against native costs."""

from hashlib import sha1
from math import ceil, log2
from pathlib import Path

import pytest

from flagquantum.compiler._component_trace.gate_budget import (
    legacy_gate_budget,
    t_and_ccz_budget,
)
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    ArithmeticCall,
    FixedTCall,
    ResourceCall,
    RotationCall,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler._component_trace.terminal_t_cost import (
    estimate_terminal_t_cost,
    query_terminal_t_cost,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_terminal_t_cost_frozen_source_files() -> None:
    root = (
        Path(__file__).resolve().parents[3]
        / "ftqc-reference-env/qualtran/resource_counting"
    )
    for file, expected in (
        ("t_counts_from_sigma.py", "eb4578e362849eb3e7719944bb7c01b2dc9adbc4"),
        ("t_counts_from_sigma_test.py", "d0d6aecc3892742dee33a535a97e473724e20858"),
    ):
        path = root / file
        if not path.is_file():
            pytest.skip("Frozen terminal T cost files are not installed")
        data = path.read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == expected


def test_terminal_t_rounds_each_rotation_and_preserves_model_profiles() -> None:
    source = pytest.importorskip("qualtran.resource_counting.t_counts_from_sigma")
    from qualtran.bloqs.basic_gates import Rx

    default = RotationCall("rx", 0.3)
    custom = RotationCall("rx", 0.3, 1e-4)
    with pytest.warns(DeprecationWarning):
        assert (
            estimate_terminal_t_cost({default: 1})
            == source.t_counts_from_sigma({Rx(0.3): 1})
            == 52
        )
    with pytest.warns(DeprecationWarning):
        assert (
            estimate_terminal_t_cost({custom: 3})
            == source.t_counts_from_sigma({Rx(0.3, eps=1e-4): 3})
            == 75
        )
    assert ceil(3 * (1.149 * log2(1e4) + 9.2)) == 74
    assert query_qec_gate_cost(custom).rotation == 1
    assert legacy_gate_budget(query_qec_gate_cost(custom))["t"] == 0
    assert t_and_ccz_budget(query_qec_gate_cost(custom))["n_t"] == 11


def test_unexpanded_arithmetic_requires_a_native_declaration() -> None:
    source = pytest.importorskip("qualtran.resource_counting.t_counts_from_sigma")
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add

    with pytest.warns(DeprecationWarning):
        assert source.t_counts_from_sigma({Add(QUInt(3)): 1}) == 0
    arithmetic = ArithmeticCall("add", (3, 3))
    with pytest.raises(NotImplementedError, match="Expand arithmetic"):
        estimate_terminal_t_cost({arithmetic: 1})

    leaf = FixedTCall("toffoli")
    calls: dict[ResourceCall, dict[ResourceCall, SymbolicInt]] = {arithmetic: {leaf: 2}}
    cache: dict[ResourceCall, SymbolicInt] = {}
    assert query_terminal_t_cost(arithmetic, declared_calls=calls, cache=cache) == 8
    assert cache == {leaf: 4, arithmetic: 8}
