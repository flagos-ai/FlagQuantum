"""Compare every ECC addition cost declaration without pricing opaque symbols."""

from collections import Counter
from collections.abc import Hashable
from hashlib import sha1
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
    import sympy

    import_module("qualtran")
    import_module("cirq")
    import_module("kickmix")
except ModuleNotFoundError as error:
    unavailable_reference(error)

from flagquantum.algorithms.primitives.elliptic_curve_addition import (
    EllipticCurveAddition,
)
from flagquantum.algorithms.primitives.elliptic_curve_addition_model import (
    elliptic_curve_addition_callees,
    elliptic_curve_addition_parameterized_graph,
)
from flagquantum.core.ir.program.register_types import (
    montgomery_register_type,
    quantum_bit_type,
)
from flagquantum.core.value_math import SymbolicInt

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.cryptography.ecc import ec_add

    for filename, digest in (
        ("ec_add.py", "2caf4f002152e51f1a09048d459c0b839280a4d1"),
        ("ec_add_test.py", "2fa735023da4c1b79d37079a7945aa2929e748b8"),
    ):
        raw = (Path(ec_add.__file__).parent / filename).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    kickmix = import_module("kickmix")
    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")


def source_key(source: Any) -> Hashable:
    """View original constructor fields, including their actual wrapper class."""
    name = type(source).__name__.removeprefix("_")
    if name in {"Adjoint", "ControlledViaAnd"}:
        child = source_key(source.subbloq)
        if name == "Adjoint":
            return name, child
        clauses = tuple(
            (
                (
                    quantum_bit_type()
                    if type(dtype).__name__ == "QBit"
                    else montgomery_register_type(dtype.bitsize)
                ),
                tuple(values.shape),
                tuple(int(v) for v in np.asarray(values).flat),
            )
            for dtype, values in zip(
                source.ctrl_spec.qdtypes, source.ctrl_spec.cvs, strict=True
            )
        )
        return name, child, clauses
    if name.startswith("ECAdd"):
        return (
            name,
            source.n,
            source.mod,
            *((source.window_size,) if hasattr(source, "window_size") else ()),
        )
    if name in {"Equals", "Xor"}:
        return name, montgomery_register_type(source.dtype.bitsize)
    if name in {"ModNeg", "ModSub", "ModDbl", "CModSub", "CModAdd", "CModNeg"}:
        return (
            name,
            montgomery_register_type(source.dtype.bitsize),
            source.mod,
            *((source.cv,) if name.startswith("C") else ()),
        )
    if name == "ModAdd":
        return name, source.bitsize, source.mod
    if name == "KaliskiModInverse":
        return name, source.bitsize, source.mod, source.uncompute
    if name == "DirtyOutOfPlaceMontgomeryModMul":
        return name, source.bitsize, source.window_size, source.mod, source.uncompute
    if name in {"MultiControlX", "MultiAnd"}:
        pattern = source.cvs
        return name, (
            ("HasLength", pattern.n)
            if type(pattern).__name__ == "HasLength"
            else tuple(pattern)
        )
    if name == "MultiTargetCNOT":
        return name, source.bitsize
    assert name in {"CNOT", "Toffoli", "XGate"}, name
    return (name,)


def original(
    width: SymbolicInt, modulus: SymbolicInt, window: SymbolicInt, stage: int | None
) -> Any:
    from qualtran.bloqs.cryptography.ecc import ec_add

    if stage is None:
        return ec_add.ECAdd(width, modulus, window)
    name = ("One", "Two", "Three", "Four", "Five", "Six")[stage - 1]
    cls = getattr(ec_add, "_ECAddStep" + name)
    return cls(width, modulus, **({} if stage in (1, 6) else {"window_size": window}))


@pytest.mark.parametrize(
    "profile",
    (
        "small",
        "wide",
        "symbolic_width",
        "symbolic_window",
        "all_symbolic",
        "symbolic_constant",
    ),
)
@pytest.mark.parametrize("stage", (None, 1, 2, 3, 4, 5, 6))
def test_complete_source_calls_parameters_directions_and_symbolic_aggregation(
    profile: str, stage: int | None
) -> None:
    from qualtran.resource_counting import SympySymbolAllocator

    n, p, w = sympy.symbols("n p w", positive=True, integer=True)
    profiles = {
        "small": (3, 3, 1),
        "wide": (8, 17, 4),
        "symbolic_width": (n, 17, 1),
        "symbolic_window": (8, 17, w),
        "all_symbolic": (n, p, w),
        "symbolic_constant": (sympy.Integer(3), 3, 1),
    }
    width, modulus, window = profiles[profile]
    source = original(width, modulus, window, stage)
    expected = {
        source_key(child): count
        for child, count in source.build_call_graph(SympySymbolAllocator()).items()
    }
    actual = elliptic_curve_addition_callees(width, modulus, window, stage=stage)
    assert list(actual.items()) == list(expected.items())
    graph = elliptic_curve_addition_parameterized_graph(
        width, modulus, window, stage=stage
    )
    assert graph.root == source_key(source)
    assert {
        child: value
        for (parent, child), value in graph.direct_calls.items()
        if parent == graph.root
    } == expected
    leaves: Counter[Hashable] = Counter()
    if stage is None:
        for child, repetitions in source.build_call_graph(
            SympySymbolAllocator()
        ).items():
            for leaf, count in child.build_call_graph(SympySymbolAllocator()).items():
                key = source_key(leaf)
                leaves[key] += repetitions * count
    else:
        leaves.update(expected)
    assert graph.leaf_counts == dict(leaves)
    # Independent symbolic prices exercise the common aggregator, without
    # inventing a physical gate estimate for opaque Montgomery children.
    prices = {key: sympy.Symbol(f"price_{i}") for i, key in enumerate(leaves)}
    assert (
        sympy.expand(
            graph.estimate_leaf_cost(prices.__getitem__)
            - sum(prices[key] * count for key, count in leaves.items())
        )
        == 0
    )


def test_existing_operation_consumes_parameterized_graph_without_sdk_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    for name in tuple(sys.modules):
        if name.split(".", 1)[0] in {"qualtran", "cirq", "kickmix"}:
            monkeypatch.setitem(sys.modules, name, None)
    operation = EllipticCurveAddition(3, 3)
    assert operation.source_call_graph() == elliptic_curve_addition_parameterized_graph(
        3, 3
    )
    assert len(operation.source_call_graph().direct_calls) > 6


@pytest.mark.parametrize("stage", (None, 1, 2, 3, 4, 5, 6))
@pytest.mark.parametrize("symbolic", (False, True))
def test_declared_aggregation_with_explicit_original_leaf_prices(
    stage: int | None, symbolic: bool
) -> None:
    from qualtran.resource_counting import (
        QECGatesCost,
        SympySymbolAllocator,
        get_cost_value,
    )

    n, p = sympy.symbols("n p", positive=True, integer=True)
    width, modulus = (n, p) if symbolic else (3, 3)
    source = original(width, modulus, 1, stage)
    graph = elliptic_curve_addition_parameterized_graph(width, modulus, stage=stage)
    raw = source.build_call_graph(SympySymbolAllocator())
    leaves = (
        [
            leaf
            for child in raw
            for leaf in child.build_call_graph(SympySymbolAllocator())
        ]
        if stage is None
        else list(raw)
    )
    prices = {source_key(leaf): get_cost_value(leaf, QECGatesCost()) for leaf in leaves}
    expected = get_cost_value(source, QECGatesCost())
    for field in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        metric_prices: dict[Hashable, SymbolicInt] = {
            key: getattr(value, field) for key, value in prices.items()
        }
        actual = graph.estimate_leaf_cost(metric_prices.__getitem__)
        assert sympy.expand(actual - getattr(expected, field)) == 0
    # These rates are explicitly supplied reference inputs, not an independent
    # native default Montgomery pricer or measured primitive instruction counts.


@pytest.mark.parametrize("stage", (0, 7, True))
def test_invalid_stage_is_not_a_silent_root_fallback(stage: int) -> None:
    with pytest.raises(ValueError, match="stage must be an integer"):
        elliptic_curve_addition_parameterized_graph(3, 3, stage=stage)
