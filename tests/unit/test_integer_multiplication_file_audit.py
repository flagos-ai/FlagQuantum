"""Compare all frozen multiplication examples to native metadata and graphs."""

from __future__ import annotations

import hashlib
from collections.abc import Hashable
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

from examples.algorithms.integer_multiplication_catalog import (
    CX,
    TOFFOLI,
    T,
    X,
    integer_multiplication_example_graph,
    integer_multiplication_examples,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
SOURCE = (
    Path(__file__).resolve().parents[3] / "ftqc-reference-env/qualtran/bloqs/arithmetic"
)


def test_locked_assets() -> None:
    import kickmix

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    for name, sha in (
        ("multiplication.py", "efed2d6652c7dcc943c82e0c5d35ae5b3b627c09"),
        ("multiplication_test.py", "4bc31a09836b04d642bdcc2f2420b06a4f8cac7e"),
        ("multiplication.ipynb", "40eff7a98107a9d0e83696579695459d011bfe17"),
    ):
        data = (SOURCE / name).read_bytes()
        assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == sha


def _source_key(child: Any) -> Hashable:
    from qualtran.bloqs.arithmetic import Subtract
    from qualtran.bloqs.arithmetic.multiplication import (
        MultiplyTwoReals,
        ScaleIntByReal,
        SquareRealNumber,
    )
    from qualtran.bloqs.basic_gates import CNOT, TGate, Toffoli, XGate
    from qualtran.bloqs.mcmt import MultiControlX

    if isinstance(child, Toffoli):
        return TOFFOLI
    if isinstance(child, TGate):
        return T
    if isinstance(child, CNOT):
        return CX
    if isinstance(child, XGate):
        return X
    if isinstance(child, SquareRealNumber):
        return ("fixed_point_square", child.bitsize)
    if isinstance(child, MultiplyTwoReals):
        return ("fixed_point_product", child.bitsize)
    if isinstance(child, ScaleIntByReal):
        return ("fixed_point_scaling", child.r_bitsize, child.i_bitsize)
    if isinstance(child, Subtract):
        return ("unsigned_subtraction", child.a_dtype.bitsize)
    if isinstance(child, MultiControlX):
        return ("multi_control_x", tuple(child.cvs))
    raise AssertionError(f"Unmapped source child {child}")


@pytest.mark.parametrize(
    "index,factory",
    list(
        enumerate(
            (
                "_plus_equal_product",
                "_product",
                "_square",
                "_sum_of_squares",
                "_scale_int_by_real",
                "_multiply_two_reals",
                "_square_real_number",
                "_invert_real_number",
            )
        )
    ),
)
def test_complete_factory_signatures_and_graphs(index: int, factory: str) -> None:
    from qualtran.bloqs.arithmetic import multiplication
    from qualtran.resource_counting.generalizers import ignore_split_join

    source = getattr(multiplication, factory).make()
    example = integer_multiplication_examples()[index]
    name, signature, _ = example
    assert [(r.name, r.bitsize, r.shape, r.side.value) for r in signature] == [
        (r.name, r.bitsize, r.shape, r.side.value) for r in source.signature
    ]
    assert [
        (r.dtype.encoding, dict(r.dtype.parameters).get("fractional_width"))
        for r in signature
    ] == [
        (
            ("fixed_point", r.dtype.num_frac)
            if hasattr(r.dtype, "num_frac")
            else ("uint", None)
        )
        for r in source.signature
    ]
    graph, leaves = source.call_graph(max_depth=1, generalizer=ignore_split_join)
    native = integer_multiplication_example_graph(example)
    assert native.direct_calls == {
        (name, _source_key(child)): edge["n"]
        for _, child, edge in graph.edges(data=True)
    }
    assert native.leaf_counts == {
        _source_key(child): count for child, count in leaves.items()
    }
