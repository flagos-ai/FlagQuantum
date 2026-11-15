"""Frozen trigonometric declarations and source expression parity."""

from __future__ import annotations

from hashlib import sha1
from importlib.metadata import version
from pathlib import Path

import pytest
from sympy import Symbol, symbols

from examples.algorithms.trigonometric_catalog import (
    arcsine_example,
    arctangent_example,
    trigonometric_example_graph,
)
from flagquantum.compiler.registers.fixed_point.trigonometric_resource_model import (
    ARCTANGENT_PRODUCT_CALLS,
    source_arcsine_toffoli_count,
    source_arctangent_t_count,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
SOURCE = (
    Path(__file__).resolve().parents[3]
    / "ftqc-reference-env/qualtran/bloqs/arithmetic/trigonometric"
)


def test_locked_entire_source_family() -> None:
    import kickmix

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    for name, digest in (
        ("__init__.py", "4fcdb07c7c0593d2eca3adbd64ab7d971ebf6f41"),
        ("arcsin.py", "2405e65efa8b2e315860b8c8c6724b3b5d78010b"),
        ("arcsin_test.py", "e1ccdc52bdacd083665ed78a1ff7b3421a0be752"),
        ("arctan.py", "746411d9f44fa95c5209f5575419a99749937e99"),
        ("arctan_test.py", "a923a214a6c40e33bb0bc3476ff0143176878ca2"),
        ("trigonometric.ipynb", "6caf4979e9bd653ae67c94af7706049b23ee3719"),
    ):
        data = (SOURCE / name).read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest


def test_arcsine_complete_original_factory_signature_and_graph() -> None:
    from qualtran import QFxp
    from qualtran.bloqs.arithmetic.trigonometric.arcsin import _arcsin
    from qualtran.bloqs.basic_gates import Toffoli

    source = _arcsin.make()
    name, signature, _ = example = arcsine_example()
    assert all(isinstance(r.dtype, QFxp) for r in source.signature)
    assert [
        (r.name, r.bitsize, r.side.value, dict(r.dtype.parameters)["fractional_width"])
        for r in signature
    ] == [(r.name, r.bitsize, r.side.value, r.dtype.num_frac) for r in source.signature]
    graph, counts = source.call_graph(max_depth=1)
    actual = trigonometric_example_graph(example)
    assert all(isinstance(child, Toffoli) for child in counts)
    assert actual.leaf_counts == {"toffoli": counts[Toffoli()]}
    assert actual.direct_calls == {
        (name, "toffoli"): e["n"] for _, _, e in graph.edges(data=True)
    }


@pytest.mark.parametrize("iterations", [3, 4, 5])
@pytest.mark.parametrize("degree", [3, 4, 5, 6])
@pytest.mark.parametrize("width,fraction", [(4, 2), (10, 7), (16, 16)])
def test_arcsine_original_resource_parameter_matrix(
    width: int, fraction: int, iterations: int, degree: int
) -> None:
    from qualtran.bloqs.arithmetic.trigonometric import ArcSin
    from qualtran.bloqs.basic_gates import Toffoli

    source = ArcSin(width, fraction, num_iters=iterations, degree=degree)
    assert (
        source_arcsine_toffoli_count(
            width, fraction, iterations=iterations, degree=degree
        )
        == source.bloq_counts()[Toffoli()]
    )


def test_arcsine_unrestricted_symbolic_count() -> None:
    from qualtran.bloqs.arithmetic.trigonometric import ArcSin
    from qualtran.bloqs.basic_gates import Toffoli

    n, f, m, d = symbols("n f m d")
    source = ArcSin(n, f, num_iters=m, degree=d)
    assert (
        source_arcsine_toffoli_count(n, f, iterations=m, degree=d)
        == source.bloq_counts()[Toffoli()]
    )


def test_arctangent_symbolic_signature_complete_graph_and_legacy_cost() -> None:
    from qualtran import QAny, QBit
    from qualtran.bloqs.arithmetic import PlusEqualProduct
    from qualtran.bloqs.arithmetic.trigonometric import ArcTan

    n = Symbol("n")
    source = ArcTan(4, n)
    name, signature, _ = example = arctangent_example(4, n)
    assert tuple(type(r.dtype) for r in source.signature) == (QAny, QBit, QAny)
    assert tuple(r.dtype.encoding for r in signature) == ("opaque", "bit", "opaque")
    assert [(r.name, r.bitsize, r.side.value) for r in signature] == [
        (r.name, r.bitsize, r.side.value) for r in source.signature
    ]
    graph, counts = source.call_graph(max_depth=1)
    child = PlusEqualProduct(n, n, 2 * n)
    actual = trigonometric_example_graph(example)
    key = ("plus_equal_product", n, n, 2 * n)
    assert actual.direct_calls == {
        (name, key): e["n"] for _, _, e in graph.edges(data=True)
    }
    assert actual.leaf_counts == {key: counts[child]}
    assert (
        source_arctangent_t_count(n, ARCTANGENT_PRODUCT_CALLS)
        == source.t_complexity().t
    )
    with pytest.raises(TypeError, match="symbolic bitsizes"):
        source.registers()
    with pytest.raises(NotImplementedError):
        ArcTan(4, 3).with_registers((2,), (2,), (2,))


@pytest.mark.parametrize("source_width", [3, 4])
@pytest.mark.parametrize("target_width", [3, 5, 6])
def test_all_original_arctangent_width_pairs_and_occupied_outputs(
    source_width: int, target_width: int
) -> None:
    from qualtran.bloqs.arithmetic.trigonometric import ArcTan

    from flagquantum.compiler.registers.lookup.arctan_lookup import (
        arctan_lookup_ir,
        arctan_lookup_requirements,
    )
    from tests.unit.measurement_checks import certified_and_channel_map

    source = ArcTan(source_width, target_width)
    assert source.adjoint() is source
    assert source.registers() == ((2,) * source_width, (2,), (2,) * target_width)
    assert len(source.signature) == 3
    work, slots = arctan_lookup_requirements(source_width, target_width)
    logical_width = source_width + 1 + target_width
    gates = arctan_lookup_ir(
        tuple(range(source_width)),
        source_width,
        tuple(range(source_width + 1, logical_width)),
        tuple(range(logical_width, logical_width + work)),
        tuple(range(slots)),
    )

    def pack(x: int, sign: int, target: int) -> tuple[int, ...]:
        word = (((x << 1) | sign) << target_width) | target
        return (
            tuple((word >> i) & 1 for i in reversed(range(logical_width))) + (0,) * work
        )

    before = []
    after = []
    for x in range(1 << source_width):
        for sign, y in ((0, 0), (1, (1 << target_width) - 1)):
            before.append(pack(x, sign, y))
            after.append(pack(*source.apply(x, sign, y)))
    events = sum(gate.name == "measure" for gate in gates)
    assert certified_and_channel_map(gates, tuple(before)) == (tuple(after), events)
    assert certified_and_channel_map(gates, tuple(after)) == (tuple(before), events)
