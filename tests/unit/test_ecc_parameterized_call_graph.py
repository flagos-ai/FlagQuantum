"""Preserve ECC source parameter identities through two levels of modeling."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import pytest

from flagquantum.algorithms.primitives.elliptic_curve_double_phase import (
    EllipticCurveDoublePhase,
)
from flagquantum.algorithms.primitives.elliptic_curve_phase_estimation import (
    EllipticCurvePhaseEstimation,
)
from flagquantum.algorithms.primitives.elliptic_curve_point import ECPoint
from flagquantum.compiler._component_trace.call_graph import SymbolAllocator

if TYPE_CHECKING:
    from qualtran.bloqs.cryptography._factoring_shims import MeasureQFT
    from qualtran.bloqs.cryptography.ecc.ec_add_r import ECAddR, ECWindowAddR
    from qualtran.bloqs.cryptography.ecc.ec_phase_estimate_r import ECPhaseEstimateR

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def native_identity(node: Hashable) -> tuple[object, ...]:
    if isinstance(node, tuple):
        return node
    if isinstance(node, EllipticCurvePhaseEstimation):
        return (
            "ECPhaseEstimateR",
            node.width,
            node.point,
            node.window,
            node.multiplication_window,
        )
    raise AssertionError(node)


def source_identity(
    node: ECAddR | ECWindowAddR | ECPhaseEstimateR | MeasureQFT,
) -> tuple[object, ...]:
    from qualtran.bloqs.cryptography._factoring_shims import MeasureQFT
    from qualtran.bloqs.cryptography.ecc.ec_add_r import ECAddR, ECWindowAddR

    name = type(node).__name__
    if isinstance(node, MeasureQFT):
        return (name, node.n)
    point = node.R if isinstance(node, (ECAddR, ECWindowAddR)) else node.point
    point = ECPoint(point.x, point.y, point.mod, point.curve_a)
    if isinstance(node, ECAddR):
        return (name, node.n, point)
    return (name, node.n, point, node.add_window_size, node.mul_window_size)


@pytest.mark.parametrize("width,window", [(3, 1), (3, 3), (4, 2), (5, 2), (6, 3)])
@pytest.mark.parametrize("multiplication_window", [1, 2])
def test_phase_calls_preserve_exact_source_parameters(
    width: int, window: int, multiplication_window: int
) -> None:
    from qualtran.bloqs.cryptography.ecc.ec_phase_estimate_r import ECPhaseEstimateR
    from qualtran.bloqs.cryptography.ecc.ec_point import ECPoint as SourcePoint
    from qualtran.resource_counting import SympySymbolAllocator
    from sympy import symbols

    x, y, modulus, coefficient = symbols("x y p a")
    native = EllipticCurvePhaseEstimation(
        width, ECPoint(x, y, modulus, coefficient), window, multiplication_window
    )
    source = ECPhaseEstimateR(
        width, SourcePoint(x, y, modulus, coefficient), window, multiplication_window
    )
    expected = {
        source_identity(child): count
        for child, count in source.build_call_graph(SympySymbolAllocator()).items()
    }
    graph = native.source_call_graph()
    assert {
        native_identity(child): count for child, count in graph.leaf_counts.items()
    } == expected
    assert graph.direct_calls == {
        (native, child): count for child, count in native.source_callees().items()
    }
    # A consumer can price windows independently without losing constructor parameters.
    assert (
        graph.estimate_leaf_cost(
            lambda child: (
                child[4]
                if isinstance(child, tuple) and child[0] == "ECWindowAddR"
                else 1
            )
        )
        == (width // window) * (multiplication_window if window != 1 else 1) + 1
    )


@pytest.mark.parametrize("window", [1, 2])
@pytest.mark.parametrize("multiplication_window", [1, 3])
def test_double_phase_generic_identity_and_recursive_counts(
    window: int, multiplication_window: int
) -> None:
    from qualtran.bloqs.cryptography.ecc.ec_point import ECPoint as SourcePoint
    from qualtran.bloqs.cryptography.ecc.find_ecc_private_key import FindECCPrivateKey
    from qualtran.resource_counting import SympySymbolAllocator
    from sympy import symbols

    modulus, coefficient = symbols("p a")
    source = FindECCPrivateKey(
        4,
        SourcePoint(1, 2, modulus, coefficient),
        SourcePoint(3, 4, modulus, coefficient),
        window,
        multiplication_window,
    )
    native = EllipticCurveDoublePhase(
        4,
        ECPoint(1, 2, modulus, coefficient),
        ECPoint(3, 4, modulus, coefficient),
        window,
        multiplication_window,
    )
    source_allocator, native_allocator = SympySymbolAllocator(), SymbolAllocator()
    # Preallocated names detect incorrect resetting of the caller's allocator.
    assert source_allocator.new_symbol("Rx") == native_allocator.new_symbol("Rx")
    expected = source.build_call_graph(source_allocator)
    actual = native.source_callees(native_allocator)
    assert {native_identity(child): count for child, count in actual.items()} == {
        source_identity(child): count for child, count in expected.items()
    }
    assert native.cost_attrs() == source.cost_attrs()
    source_phase = next(iter(source.build_call_graph(SympySymbolAllocator())))
    expected_leaves = {
        source_identity(child): 2 * count
        for child, count in source_phase.build_call_graph(
            SympySymbolAllocator()
        ).items()
    }
    graph = native.source_call_graph()
    assert {
        native_identity(child): count for child, count in graph.leaf_counts.items()
    } == expected_leaves
    assert native.source_call_graph() == graph
    generic_phase = next(
        child for parent, child in graph.direct_calls if parent == native
    )
    assert isinstance(generic_phase, EllipticCurvePhaseEstimation)
    assert generic_phase.point not in (native.base_point, native.public_key)
