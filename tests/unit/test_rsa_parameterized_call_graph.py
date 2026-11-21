"""Compare complete RSA model identities and factor generalization to the source."""

from collections import Counter
from collections.abc import Hashable
from typing import TYPE_CHECKING

import pytest

from flagquantum.algorithms.primitives.rsa_phase_estimation import (
    RSAPhaseEstimation,
    generalize_rsa_multiplier_call,
)
from flagquantum.compiler._component_trace.call_graph import (
    SymbolAllocator,
    symbolic_call_graph,
)

if TYPE_CHECKING:
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import CModMulK

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def multiplier_identity(operation: "CModMulK") -> tuple[object, ...]:
    return ("CModMulK", operation.dtype.bitsize, operation.k, operation.mod)


def test_source_symbols_constructor_examples_and_allocator_ownership() -> None:
    from qualtran.bloqs.cryptography.rsa.rsa_phase_estimate import (
        _rsa_pe,
        _rsa_pe_small,
    )
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import CModMulK
    from qualtran.resource_counting import SympySymbolAllocator

    for source in (_rsa_pe(), _rsa_pe_small()):
        native = RSAPhaseEstimation(source.n, source.mod, source.base)
        reference_symbols, native_symbols = SympySymbolAllocator(), SymbolAllocator()
        assert reference_symbols.new_symbol("k") == native_symbols.new_symbol("k")
        expected = {
            (
                multiplier_identity(child)
                if isinstance(child, CModMulK)
                else (type(child).__name__, child.n)
            ): count
            for child, count in source.build_call_graph(reference_symbols).items()
        }
        assert native.source_callees(native_symbols) == expected
        assert native.source_call_graph().root == native
        assert native.source_call_graph() == native.source_call_graph()


def test_source_generalization_combines_actual_composite_multipliers() -> None:
    from qualtran.bloqs.cryptography.rsa.rsa_phase_estimate import (
        RSAPhaseEstimate,
        _generalize_k,
    )
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import CModMulK

    source = RSAPhaseEstimate(4, 15, 2)
    children = [
        instance.bloq
        for instance, _, _ in source.decompose_bloq().iter_bloqsoqs()
        if isinstance(instance.bloq, CModMulK)
    ]
    expected = Counter(multiplier_identity(_generalize_k(child)) for child in children)
    calls = Counter(multiplier_identity(child) for child in children)
    declarations: dict[Hashable, dict[Hashable, int]] = {
        "physical multiplier calls": {node: count for node, count in calls.items()}
    }
    root: Hashable = "physical multiplier calls"
    graph = symbolic_call_graph(
        root,
        declarations,
        generalizer=generalize_rsa_multiplier_call,
    )
    assert graph.leaf_counts == expected
    assert list(expected.values()) == [8]
    for node in (
        ("MeasureQFT", 4),
        ("OtherMultiplier", 4, 2, 15),
        "physical multiplier calls",
    ):
        assert generalize_rsa_multiplier_call(node) == node
    qft = next(
        instance.bloq
        for instance, _, _ in source.decompose_bloq().iter_bloqsoqs()
        if type(instance.bloq).__name__ == "MeasureQFT"
    )
    assert _generalize_k(qft) is qft


def test_generalized_resource_consumers_keep_modulus_and_width() -> None:
    from qualtran import QUInt
    from qualtran.bloqs.cryptography.rsa.rsa_phase_estimate import _generalize_k
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import CModMulK

    sources = [
        CModMulK(QUInt(n), k=k, mod=p)
        for n, p, k in ((4, 15, 2), (4, 15, 4), (5, 21, 2), (5, 21, 4))
    ]
    counts = Counter(multiplier_identity(source) for source in sources)
    declarations: dict[Hashable, dict[Hashable, int]] = {
        "root": {node: count for node, count in counts.items()}
    }
    root: Hashable = "root"
    graph = symbolic_call_graph(
        root, declarations, generalizer=generalize_rsa_multiplier_call
    )
    assert graph.leaf_counts == Counter(
        multiplier_identity(_generalize_k(source)) for source in sources
    )
    assert len(graph.leaf_counts) == 2

    def cost(node: Hashable) -> int:
        assert isinstance(node, tuple)
        return int(node[1]) * int(node[3])

    assert graph.estimate_leaf_cost(cost) == 330
