"""Complete frozen Kikuchi matrix-entry declarations and direct dependencies."""

from dataclasses import replace
from hashlib import sha1
from pathlib import Path

import pytest
import sympy
from qualtran import DecomposeNotImplementedError
from qualtran.bloqs.optimization.k_xor_sat import kikuchi_adjacency_matrix as source
from qualtran.resource_counting import (
    QECGatesCost,
    SympySymbolAllocator,
    get_cost_value,
)

from flagquantum.algorithms.kikuchi_matrix import (
    example_kikuchi_matrix_entry,
    symbolic_kikuchi_matrix_entry,
)

_BLOBS = {
    "kikuchi_adjacency_matrix.py": "fbd3829978d36d1fccd179f979af2b5b67d90fe9",
    "kikuchi_adjacency_matrix_test.py": "af1fdd8bdc03ff1bded9c87467ab976138caadb9",
}

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_source_and_original_tests_are_frozen() -> None:
    assert source.__file__ is not None
    directory = Path(source.__file__).parent
    for name, blob in _BLOBS.items():
        raw = (directory / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize("symbolic", (False, True))
def test_complete_factories_signatures_child_operands_and_both_graph_directions(
    symbolic: bool,
) -> None:
    native = (
        symbolic_kikuchi_matrix_entry() if symbolic else example_kikuchi_matrix_entry()
    )
    original = (
        source._kikuchi_matrix_entry_symb()
        if symbolic
        else source._kikuchi_matrix_entry()
    )
    assert (
        native.ell == original.ell
        and native.entry_bitsize == original.entry_bitsize == 3
    )
    assert (
        native.inst.n == original.inst.n
        and native.inst.m == original.inst.m
        and native.inst.k == original.inst.k
    )
    assert native.composite_index_bitsize == original.composite_index_bitsize
    assert native.index_dtype.num_qubits == original.index_dtype.num_qubits
    assert (
        native.rotation_angle_dtype.num_qubits
        == original.rotation_angle_dtype.num_qubits
    )
    assert (
        dict(native.rotation_angle_dtype.parameters)["fractional_width"]
        == original.rotation_angle_dtype.num_frac
    )
    assert [(r.name, r.bitsize, r.shape, r.side.name) for r in native.signature] == [
        (r.name, r.bitsize, r.shape, r.side.name) for r in original.signature
    ]
    calls = original.build_call_graph(SympySymbolAllocator())
    normalized = {}
    for child, count in calls.items():
        if type(child).__name__ == "Always":
            inner = child.subbloq
            adjoint = type(inner).__name__ == "Adjoint"
            body = inner.subbloq if adjoint else inner
            if type(body).__name__ == "SymmetricDifference":
                role = "diff_inverse" if adjoint else "diff"
                assert body.n_lhs == body.n_rhs == native.ell
                assert body.n_diff == native.inst.k
                assert body.dtype.num_qubits == native.index_dtype.num_qubits
            else:
                assert type(body).__name__ == "LoadUniqueScopeIndex"
                role = "index_inverse" if adjoint else "index"
                assert body.inst == original.inst
                assert native.scope_index.inst == native.inst
        else:
            assert type(child).__name__ == "PRGAUniqueConstraintRHS"
            role = "rhs"
            assert (
                child.inst == original.inst
                and child.angle_bitsize == native.entry_bitsize
            )
        normalized[role] = count
    graph = native.source_graph(
        root="matrix",
        difference="diff",
        inverse_difference="diff_inverse",
        index="index",
        inverse_index="index_inverse",
        rotation="rhs",
    )
    assert (
        graph.leaf_counts
        == normalized
        == dict.fromkeys(("diff", "diff_inverse", "index", "index_inverse", "rhs"), 1)
    )
    with pytest.raises(DecomposeNotImplementedError):
        original.decompose_bloq()


@pytest.mark.parametrize("symbolic", (False, True))
@pytest.mark.parametrize("controlled", (False, True))
def test_full_dependency_qec_categories_and_control_bypass(
    symbolic: bool, controlled: bool
) -> None:
    native = (
        symbolic_kikuchi_matrix_entry() if symbolic else example_kikuchi_matrix_entry()
    )
    original = (
        source._kikuchi_matrix_entry_symb()
        if symbolic
        else source._kikuchi_matrix_entry()
    )
    queried = original.controlled() if controlled else original
    model = native.resource_gates(controlled=controlled)
    expected = get_cost_value(queried, QECGatesCost())
    for field in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        assert (
            sympy.simplify(getattr(model, field) - getattr(expected, field)) == 0
        ), field
    if not symbolic and not controlled:
        assert model.cswap == 512 and model.and_bloq == model.measurement == 1301
        assert model.clifford == 12518
    # Original Always wrappers bypass controls on both compute/uncompute pairs.
    _, plain = original.call_graph(max_depth=2)
    _, positive = original.controlled().call_graph(max_depth=2)
    removed, added = set(plain.items()) - set(positive.items()), set(
        positive.items()
    ) - set(plain.items())
    assert len(removed) == len(added) == 1
    ((qrom, count),), ((controlled_qrom, controlled_count),) = removed, added
    assert count == controlled_count
    assert native.rhs().amplitude_qrom.num_controls == qrom.num_controls == 0
    assert (
        native.rhs(controlled=True).amplitude_qrom.num_controls
        == controlled_qrom.num_controls
        == 1
    )
    assert (
        replace(native.rhs().amplitude_qrom, num_controls=1)
        == native.rhs(controlled=True).amplitude_qrom
    )
