# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Assertions adapted from frozen Qualtran v0.7.0 gf_poly_arithmetic sources.

"""Whole-file polynomial examples, view tensors, source graphs and append checks."""

from __future__ import annotations

import ast
import csv
import hashlib
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.field_addition_catalog import field_signature_svg
from examples.algorithms.field_polynomial_catalog import (
    polynomial_addition_examples,
    polynomial_example_graph,
    polynomial_view_examples,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.resource_generalization import CliffordCall
from flagquantum.compiler.registers.finite_field.field_polynomial_addition import (
    gf2_polynomial_add_constant_ir,
    gf2_polynomial_add_ir,
)
from flagquantum.compiler.registers.finite_field.field_polynomial_layout import (
    FieldPolynomialLayout,
)
from flagquantum.simulation.finite_field_addition import binary_field_encoding
from flagquantum.simulation.finite_field_polynomial_encoding import (
    FiniteFieldPolynomialEncoding,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env"


@pytest.fixture(scope="module", autouse=True)
def complete_locked_assets() -> None:
    import kickmix

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    with (REFERENCE.parent / "FlagQuantum-Qualtran-Kickmix-移植进度.csv").open(
        encoding="utf-8-sig"
    ) as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["编号"] in {f"Q-{i:04d}" for i in range(526, 535)}
        ]
    assert len(rows) == 9
    for row in rows:
        data = (REFERENCE / row["源文件"]).read_bytes()
        assert (
            hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            == row["源对象"]
        )


def test_complete_original_test_inventory() -> None:
    expected = {
        "gf2_poly_add_test.py": {
            "test_gf2_poly_4_8_add",
            "test_gf2_poly_symbolic_add_k",
            "test_gf2_poly_add_classical_sim",
            "test_gf2_poly_add_resource",
        },
        "gf2_poly_add_k_test.py": {
            "test_gf2_poly_4_8_add_k",
            "test_gf2_poly_symbolic_add_k",
            "test_gf2_poly_add_k_resource",
            "test_gf2_poly_add_k_classical_sim",
        },
        "gf_poly_split_and_join_test.py": {
            "test_gf_poly_split",
            "test_gf_poly_join",
            "test_no_symbolic_degree",
            "test_classical_sim",
            "test_tensor_sim",
        },
    }
    for name, functions in expected.items():
        tree = ast.parse(
            (REFERENCE / "qualtran/bloqs/gf_poly_arithmetic" / name).read_text()
        )
        assert {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        } == functions


def _source_key(node: Any, root: Any, name: str) -> Any:
    # External object normalization is test-only. Production has no foreign factory.
    from qualtran.bloqs.basic_gates import CNOT, XGate
    from qualtran.bloqs.gf_arithmetic import GF2Addition, GF2AddK
    from qualtran.bloqs.gf_poly_arithmetic.gf_poly_split_and_join import (
        GFPolyJoin,
        GFPolySplit,
    )

    if node == root:
        return name
    if isinstance(node, GFPolySplit):
        return "Polynomial split"
    if isinstance(node, GFPolyJoin):
        return "Polynomial join"
    if isinstance(node, GF2Addition):
        return node.bitsize, None
    if isinstance(node, GF2AddK):
        return node.bitsize, node.k
    if isinstance(node, CNOT):
        return CliffordCall("cx", 2)
    if isinstance(node, XGate):
        return CliffordCall("x", 1)
    raise AssertionError(f"Unmapped original graph node {node}")


@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize("depth", [1, None])
def test_all_factories_full_signatures_graphs_and_costs(
    constant: bool, depth: int | None
) -> None:
    from qualtran.bloqs.gf_poly_arithmetic.gf2_poly_add import (
        _gf2_poly_4_8_add,
        _gf2_poly_add_symbolic,
    )
    from qualtran.bloqs.gf_poly_arithmetic.gf2_poly_add_k import (
        _gf2_poly_4_8_add_k,
        _gf2_poly_add_k_symbolic,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.resource_counting.generalizers import ignore_split_join

    factories = (
        (_gf2_poly_4_8_add_k, _gf2_poly_add_k_symbolic)
        if constant
        else (_gf2_poly_4_8_add, _gf2_poly_add_symbolic)
    )
    examples = polynomial_addition_examples(constant=constant)
    for example, factory in zip(examples, factories, strict=True):
        name, signature, _ = example
        original = factory.make()
        assert [(r.name, r.bitsize, r.shape, r.side.value) for r in signature] == [
            (r.name, r.bitsize, r.shape, r.side.value) for r in original.signature
        ]
        encoding = FiniteFieldPolynomialEncoding(
            original.qgf_poly.degree,
            binary_field_encoding(original.qgf_poly.qgf.degree),
        )
        assert all(r.dtype == encoding.register_type() for r in signature)
        graph, leaves = original.call_graph(
            max_depth=depth, generalizer=ignore_split_join
        )
        native = polynomial_example_graph(example, max_depth=depth)
        expected_edges = {
            (
                _source_key(left, original, name),
                _source_key(right, original, name),
            ): data["n"]
            for left, right, data in graph.edges(data=True)
        }
        assert native.direct_calls == expected_edges
        assert native.leaf_counts == {
            _source_key(node, original, name): count for node, count in leaves.items()
        }
        expanded = polynomial_example_graph(example, max_depth=None)
        physical_model = sum(
            count
            for node, count in expanded.leaf_counts.items()
            if isinstance(node, CliffordCall)
        )
        assert physical_model == get_cost_value(original, QECGatesCost()).clifford
    from xml.etree import ElementTree

    ElementTree.fromstring(
        field_signature_svg([(name, signature, {}) for name, signature, _ in examples])
    )


def test_complete_split_join_factories_and_classical_assertions() -> None:
    import galois
    from qualtran.bloqs.gf_poly_arithmetic.gf_poly_split_and_join import (
        _gf_poly_join,
        _gf_poly_split,
    )

    for (name, signature), factory in zip(
        polynomial_view_examples(), (_gf_poly_split, _gf_poly_join), strict=True
    ):
        original = factory.make()
        assert [(r.name, r.bitsize, r.shape, r.side.value) for r in signature] == [
            (r.name, r.bitsize, r.shape, r.side.value) for r in original.signature
        ]
        encoding = FiniteFieldPolynomialEncoding(4, binary_field_encoding(3))
        field = encoding.field.gf_type
        polynomial = galois.Poly(field([1, 2, 3, 4]))
        coefficients = encoding.to_gf_coefficients(polynomial)
        assert coefficients.tolist() == [0, 1, 2, 3, 4]
        assert encoding.from_gf_coefficients(coefficients) == polynomial
        reference = original.call_classically(
            reg=polynomial if name.endswith("split") else coefficients
        )[0]
        if name.endswith("split"):
            np.testing.assert_array_equal(coefficients, reference)
        else:
            assert polynomial == reference
        from xml.etree import ElementTree

        ElementTree.fromstring(field_signature_svg([(name, signature, {})]))


@pytest.mark.parametrize("join", [False, True])
def test_original_full_view_tensor_and_native_identity(join: bool) -> None:
    from qualtran import QGF, QGFPoly
    from qualtran.bloqs.gf_poly_arithmetic.gf_poly_split_and_join import GFPolySplit

    original = GFPolySplit(QGFPoly(2, QGF(2, 2)))
    if join:
        original = original.adjoint()
    layout = FieldPolynomialLayout(2, 2)
    for index in range(64):
        assert layout.pack_basis(layout.unpack_basis(index)) == index
    np.testing.assert_array_equal(original.tensor_contract(), np.eye(64))
    state = torch.arange(1, 65, dtype=torch.float64) + 1j * torch.arange(
        64, dtype=torch.float64
    )
    state /= state.norm()
    actual = (
        Circuit(6, inputs=state, dtype=torch.complex128)
        .append(FieldOperation(6, ()))
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, state, rtol=0, atol=1e-12)


@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize("degree,width", [(0, 1), (1, 2), (2, 1)])
def test_append_noncontiguous_complex_words_and_clean_wire(
    constant: bool, degree: int, width: int
) -> None:
    layout = FieldPolynomialLayout(degree, width)
    size = layout.logical_width
    source, target = tuple(range(1, 2 * size + 1, 2)), tuple(range(2, 2 * size + 1, 2))
    n = 2 * size + 2
    state = torch.arange(
        1, 1 + (1 << (n - 1)), dtype=torch.float64
    ) + 1j * torch.arange(1 << (n - 1), dtype=torch.float64)
    state /= state.norm()
    state = torch.stack((state, torch.zeros_like(state)), dim=1).flatten()
    coefficients = tuple(((1 << width) - 1) for _ in range(degree + 1))
    gates = (
        gf2_polynomial_add_constant_ir(target, coefficients, layout)
        if constant
        else gf2_polynomial_add_ir(source, target, layout)
    )
    indices = torch.arange(1 << n)
    permutation = indices.clone()
    for left, right in zip(source, target, strict=True):
        bit = 1 if constant else ((indices >> (n - 1 - left)) & 1)
        permutation ^= bit << (n - 1 - right)
    operation = FieldOperation(n, gates)
    circuit = Circuit(n, inputs=state, dtype=torch.complex128).append(operation)
    torch.testing.assert_close(
        circuit.state().flatten(), state[permutation], rtol=0, atol=1e-12
    )
    assert torch.count_nonzero(circuit.state().flatten()[1::2]) == 0
    inverse = circuit.adjoint().to_ir().instructions
    assert [(g.name, g.wires) for g in inverse] == [
        (g.name, g.wires) for g in reversed(gates)
    ]
    inverse_gates = (
        gf2_polynomial_add_constant_ir(target, coefficients, layout, adjoint=True)
        if constant
        else gf2_polynomial_add_ir(source, target, layout, adjoint=True)
    )
    torch.testing.assert_close(
        circuit.append(FieldOperation(n, inverse_gates)).state().flatten(),
        state,
        rtol=0,
        atol=1e-12,
    )
