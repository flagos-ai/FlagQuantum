# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Assertions adapted from frozen Qualtran v0.7.0 multiplication assets.

"""Complete multiplication factories, unchanged assets and recursive source graphs."""

from __future__ import annotations

import ast
import csv
import hashlib
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest

from examples.algorithms.field_addition_catalog import field_signature_svg
from examples.algorithms.field_multiplication_catalog import (
    field_multiplication_example_graph,
    field_multiplication_examples,
)
from flagquantum.compiler.registers.finite_field.multiplication_model import (
    CX,
    TOFFOLI,
    polynomial_multiplication_graph,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env"


@pytest.fixture(scope="module", autouse=True)
def frozen_assets() -> None:
    from importlib.metadata import version

    import kickmix

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    with (REFERENCE.parent / "FlagQuantum-Qualtran-Kickmix-移植进度.csv").open(
        encoding="utf-8-sig"
    ) as handle:
        rows = [
            r
            for r in csv.DictReader(handle)
            if r["编号"] in {"Q-0518", "Q-0519", "Q-0520", "Q-0524"}
        ]
    assert len(rows) == 4
    for row in rows:
        data = (REFERENCE / row["源文件"]).read_bytes()
        assert (
            hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            == row["源对象"]
        )


def _source_key(node: Any) -> Any:
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.basic_gates import CNOT, Toffoli
    from qualtran.bloqs.bookkeeping import Join, Split
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
        BinaryPolynomialMultiplication,
        GF2MulK,
        GF2ShiftLeft,
        GF2ShiftRight,
        MultiplyPolyByOnePlusXk,
        SynthesizeLRCircuit,
    )

    if isinstance(node, CNOT):
        return CX
    if isinstance(node, Toffoli):
        return TOFFOLI
    if isinstance(node, BinaryPolynomialMultiplication):
        return ("polynomial", node.n)
    if isinstance(node, MultiplyPolyByOnePlusXk):
        return ("one_plus", node.n, node.k)
    if isinstance(node, SynthesizeLRCircuit):
        return ("linear", tuple(tuple(int(b) for b in row) for row in node.matrix))
    if isinstance(node, (GF2ShiftRight, GF2ShiftLeft)):
        q = node.qgf
        return (
            "shift",
            q.degree,
            None if q.is_symbolic() else int(q.gf_type.irreducible_poly),
            node.k,
            isinstance(node, GF2ShiftLeft),
        )
    if isinstance(node, Adjoint):
        child = _source_key(node.subbloq)
        assert child[0] == "scale"
        return (*child[:-1], True)
    if isinstance(node, GF2MulK):
        return ("scale", node.n, int(node.m_x), node.const, False)
    if isinstance(node, (Split, Join)):
        raise AssertionError("Split and join must be generalized before comparison")
    raise AssertionError(f"Unmapped source key {node}")


@pytest.mark.parametrize("index", range(9))
def test_all_original_factories_complete_signatures_and_graphs(index: int) -> None:
    from qualtran.bloqs.gf_arithmetic import gf2_multiplication as source
    from qualtran.resource_counting.generalizers import ignore_split_join

    factories = [
        "_gf16_multiplication",
        "_gf2_multiplication_symbolic",
        "_gf2_multiply_by_constant",
        "_gf2_poly_multiply_by_constant",
        "_multiplypolybyoneplusxk",
        "_binarypolynomialmultiplication",
        "_gf2shiftright",
        "_gf2mulviakaratsuba",
        "_gf2shiftleft",
    ]
    original = getattr(source, factories[index]).make()
    example = field_multiplication_examples()[index]
    name, signature, _ = example
    assert [(r.name, r.bitsize, r.shape, r.side.value) for r in signature] == [
        (r.name, r.bitsize, r.shape, r.side.value) for r in original.signature
    ]
    graph, leaves = original.call_graph(max_depth=1, generalizer=ignore_split_join)
    native = field_multiplication_example_graph(example)
    assert native.direct_calls == {
        (name, _source_key(child)): data["n"]
        for _, child, data in graph.edges(data=True)
    }
    assert native.leaf_counts == {
        _source_key(child): count for child, count in leaves.items()
    }
    ElementTree.fromstring(field_signature_svg([(name, signature, {})]))


@pytest.mark.parametrize("width", range(1, 10))
def test_entire_recursive_polynomial_graphs(width: int) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
        BinaryPolynomialMultiplication,
        MultiplyPolyByOnePlusXk,
    )
    from qualtran.resource_counting.generalizers import (
        ignore_alloc_free,
        ignore_split_join,
    )

    for shift in [None, *range(1, width + 2)]:
        original = (
            BinaryPolynomialMultiplication(width)
            if shift is None
            else MultiplyPolyByOnePlusXk(width, shift)
        )
        graph, leaves = original.call_graph(
            generalizer=(ignore_alloc_free, ignore_split_join)
        )
        native = polynomial_multiplication_graph(width, shift)
        assert native.direct_calls == {
            (_source_key(a), _source_key(b)): data["n"]
            for a, b, data in graph.edges(data=True)
        }
        assert native.leaf_counts == {_source_key(a): n for a, n in leaves.items()}


def test_complete_source_inventory() -> None:
    tree = ast.parse(
        (REFERENCE / "qualtran/bloqs/gf_arithmetic/gf2_multiplication.py").read_text()
    )
    assert {n.name for n in tree.body if isinstance(n, ast.ClassDef)} == {
        "SynthesizeLRCircuit",
        "GF2Multiplication",
        "Parity",
        "GF2MulMBUC",
        "GF2MulK",
        "MultiplyPolyByOnePlusXk",
        "BinaryPolynomialMultiplication",
        "GF2ShiftLeft",
        "GF2ShiftRight",
        "_GF2MulViaKaratsubaImpl",
        "GF2MulViaKaratsuba",
    }
    tests = ast.parse(
        (
            REFERENCE / "qualtran/bloqs/gf_arithmetic/gf2_multiplication_test.py"
        ).read_text()
    )
    assert len([n for n in tests.body if isinstance(n, ast.FunctionDef)]) == 42
