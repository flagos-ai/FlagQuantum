# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Assertions adapted from locked Qualtran v0.7.0 gf2_square tests and notebook.

"""Whole-file square audit reuses accepted emitters and isolates source defects."""

from __future__ import annotations

import ast
import csv
import hashlib
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.field_addition_catalog import field_signature_svg
from examples.algorithms.field_square_catalog import (
    field_square_example_graph,
    field_square_examples,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.finite_field.field_linear import gf2_frobenius
from flagquantum.compiler.registers.finite_field.field_multiplication import (
    bind_field_outputs,
)
from flagquantum.compiler.registers.finite_field.linear_resource_model import (
    source_binary_linear_resource_formula,
)
from flagquantum.simulation.finite_field_addition import binary_field_encoding
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from flagquantum.simulation.finite_field_frobenius import (
    field_frobenius_classical_value,
    field_frobenius_matrix,
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
            if row["编号"] in {"Q-0519", "Q-0521", "Q-0522", "Q-0523", "Q-0524"}
        ]
    assert len(rows) == 5
    for row in rows:
        data = (REFERENCE / row["源文件"]).read_bytes()
        assert (
            hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            == row["源对象"]
        )


def test_whole_original_function_inventory() -> None:
    path = REFERENCE / "qualtran/bloqs/gf_arithmetic/gf2_square_test.py"
    tree = ast.parse(path.read_text())
    assert {node.name for node in tree.body if isinstance(node, ast.FunctionDef)} == {
        "test_gf16_square",
        "test_gf2_square_symbolic",
        "test_gf2_square_classical_sim_quick",
        "test_gf2_square_resource",
        "test_gf2_square_classical_sim",
    }


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 7])
@pytest.mark.parametrize("adjoint", [False, True])
def test_source_scalar_values_and_native_mathematical_matrix(
    width: int, count: int, adjoint: bool
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_square import GF2Square

    encoding = binary_field_encoding(width)
    field = encoding.gf_type
    source = GF2Square(width, k=count, uncompute=adjoint)
    matrix = field_frobenius_matrix(encoding, count)
    for value in field.elements:
        actual = field_frobenius_classical_value(
            encoding, value, count, adjoint=adjoint
        )
        assert actual == source.on_classical_vals(x=value)["x"]
        bits = np.array([(int(value) >> i) & 1 for i in range(width)])
        transformed = (matrix @ bits) % 2
        expected = sum(int(bit) << i for i, bit in enumerate(transformed))
        forward = field_frobenius_classical_value(encoding, value, count)
        assert expected == int(forward)
        restored = field_frobenius_classical_value(
            encoding, forward, count, adjoint=True
        )
        assert restored == value


@pytest.mark.parametrize(
    "width,count", [(1, 1), (2, 1), (3, 2), (4, 4), (5, 1), (8, 2)]
)
def test_linear_dependency_full_model_matches_source(width: int, count: int) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import SynthesizeLRCircuit
    from qualtran.bloqs.gf_arithmetic.gf2_square import GF2Square
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    matrix = field_frobenius_matrix(binary_field_encoding(width), count)
    source = GF2Square(width, k=count)
    np.testing.assert_array_equal(matrix, source.squaring_matrix)
    model = source_binary_linear_resource_formula(
        tuple(tuple(int(v) for v in row) for row in matrix)
    )
    linear = SynthesizeLRCircuit(matrix)
    assert model.clifford == get_cost_value(linear, QECGatesCost()).clifford
    assert model.clifford == get_cost_value(source, QECGatesCost()).clifford
    assert model.logical_qubits == width
    assert model.t == model.measurement == 0
    encoding = binary_field_encoding(width)
    plan = gf2_frobenius(
        tuple(range(width)), count, modulus=int(encoding.gf_type.irreducible_poly)
    )
    assert len(plan.instructions) == model.clifford
    assert plan.scratch == ()


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 7])
def test_append_noncontiguous_complex_state_and_exact_restore(count: int) -> None:
    encoding = binary_field_encoding(3)
    wires = (1, 3, 4)
    n = 6
    state = torch.zeros(1 << n, dtype=torch.complex128)
    generator = torch.Generator().manual_seed(count)
    amplitudes = torch.randn(32, dtype=torch.complex128, generator=generator)
    amplitudes /= amplitudes.norm()
    state[::2] = amplitudes
    expected = torch.empty_like(state)
    for index in range(len(state)):
        value = sum(
            ((index >> (n - 1 - wire)) & 1) << (2 - i) for i, wire in enumerate(wires)
        )
        result = int(encoding.gf_type(value) ** (2 ** (count % 3)))
        output = index
        for i, wire in enumerate(wires):
            output = (output & ~(1 << (n - 1 - wire))) | (
                ((result >> (2 - i)) & 1) << (n - 1 - wire)
            )
        expected[output] = state[index]
    modulus = int(encoding.gf_type.irreducible_poly)
    forward = gf2_frobenius(wires, count, modulus=modulus)
    reverse = gf2_frobenius(wires, count, modulus=modulus, adjoint=True)
    circuit = Circuit(n, dtype=torch.complex128, inputs=state).append(
        FieldOperation(n, bind_field_outputs(forward, (wires,)))
    )
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=1e-12)
    assert torch.count_nonzero(circuit.state().flatten()[1::2]) == 0
    restored = circuit.append(FieldOperation(n, bind_field_outputs(reverse, (wires,))))
    torch.testing.assert_close(restored.state().flatten(), state, rtol=0, atol=1e-12)
    assert forward.scratch == reverse.scratch == ()


def test_complete_factory_metadata_and_two_level_graphs() -> None:
    from xml.etree import ElementTree

    from qualtran.bloqs.gf_arithmetic.gf2_square import (
        _gf2_square_symbolic,
        _gf16_square,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    examples = field_square_examples()
    for example, factory in zip(
        examples, (_gf16_square, _gf2_square_symbolic), strict=True
    ):
        name, signature, model = example
        source = factory.make()
        assert [(reg.name, reg.bitsize) for reg in signature] == [
            (reg.name, reg.bitsize) for reg in source.signature
        ]
        assert (
            signature[0].dtype == binary_field_encoding(source.bitsize).register_type()
        )
        assert (
            sympy.simplify(
                model.clifford - get_cost_value(source, QECGatesCost()).clifford
            )
            == 0
        )
        graph = field_square_example_graph(example)
        assert (
            list(graph.direct_calls.values())
            == list(source.build_call_graph(None).values())
            == [1]
        )
        assert graph.leaf_counts == {"Binary linear map": 1}
        expanded = field_square_example_graph(example, max_depth=None)
        assert len(expanded.direct_calls) == 2
        assert list(expanded.leaf_counts.values()) == [model.clifford]
        assert expanded.root == name
    ElementTree.fromstring(
        field_signature_svg(
            tuple((name, signature, {}) for name, signature, _ in examples)
        )
    )


@pytest.mark.parametrize("matrix", [(), ((1, 0),), ((1, 2), (0, 1)), ((1, 0), (0, 0))])
def test_resource_admission_errors(matrix: tuple[tuple[int, ...], ...]) -> None:
    with pytest.raises(ValueError):
        source_binary_linear_resource_formula(matrix)


@pytest.mark.parametrize("count", [-1, True])
def test_matrix_count_admission(count: int) -> None:
    with pytest.raises(ValueError):
        field_frobenius_matrix(binary_field_encoding(3), count)


def test_matrix_symbolic_and_odd_field_admission() -> None:
    for encoding in [
        FiniteFieldEncoding(3, 2),
        binary_field_encoding(sympy.Symbol("m")),
    ]:
        with pytest.raises(ValueError):
            field_frobenius_matrix(encoding)


def test_source_value_array_and_scalar_failure_policy() -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_square import GF2Square

    encoding = binary_field_encoding(3)
    values = encoding.gf_type.elements
    for adjoint in [False, True]:
        np.testing.assert_array_equal(
            field_frobenius_classical_value(encoding, values, 2, adjoint=adjoint),
            GF2Square(3, k=2, uncompute=adjoint).on_classical_vals(x=values)["x"],
        )
    with pytest.raises(AssertionError):
        field_frobenius_classical_value(encoding, 1)
    with pytest.raises(AssertionError):
        field_frobenius_classical_value(encoding, binary_field_encoding(2).gf_type(1))
