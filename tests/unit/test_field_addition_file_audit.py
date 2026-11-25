# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Qualtran v0.7.0 gf2_addition and gf2_add_k tests.

"""Whole field-addition files exercise native owners against unchanged source."""

from __future__ import annotations

import ast
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.field_addition_catalog import (
    field_addition_examples,
    field_signature_svg,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.resource_generalization import CliffordCall
from flagquantum.compiler.registers.finite_field.addition_model import (
    field_add_constant_calls,
    field_addition_calls,
    field_addition_resource_formula,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    gf2_add_constant_ir,
    gf2_add_ir,
)
from flagquantum.simulation.finite_field_addition import (
    binary_field_encoding,
    field_add_classical_values,
    field_add_constant_classical_value,
)
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from tests.unit.source_example_protocols import source_example_protocols

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env"


@pytest.fixture(scope="module", autouse=True)
def locked_source_assets() -> None:
    from importlib.metadata import version

    import kickmix

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    # These are the original seven ledger blobs, including notebooks and converter.
    blobs = {
        "gf2_add_k.ipynb": "6e8ab3396ad7b60f1680b6ee4f9e41756c9a6df8",
        "gf2_add_k.py": "41e380ea8bcaa7035837c0b3040072c6fe522cc1",
        "gf2_add_k_test.py": "0add69ae4e42c2b7ed8fa17a5618d5a2cbce3ab7",
        "gf2_addition.ipynb": "e5a299a88e983878bad3dc9bcc6a97551c66f776",
        "gf2_addition.py": "c63b50340bbe9a7653d8e23c497b4dfcf81f081d",
        "gf2_addition_test.py": "ed1ed34345933257551e206d1df2ca045883dc4c",
        "gf_utils.py": "3ecc74fb2657ff92c19ca0097ddf2c3d73720c5d",
    }
    for filename, blob in blobs.items():
        data = (REFERENCE / "qualtran/bloqs/gf_arithmetic" / filename).read_bytes()
        assert (
            hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            == blob
        )


def test_complete_original_test_function_inventory() -> None:
    directory = REFERENCE / "qualtran/bloqs/gf_arithmetic"
    expected = {
        "gf2_addition_test.py": {
            "test_gf16_addition",
            "test_gf2_addition_symbolic",
            "test_gf2_addition_classical_sim_quick",
            "test_gf2_addition_resource",
            "test_gf2_addition_classical_sim",
        },
        "gf2_add_k_test.py": {
            "test_gf16_add_k",
            "test_gf2_add_k_symbolic",
            "test_gf2_add_k_classical_sim_quick",
            "test_gf2_add_k_classical_sim",
        },
    }
    for path, names in expected.items():
        tree = ast.parse((directory / path).read_text())
        assert {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        } == names


@pytest.mark.parametrize("constant", (False, True))
@pytest.mark.parametrize("symbolic", (False, True))
def test_actual_factory_protocols_and_native_stored_graph_boundaries(
    constant: bool, symbolic: bool
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_add_k import (
        _gf2_add_k_symbolic,
        _gf16_add_k,
    )
    from qualtran.bloqs.gf_arithmetic.gf2_addition import (
        _gf2_addition_symbolic,
        _gf16_addition,
    )

    from examples.algorithms import field_ui_export as native
    from flagquantum.compiler.registers.finite_field.addition_component import (
        field_addition_program,
    )
    from flagquantum.core.ir.program import verify_program_record
    from flagquantum.core.ir.program.physical_codec import native_program_to_dict
    from flagquantum.errors import SerializationError

    example = (
        (_gf2_add_k_symbolic if symbolic else _gf16_add_k)
        if constant
        else (_gf2_addition_symbolic if symbolic else _gf16_addition)
    )
    original = example.make()
    signature = native.signature(example.name)
    assert [(r.name, r.bitsize, r.shape_symbolic, r.side.name) for r in signature] == [
        (r.name, r.bitsize, r.shape_symbolic, r.side.name) for r in original.signature
    ]
    expected = {
        "make": "PASS",
        "decompose": "NA" if symbolic else "PASS",
        "counts": "UNVERIFIED" if symbolic else "FAIL",
        "serialize": "FAIL",
        "qtyping": "NA" if symbolic else "PASS",
    }
    observed = source_example_protocols(example)
    assert {mode: status for mode, (status, _) in observed.items()} == expected
    assert "QGF" in observed["serialize"][1]
    if not symbolic:
        assert "Split" in observed["counts"][1] and "Join" in observed["counts"][1]

    record = native.decomposition(example.name)
    if symbolic:
        assert record is None
        assert dict(signature[0].dtype.parameters)["degree"] == original.bitsize
    else:
        assert record is not None and verify_program_record(record) == ()
        assert record == field_addition_program(
            signature[0].dtype, constant=1 if constant else None
        )
        assert dict(signature[0].dtype.parameters)["polynomial"] == int(
            original.qgf.irreducible_poly
        )
        # This codec admits physical-context Programs, not logical field graphs.
        # Bound CircuitIR JSON and execution are checked in test_typed_field_addition.
        with pytest.raises(SerializationError, match="requires a physical context"):
            native_program_to_dict(record)


@pytest.mark.parametrize("width", range(1, 9))
def test_register_addition_literal_source_order(width: int) -> None:
    import cirq
    from qualtran.bloqs.gf_arithmetic.gf2_addition import GF2Addition

    wires = np.asarray(cirq.LineQubit.range(2 * width))
    source, _ = (
        GF2Addition(width)
        .decompose_bloq()
        .to_cirq_circuit_and_quregs(x=wires[:width], y=wires[width:])
    )
    gates = gf2_add_ir(tuple(range(width)), tuple(range(width, 2 * width)))
    assert [(gate.name, gate.wires) for gate in gates] == [
        ("cx", tuple(q.x for q in operation.qubits))
        for operation in source.all_operations()
    ]
    assert field_addition_calls(width) == {CliffordCall("cx", 2): width}
    assert field_addition_resource_formula(width).clifford == len(gates) == width


@pytest.mark.parametrize("width", [2, 3, 4, 5])
def test_complete_original_classical_domains(width: int) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_add_k import GF2AddK
    from qualtran.bloqs.gf_arithmetic.gf2_addition import GF2Addition

    encoding = binary_field_encoding(width)
    field = encoding.gf_type
    source = GF2Addition(width)
    for x in field.elements:
        for y in field.elements:
            left, right = field_add_classical_values(encoding, x, y)
            expected = source.on_classical_vals(x=x, y=y)
            assert left is x
            assert (left, right) == (expected["x"], expected["y"])
            assert int(right) == int(x) ^ int(y)
    for k in range(1 << width):
        constant_source = GF2AddK(width, k)
        for x in field.elements:
            actual = field_add_constant_classical_value(encoding, x, k)
            assert actual == constant_source.on_classical_vals(x=x)["x"]
            assert int(actual) == int(x) ^ k


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 8])
def test_all_constant_gate_streams_and_models(width: int) -> None:
    import cirq
    from qualtran.bloqs.gf_arithmetic.gf2_add_k import GF2AddK
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    wires = tuple(range(width))
    qubits = np.asarray(cirq.LineQubit.range(width))
    for k in range(1 << width):
        original = GF2AddK(width, k)
        decomposed, _ = original.decompose_bloq().to_cirq_circuit_and_quregs(x=qubits)
        gates = gf2_add_constant_ir(wires, k)
        assert [(gate.name, gate.wires) for gate in gates] == [
            ("x", tuple(q.x for q in operation.qubits))
            for operation in decomposed.all_operations()
        ]
        declarations = field_add_constant_calls(width, k)
        assert declarations == {CliffordCall("x", 1): k.bit_count()}
        assert list(original.build_call_graph(None).values()) == list(
            declarations.values()
        )
        model = field_addition_resource_formula(width, constant=k)
        assert Counter(g.name for g in gates) == (
            Counter(x=k.bit_count()) if k else Counter()
        )
        source_cost = get_cost_value(original, QECGatesCost())
        assert model.clifford == source_cost.clifford == len(gates)
        assert source_cost.total_t_count() == model.t == model.measurement == 0
        assert model.logical_qubits == width


@pytest.mark.parametrize("width", [1, 2, 3, 5])
@pytest.mark.parametrize("constant", [None, 0, 1])
def test_append_full_complex_state_and_clean_spectators(
    width: int, constant: int | None
) -> None:
    # Two noncontiguous big-endian words, one complex spectator and one clean wire.
    source = tuple(range(1, 2 * width + 1, 2))
    target = tuple(range(2, 2 * width + 1, 2))
    n = 2 * width + 2
    generator = torch.Generator().manual_seed(width)
    state = torch.randn(1 << (n - 1), dtype=torch.complex128, generator=generator)
    state /= state.norm()
    state = torch.stack((state, torch.zeros_like(state)), dim=1).flatten()
    gates = (
        gf2_add_ir(source, target)
        if constant is None
        else gf2_add_constant_ir(target, constant)
    )
    expected = torch.empty_like(state)
    for index in range(len(state)):
        output = index
        for offset, wire in enumerate(target):
            bit = (
                ((index >> (n - 1 - source[offset])) & 1)
                if constant is None
                else ((constant >> (width - 1 - offset)) & 1)
            )
            output ^= bit << (n - 1 - wire)
        expected[output] = state[index]
    operation = FieldOperation(n, gates)
    circuit = Circuit(n, dtype=torch.complex128, inputs=state).append(operation)
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=1e-12)
    assert torch.count_nonzero(circuit.state().flatten()[1::2]) == 0
    torch.testing.assert_close(
        circuit.append(operation).state().flatten(), state, rtol=0, atol=1e-12
    )
    assert all(g.name in {"cx", "x"} for g in gates)


def test_symbolic_models_preserve_original_dispatch_and_zero_edges() -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_add_k import GF2AddK, _gf2_add_k_symbolic
    from qualtran.bloqs.gf_arithmetic.gf2_addition import _gf2_addition_symbolic
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    add = _gf2_addition_symbolic.make()
    assert field_addition_calls(add.bitsize) == {CliffordCall("cx", 2): add.bitsize}
    model = field_addition_resource_formula(add.bitsize)
    assert model.clifford == get_cost_value(add, QECGatesCost()).clifford
    assert model.logical_qubits == 2 * add.bitsize
    symbolic = _gf2_add_k_symbolic.make()
    for width, k in [
        (symbolic.bitsize, symbolic.k),
        (4, symbolic.k),
        (4, sympy.Integer(0)),
        (symbolic.bitsize, 0),
    ]:
        source = GF2AddK(width, k)
        declared = field_add_constant_calls(width, k)
        assert (
            list(declared.values())
            == list(source.build_call_graph(None).values())
            == [width]
        )
    assert field_add_constant_calls(4, 0) == {CliffordCall("x", 1): 0}
    for width, value in [(0, 0), (2, -1), (2, 4)]:
        with pytest.raises(ValueError):
            field_add_constant_calls(width, value)


def test_converter_identity_polynomial_and_original_scalar_failures() -> None:
    import galois
    from qualtran.bloqs.gf_arithmetic.gf_utils import qgf_converter

    polynomial = galois.Poly.Degrees([3, 1, 0])
    for value in [3, sympy.Symbol("m"), polynomial, [3, 1, 0]]:
        source = qgf_converter(value)
        actual = binary_field_encoding(value)
        assert (actual.characteristic, actual.degree, actual.irreducible_poly) == (
            source.characteristic,
            source.degree,
            source.irreducible_poly,
        )
    encoding = binary_field_encoding(polynomial)
    assert binary_field_encoding(encoding) is encoding
    field = encoding.gf_type
    other = FiniteFieldEncoding(2, 2).gf_type
    for left, right in [(1, field(1)), (field(1), 1), (field(1), other(1))]:
        with pytest.raises(AssertionError):
            field_add_classical_values(encoding, left, right)
    with pytest.raises(AssertionError):
        field_add_constant_classical_value(encoding, 1, 1)
    for k in [-1, 8]:
        with pytest.raises(ValueError):
            field_add_constant_classical_value(encoding, field(1), k)


@pytest.mark.parametrize("constant", [False, True])
def test_complete_notebook_examples(constant: bool) -> None:
    from xml.etree import ElementTree

    from qualtran.bloqs.gf_arithmetic.gf2_add_k import (
        _gf2_add_k_symbolic,
        _gf16_add_k,
    )
    from qualtran.bloqs.gf_arithmetic.gf2_addition import (
        _gf2_addition_symbolic,
        _gf16_addition,
    )

    factories = (
        (_gf16_add_k, _gf2_add_k_symbolic)
        if constant
        else (_gf16_addition, _gf2_addition_symbolic)
    )
    examples = field_addition_examples(constant=constant)
    for (_, signature, calls), factory in zip(examples, factories, strict=True):
        original = factory.make()
        assert [(r.name, r.bitsize) for r in signature] == [
            (r.name, r.bitsize) for r in original.signature
        ]
        assert list(calls.values()) == list(original.build_call_graph(None).values())
        assert all(r.dtype.encoding == "finite_field" for r in signature)
        field = binary_field_encoding(original.bitsize)
        assert all(r.dtype == field.register_type() for r in signature)
    svg = field_signature_svg(examples)
    ElementTree.fromstring(svg)


@pytest.mark.parametrize("value", [0, -1, True, None, 1.5, [], [3, 3]])
def test_converter_boundary_dispatch(value: object) -> None:
    from qualtran.bloqs.gf_arithmetic.gf_utils import qgf_converter

    try:
        original = qgf_converter(value)
    except (ValueError, TypeError, IndexError) as error:
        with pytest.raises(type(error)):
            binary_field_encoding(value)
    else:
        actual = binary_field_encoding(value)
        assert (actual.degree, actual.irreducible_poly) == (
            original.degree,
            original.irreducible_poly,
        )
        assert qgf_converter(original) is original


def test_constant_generic_adjoint_uses_existing_circuit_inverse() -> None:
    import cirq
    from qualtran.bloqs.gf_arithmetic.gf2_add_k import GF2AddK

    gates = gf2_add_constant_ir((0, 1, 2, 3), 13)
    circuit = Circuit(4).append(FieldOperation(4, gates))
    inverse = circuit.adjoint().to_ir().instructions
    source, _ = (
        GF2AddK(4, 13)
        .adjoint()
        .decompose_bloq()
        .to_cirq_circuit_and_quregs(x=np.asarray(cirq.LineQubit.range(4)))
    )
    assert [(g.name, g.wires) for g in inverse] == [
        ("x", tuple(q.x for q in operation.qubits))
        for operation in source.all_operations()
    ]
