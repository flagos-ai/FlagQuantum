# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Assertions adapted from frozen Qualtran v0.7.0 GF2Inverse assets.

"""Complete inverse declarations, resource accounts and raw coherent outputs."""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import pytest
import torch
from sympy import Symbol, simplify

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.field_inverse_catalog import (
    field_inverse_example,
    field_inverse_example_gate_model,
    field_inverse_example_graph,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.finite_field.field_inverse import (
    gf2_inverse_decomposition,
    gf2_inverse_workspace_size,
)
from flagquantum.compiler.registers.finite_field.inverse_model import (
    inverse_gate_model,
    inverse_junk_register_count,
)
from flagquantum.compiler.registers.finite_field.multiplication_model import CX, TOFFOLI
from flagquantum.simulation.finite_field_addition import binary_field_encoding
from flagquantum.simulation.finite_field_inverse import field_inverse_classical_values
from tests.unit.source_example_protocols import source_example_protocols

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env"


@pytest.fixture(scope="module", autouse=True)
def frozen_assets() -> None:
    from importlib.metadata import version

    import kickmix

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    blobs = {
        "gf2_inverse.ipynb": "30174d21584b13dfc2da26c03b43de6958a58f06",
        "gf2_inverse.py": "0cabf787ad5716c6099b613ac2f038253402396b",
        "gf2_inverse_test.py": "b0fb83bcb11ea08a0854f16c17e8a082202396ce",
        "gf2_multiplication.py": "1338bb4b9f09a0a4b813a1124481e26ee3446cd8",
        "gf2_square.py": "5dae828b6d880ee45354d0c1173eb7a275f0fdee",
    }
    for filename, blob in blobs.items():
        data = (REFERENCE / "qualtran/bloqs/gf_arithmetic" / filename).read_bytes()
        assert (
            hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            == blob
        )


def _source_key(node: Any) -> Hashable:
    from qualtran.bloqs.gf_arithmetic.gf2_addition import GF2Addition
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
        GF2MulViaKaratsuba,
        SynthesizeLRCircuit,
    )
    from qualtran.bloqs.gf_arithmetic.gf2_square import GF2Square

    if isinstance(node, GF2Addition):
        return ("addition", node.qgf.degree)
    if isinstance(node, GF2MulViaKaratsuba):
        return ("product", node.dtype.degree)
    if isinstance(node, GF2Square):
        return ("square", node.qgf.degree, node.k, node.uncompute)
    if isinstance(node, SynthesizeLRCircuit):
        return ("linear_bound", node.matrix.shape[0])
    raise AssertionError(f"Unmapped source child {node}")


@pytest.mark.parametrize("degree", range(1, 17))
def test_native_counts_match_both_actual_source_decomposition_and_annotation(
    degree: int,
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_inverse import GF2Inverse
    from qualtran.resource_counting.generalizers import (
        ignore_alloc_free,
        ignore_split_join,
    )

    source = GF2Inverse(degree)
    generalizers = [ignore_split_join, ignore_alloc_free]
    decomposed = source.decompose_bloq().bloq_counts(generalizer=generalizers)
    annotated = source.bloq_counts(generalizer=generalizers)
    assert decomposed == annotated
    native = field_inverse_example_graph(field_inverse_example(degree))
    assert native.leaf_counts == {
        _source_key(child): count for child, count in decomposed.items()
    }


@pytest.mark.parametrize("symbolic", (False, True))
def test_actual_inverse_factory_protocols_and_native_root(symbolic: bool) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_inverse import (
        _gf2_inverse_symbolic,
        _gf16_inverse,
    )

    from examples.algorithms import field_ui_export as native
    from examples.algorithms.field_power_decomposition import field_power_program
    from flagquantum.compiler.registers.finite_field.inverse_component import (
        field_inverse_program,
    )
    from flagquantum.core.ir.program import verify_program_record
    from flagquantum.core.ir.program.physical_codec import native_program_to_dict
    from flagquantum.errors import SerializationError

    factory = _gf2_inverse_symbolic if symbolic else _gf16_inverse
    original = factory.make()
    observed = source_example_protocols(factory)
    assert {mode: status for mode, (status, _) in observed.items()} == {
        "make": "PASS",
        "decompose": "NA" if symbolic else "PASS",
        "counts": "UNVERIFIED" if symbolic else "PASS",
        "serialize": "FAIL",
        "qtyping": "NA" if symbolic else "PASS",
    }
    assert "QGF" in observed["serialize"][1]
    if symbolic:
        # The original DocSpec exports only its concrete example. Keep the
        # symbolic class factory on the domain catalog path, not the UI index.
        assert factory.name not in native.INSTANCES
        _, signature, _ = field_inverse_example(original.bitsize)
        record = field_power_program("GF2Inverse", signature, {})
    else:
        signature = native.signature(factory.name)
        record = native.decomposition(factory.name)
    assert [(r.name, r.bitsize, r.shape_symbolic, r.side.name) for r in signature] == [
        (r.name, r.bitsize, r.shape_symbolic, r.side.name) for r in original.signature
    ]
    if symbolic:
        assert record is None
    else:
        assert record == field_inverse_program(
            signature[0].dtype, modulus=int(original.qgf.irreducible_poly)
        )
        assert record is not None and verify_program_record(record) == ()
        with pytest.raises(SerializationError, match="requires a physical context"):
            native_program_to_dict(record)


@pytest.mark.parametrize(
    "degree", [*range(1, 17), Symbol("m", positive=True, integer=True), Symbol("m")]
)
def test_complete_signature_graph_and_static_width(degree: Any) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_inverse import GF2Inverse
    from qualtran.resource_counting import QubitCount, get_cost_value
    from qualtran.resource_counting.generalizers import ignore_split_join

    source = GF2Inverse(degree)
    example = field_inverse_example(degree)
    name, signature, _ = example
    assert [(r.name, r.bitsize, r.shape_symbolic, r.side.value) for r in signature] == [
        (r.name, r.bitsize, r.shape_symbolic, r.side.value) for r in source.signature
    ]
    assert inverse_junk_register_count(degree) == source.n_junk_regs
    assert (
        simplify(
            sum(r.total_bits() for r in signature.rights())
            - get_cost_value(source, QubitCount())
        )
        == 0
    )
    graph, leaves = source.call_graph(max_depth=1, generalizer=ignore_split_join)
    native = field_inverse_example_graph(example)
    assert native.direct_calls == {
        (name, _source_key(child)): data["n"]
        for _, child, data in graph.edges(data=True)
    }
    assert native.leaf_counts == {_source_key(child): n for child, n in leaves.items()}


@pytest.mark.parametrize(
    "degree", [*range(1, 17), Symbol("m", positive=True, integer=True), Symbol("m")]
)
def test_independent_source_gate_model(degree: Any) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_inverse import GF2Inverse
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    model = (
        field_inverse_example_gate_model(degree)
        if isinstance(degree, int)
        else inverse_gate_model(degree)
    )
    original = get_cost_value(GF2Inverse(degree), QECGatesCost())
    assert simplify(model[CX] - original.clifford) == 0
    assert simplify(model[TOFFOLI] - original.toffoli) == 0


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_raw_inverse_append_complex_junk_scratch_and_exact_restore(width: int) -> None:
    encoding = binary_field_encoding(width)
    modulus = int(encoding.gf_type.irreducible_poly)
    needed = gf2_inverse_workspace_size(width, modulus=modulus)
    count = width + needed + 1
    plan = gf2_inverse_decomposition(
        tuple(range(width)), tuple(range(width, count - 1)), modulus=modulus
    )
    state = torch.zeros(1 << count, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for value in range(1 << width):
        source, inverse, junk = field_inverse_classical_values(
            encoding, encoding.gf_type(value)
        )
        outputs = [int(source), int(inverse), *(int(v) for v in junk)]
        for spectator in [0, 1]:
            initial = (value << (needed + 1)) | spectator
            bits = [0] * count
            bits[-1] = spectator
            for word, wires in zip(outputs, plan.outputs, strict=True):
                for bit, wire in enumerate(reversed(wires)):
                    bits[wire] = (word >> bit) & 1
            final = int("".join(map(str, bits)), 2)
            state[initial] = torch.tensor(
                complex(value + 1, 2 * spectator - 1), dtype=torch.complex128
            )
            expected[final] = state[initial]
    expected /= state.norm()
    state /= state.norm()
    circuit = Circuit(count, inputs=state, dtype=torch.complex128).append(
        FieldOperation(count, plan.instructions)
    )
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=1e-12)
    restored = circuit.append(FieldOperation(count, tuple(reversed(plan.instructions))))
    torch.testing.assert_close(restored.state().flatten(), state, rtol=0, atol=1e-12)


def test_whole_source_inventory() -> None:
    source = ast.parse(
        (REFERENCE / "qualtran/bloqs/gf_arithmetic/gf2_inverse.py").read_text()
    )
    classes = [n for n in source.body if isinstance(n, ast.ClassDef)]
    assert [n.name for n in classes] == ["GF2Inverse"]
    assert {n.name for n in classes[0].body if isinstance(n, ast.FunctionDef)} == {
        "__init__",
        "signature",
        "bitsize",
        "n_junk_regs",
        "bitsize_hamming_weight",
        "my_static_costs",
        "_bits",
        "gf2_multiplier",
        "build_composite_bloq",
        "build_call_graph",
        "on_classical_vals",
    }
    assert {n.name for n in source.body if isinstance(n, ast.FunctionDef)} == {
        "_gf16_inverse",
        "_gf2_inverse_symbolic",
    }
    tests = ast.parse(
        (REFERENCE / "qualtran/bloqs/gf_arithmetic/gf2_inverse_test.py").read_text()
    )
    assert {n.name for n in tests.body if isinstance(n, ast.FunctionDef)} == {
        "test_gf16_inverse",
        "test_gf2_inverse_symbolic",
        "test_gf2_inverse_symbolic_toffoli_complexity",
        "test_gf2_inverse_classical_sim_quick",
        "test_gf2_inverse_classical_sim",
        "test_gf2_equivalent_bloq_counts",
    }


def test_concrete_model_requires_independent_square_costs() -> None:
    with pytest.raises(ValueError, match="square costs"):
        inverse_gate_model(2, modulus=7)
