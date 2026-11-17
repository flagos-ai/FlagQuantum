"""Verify source tensor order, declared calls, and the native inverse."""

from cmath import exp
from itertools import product
from math import pi

import numpy as np
import pytest
import torch
from sympy import Symbol

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.two_bit_fermionic_fourier_catalog import (
    TWO_BIT_FOURIER_EXAMPLE,
    two_bit_fermionic_fourier_graph,
)
from flagquantum.algorithms.primitives.fermionic_fourier import (
    source_two_bit_fermionic_fourier_calls,
    source_two_bit_fermionic_fourier_signature,
    two_bit_fermionic_fourier_ir,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    PhaseCall,
    RotationCall,
)
from flagquantum.core.ir.program.registers import RegisterSide
from flagquantum.simulation.fermionic_fourier import two_bit_fermionic_fourier_matrix

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(("k", "n"), tuple(product(range(4), range(1, 4))))
def test_original_matrix_grid_with_exact_circuit_phase(k: int, n: int) -> None:
    source = pytest.importorskip("qualtran.bloqs.qft.two_bit_ffft")
    matrix = two_bit_fermionic_fourier_matrix(k, n)
    np.testing.assert_array_equal(matrix, source._fkn_matrix(k, n))
    phase = exp(-1j * pi * k / n)
    decomposition = source.TwoBitFFFT(k, n).decompose_bloq().tensor_contract()
    np.testing.assert_allclose(decomposition, phase * matrix, atol=1e-12, rtol=0)

    for inverse in (False, True):
        analytical = two_bit_fermionic_fourier_matrix(k, n, adjoint=inverse)
        tagged = source.TwoBitFFFT(k, n, is_adjoint=inverse)
        incoming = {"x": "in_x", "y": "in_y"}
        outgoing = {"x": "out_x", "y": "out_y"}
        (tensor,) = tagged.my_tensors(incoming, outgoing)
        assert tensor.inds == (("out_x", 0), ("out_y", 0), ("in_x", 0), ("in_y", 0))
        np.testing.assert_array_equal(tensor.data.reshape(4, 4), analytical)
        circuit = Circuit(
            2, inputs=torch.eye(4, dtype=torch.complex128), dtype=torch.complex128
        )
        circuit.append(
            FieldOperation(2, two_bit_fermionic_fourier_ir(0, 1, k, n, adjoint=inverse))
        )
        expected_phase = phase.conjugate() if inverse else phase
        torch.testing.assert_close(
            circuit.state().reshape(4, 4).T,
            torch.tensor(expected_phase * analytical),
            atol=3e-12,
            rtol=0,
        )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("precision", [1e-10, Symbol("eps")])
def test_source_signature_and_precise_direct_calls(precision: float | Symbol) -> None:
    source = pytest.importorskip("qualtran.bloqs.qft.two_bit_ffft")
    resource = pytest.importorskip("qualtran.resource_counting")
    k, n = TWO_BIT_FOURIER_EXAMPLE
    op = source.TwoBitFFFT(k, n, eps=precision)
    assert [(reg.name, reg.bitsize) for reg in op.signature] == [("x", 1), ("y", 1)]
    assert [
        (reg.name, reg.bitsize, reg.side)
        for reg in source_two_bit_fermionic_fourier_signature()
    ] == [
        ("x", 1, RegisterSide.THRU),
        ("y", 1, RegisterSide.THRU),
    ]
    assert op.wire_symbol(None).text == "F(k, n)"
    lookup: dict[str, PhaseCall | CliffordCall] = {
        "SGate": PhaseCall("s"),
        "TGate": PhaseCall("t"),
        "Hadamard": CliffordCall("h", 1),
        "CNOT": CliffordCall("cx", 2),
    }
    normalized: dict[RotationCall | PhaseCall | CliffordCall, int] = {}
    for node, count in op.build_call_graph(resource.SympySymbolAllocator()).items():
        name = type(node).__name__
        call = (
            RotationCall("rz", node.angle, precision=node.eps)
            if name == "Rz"
            else lookup[name]
        )
        normalized[call] = count
    assert source_two_bit_fermionic_fourier_calls(k, n, epsilon=precision) == normalized


def test_notebook_declared_graph_uses_combined_t_count() -> None:
    graph = two_bit_fermionic_fourier_graph(*TWO_BIT_FOURIER_EXAMPLE)
    assert graph.root == "F(2, 3)"
    assert graph.leaf_counts["T"] == 2
    assert graph.leaf_counts["S"] == 3
    assert graph.leaf_counts["H"] == 6
    assert graph.leaf_counts["CX"] == 3
    assert len(graph.leaf_counts) == 5
