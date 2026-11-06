"""Verify the paper plaquette using independent Jordan–Wigner ladder matrices."""

import pytest
import torch

from flagquantum.algorithms.primitives.hubbard_plaquette import HubbardPlaquette
from flagquantum.circuit import Circuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from flagquantum.simulation.unitary import get_unitary

pytestmark = pytest.mark.unit


def _hopping_hamiltonian():
    annihilate = []
    for mode in range(4):
        operator = torch.zeros((16, 16), dtype=torch.complex128)
        mask = 1 << (3 - mode)
        for column in range(16):
            if column & mask:
                operator[column ^ mask, column] = (-1) ** (
                    (column >> (4 - mode)).bit_count()
                )
        annihilate.append(operator)
    result = torch.zeros((16, 16), dtype=torch.complex128)
    for mode in range(4):
        hopping = annihilate[mode].mH @ annihilate[(mode + 1) % 4]
        result += hopping + hopping.mH
    return result


@pytest.mark.parametrize("time", [0, 0.123, -0.39])
@pytest.mark.parametrize("adjoint", [False, True])
def test_paper_plaquette_matches_complete_fermion_unitary(time, adjoint):
    op = HubbardPlaquette.make_on(modes=(0, 1, 2, 3), timestep=time, adjoint=adjoint)
    circuit = Circuit(4).append(op)
    expected = torch.linalg.matrix_exp(
        ((-1 if adjoint else 1) * 1j * time) * _hopping_hamiltonian()
    )
    torch.testing.assert_close(
        get_unitary(circuit, dtype=torch.complex128), expected, atol=2e-12, rtol=0
    )
    report = estimate_ftqc_resources(op, emitted_ir=circuit.to_ir())
    assert report.emitted is not None
    assert report.emitted.t_count == 8
    assert report.emitted.unconditional["rz"] == 6
    assert "measure" not in report.emitted.unconditional
    assert report.logical.gates.rotation == (2 if time else 0)
    assert report.logical.peak_logical_qubits == 4
