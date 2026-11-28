"""Verify the Figure 7 SELECT action and report Qualtran's resource estimate."""

from collections import Counter
from importlib.metadata import version
from itertools import product

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.primitives.multiplexers import SelectTHC


def test_select_thc_complex_amplitudes_and_workspace() -> None:
    np = pytest.importorskip("numpy")
    cirq = pytest.importorskip("cirq")
    assert version("qualtran") == "0.7.0"
    from qualtran.bloqs.chemistry.thc.select_bloq import SelectTHC as CostModel
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    rng = np.random.default_rng(23)
    state = rng.normal(size=8192) + 1j * rng.normal(size=8192)
    state /= np.linalg.norm(state)
    circuit = fq.Circuit(13, dtype=torch.complex128, inputs=torch.from_numpy(state))
    op = SelectTHC.make_on(
        mu=(0, 1),
        nu=(2, 3),
        succ=4,
        nu_eq_mp1=5,
        plus_mn=6,
        plus_a=7,
        plus_b=8,
        sys_a=(9, 10),
        sys_b=(11, 12),
        networks=(((0, 1, 0.2),), ((0, 1, 0.5),)),
        one_body_networks=((), ((0, 1, -0.4),)),
    )
    circuit.append(op)

    identity = np.eye(4)
    z = np.diag([1, 1, -1, -1])

    def reflection(angle: float):
        rotation = cirq.unitary(cirq.givens(angle))
        return rotation.conj().T @ z @ rotation

    source = state.reshape(4, 4, 2, 2, 2, 2, 2, 16)
    expected = np.empty_like(source)
    for address in product(range(4), range(4), *([range(2)] * 5)):
        mu, nu, success, one_body, order, spin_mu, spin_nu = address
        branch = source[address]
        if success:
            mu_angle = (
                (0.0, -0.4, 0.0, 0.0)[mu] if one_body else (0.2, 0.5, 0.0, 0.0)[mu]
            )
            first = reflection(mu_angle)
            first = (
                np.kron(first, identity) if spin_mu == 0 else np.kron(identity, first)
            )
            branch = first @ branch
            if not one_body:
                second = reflection((0.2, 0.5, 0.0, 0.0)[nu])
                second = (
                    np.kron(second, identity)
                    if spin_nu == 0
                    else np.kron(identity, second)
                )
                branch = second @ branch
        output_address = (
            address
            if one_body
            else (nu, mu, success, one_body, 1 - order, spin_nu, spin_mu)
        )
        expected[output_address] = branch

    actual = circuit.state().detach().numpy().reshape(-1)
    np.testing.assert_allclose(actual, expected.ravel(), atol=1e-12, rtol=0)
    ir = circuit.to_ir()
    assert ir.n_wires == 13  # No compiler workspace or angle-data register.
    assert all(item.matrix is None for item in ir.instructions)
    print(
        "FlagQuantum emitted gates:",
        dict(Counter(item.name for item in ir.instructions)),
    )
    model = CostModel(num_mu=2, num_spin_orb=4, num_bits_theta=8, keep_bitsize=1)
    print("Qualtran modeled resources:", get_cost_value(model, QECGatesCost()))

    circuit.append(op)
    torch.testing.assert_close(
        circuit.state().flatten(), torch.from_numpy(state), rtol=0, atol=1e-12
    )
