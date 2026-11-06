"""Compare unnormalized tensor-network effects with pinned Qualtran bras."""

from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum.simulation.tensor_network.entrypoints import (
    build_tensor_network,
    tensor_network_amplitude,
    tensor_network_basis_effect,
)

pytestmark = pytest.mark.unit
Selection = dict[int, tuple[Literal["x", "z"], int]]


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "effects",
    [
        {1: ("z", 0)},
        {0: ("z", 1)},
        {2: ("x", 0)},
        {3: ("x", 1)},
        {3: ("x", 1), 0: ("z", 1)},
        {1: ("x", 1), 3: ("z", 0), 0: ("x", 0), 2: ("z", 1)},
    ],
)
def test_basis_effect_preserves_complex_amplitudes_and_survivor_order(
    effects: Selection,
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import (
        MinusEffect,
        OneEffect,
        PlusEffect,
        ZeroEffect,
    )

    rows = torch.arange(1, 33, dtype=torch.float64).reshape(2, 16)
    initial = torch.complex(rows, (rows % 7 - 3).flip(1))
    initial /= initial.norm(dim=1, keepdim=True)
    circuit = fq.Circuit(4, bsz=2, inputs=initial, dtype=torch.complex128)
    circuit.h(2).cx(2, 0).rz(3, theta=-0.32)
    plan = build_tensor_network(circuit, dtype=torch.complex128)

    expected = circuit.state().reshape(2, 2, 2, 2, 2)
    for wire in sorted(effects, reverse=True):
        basis, bit = effects[wire]
        source = (
            (PlusEffect, MinusEffect) if basis == "x" else (ZeroEffect, OneEffect)
        )[bit]()
        bra = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
        expected = torch.tensordot(expected, bra.reshape(2), dims=([wire + 1], [0]))
    expected = expected.reshape(2, -1)

    actual = tensor_network_basis_effect(plan, effects)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-12)
    assert torch.all(actual.norm(dim=1) < 1)


def test_basis_effect_preserves_zero_branches_and_existing_scalar_projection() -> None:
    circuit = fq.Circuit(2, dtype=torch.complex128).h(0).s(0)
    plan = build_tensor_network(circuit, dtype=torch.complex128)
    torch.testing.assert_close(tensor_network_basis_effect(plan, {}), plan.contract())
    impossible = tensor_network_basis_effect(plan, {1: ("z", 1)})
    assert torch.count_nonzero(impossible) == 0
    complete = tensor_network_basis_effect(plan, {0: ("z", 1), 1: ("z", 0)})
    torch.testing.assert_close(complete[:, 0], tensor_network_amplitude(plan, "10"))
    with pytest.raises(ValueError, match="outside"):
        tensor_network_basis_effect(plan, {2: ("x", 0)})
    with pytest.raises(ValueError, match="binary bit"):
        tensor_network_basis_effect(plan, {1: ("x", 2)})
