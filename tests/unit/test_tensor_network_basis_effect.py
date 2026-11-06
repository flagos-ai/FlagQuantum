"""Compare unnormalized tensor-network effects with pinned Qualtran bras."""

from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum.simulation.tensor_network.entrypoints import (
    build_tensor_network,
    tensor_network_amplitude,
    tensor_network_basis_effect,
    tensor_network_basis_measurement,
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


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("basis", ["x", "z"])
@pytest.mark.parametrize("wire", [0, 1, 2])
def test_basis_measurement_matches_source_classical_quantum_superoperator(
    basis: Literal["x", "z"], wire: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import (
        MeasureX,
        MeasureZ,
        MinusEffect,
        OneEffect,
        PlusEffect,
        ZeroEffect,
    )

    measure, effects = (
        (MeasureX, (PlusEffect, MinusEffect))
        if basis == "x"
        else (MeasureZ, (ZeroEffect, OneEffect))
    )
    values = torch.arange(1, 17, dtype=torch.float64).reshape(2, 8)
    state = torch.complex(values, values.remainder(7) - 3)
    state /= state.norm(dim=1, keepdim=True)
    plan = build_tensor_network(
        fq.Circuit(3, inputs=state, bsz=2, dtype=state.dtype), dtype=state.dtype
    )
    actual = tensor_network_basis_measurement(plan, wire, basis=basis)
    ordered = state.reshape((2,) * 4).movedim(wire + 1, 1).reshape(2, 2, 4)
    actual_density = torch.zeros((2, 2, 2, 4, 4), dtype=state.dtype)
    for bit, effect in enumerate(effects):
        bra = torch.as_tensor(effect().tensor_contract(), dtype=state.dtype).reshape(2)
        expected = torch.einsum("i,bis->bs", bra, ordered)
        torch.testing.assert_close(actual[bit], expected, atol=2e-12, rtol=0)
        actual_density[:, bit, bit] = torch.einsum(
            "bi,bj->bij", actual[bit], actual[bit].conj()
        )
    superoperator = torch.as_tensor(
        measure().tensor_contract(superoperator=True), dtype=state.dtype
    )
    expected_density = torch.einsum(
        "cdij,bis,bjt->bcdst", superoperator, ordered, ordered.conj()
    )
    torch.testing.assert_close(actual_density, expected_density, atol=2e-12, rtol=0)
    probability = sum(branch.abs().square().sum(dim=-1) for branch in actual)
    torch.testing.assert_close(probability, torch.ones_like(probability))


def test_basis_measurement_retains_impossible_outcome_and_removes_only_wire() -> None:
    plan = build_tensor_network(fq.Circuit(1, dtype=torch.complex128).x(0))
    zero, one = tensor_network_basis_measurement(plan, 0, basis="z")
    assert zero.shape == one.shape == (1, 1)
    assert zero.item() == 0 and one.item() == 1
