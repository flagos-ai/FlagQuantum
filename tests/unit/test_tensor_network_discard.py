"""Compare the pinned discard compositions to native partial traces."""

from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum.simulation.tensor_network.entrypoints import (
    build_tensor_network,
    discard_basis_measurement_result,
    tensor_network_basis_measurement,
    tensor_network_discard_qubit,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("basis", ["x", "z"])
def test_measurement_then_classical_discard_matches_source_trace(
    basis: Literal["x", "z"],
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import BloqBuilder, QBit, Register, Side, Signature
    from qualtran.bloqs.basic_gates import Discard, MeasureX, MeasureZ

    builder, inputs = BloqBuilder.from_signature(
        Signature([Register("q", QBit(), side=Side.LEFT)])
    )
    measurement = MeasureX() if basis == "x" else MeasureZ()
    builder.add(Discard(), c=builder.add(measurement, q=inputs["q"]))
    reference = torch.as_tensor(
        builder.finalize().tensor_contract(superoperator=True), dtype=torch.complex128
    ).reshape(2, 2)

    values = torch.arange(1, 17, dtype=torch.float64).reshape(2, 8)
    state = torch.complex(values, values.remainder(7) - 3)
    state /= state.norm(dim=1, keepdim=True)
    plan = build_tensor_network(
        fq.Circuit(3, inputs=state, bsz=2, dtype=state.dtype), dtype=state.dtype
    )
    outcomes = tensor_network_basis_measurement(plan, 1, basis=basis)
    actual = discard_basis_measurement_result(outcomes)
    by_wire = state.reshape(2, 2, 2, 2).movedim(2, 1).reshape(2, 2, 4)
    expected = torch.einsum("ij,bis,bjt->bst", reference, by_wire, by_wire.conj())
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-12)
    torch.testing.assert_close(
        actual.diagonal(dim1=-2, dim2=-1).sum(-1),
        torch.ones(2, dtype=actual.dtype),
    )
    torch.testing.assert_close(
        tensor_network_discard_qubit(plan, 1), expected, rtol=0, atol=2e-12
    )


@pytest.mark.ftqc_reference
def test_discard_quantum_environment_matches_source_dephasing_composition() -> None:
    pytest.importorskip("qualtran")
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import CNOT, DiscardQ, ZeroState

    builder = BloqBuilder()
    q = builder.add_register("q", 1)
    q, env = builder.add(CNOT(), ctrl=q, target=builder.add(ZeroState()))
    builder.add(DiscardQ(), q=env)
    source = builder.finalize(q=q)
    superoperator = torch.as_tensor(
        source.tensor_contract(superoperator=True), dtype=torch.complex128
    )
    with pytest.raises(NotImplementedError, match="classical simulation"):
        source.call_classically(q=1)

    psi = torch.tensor([1 + 2j, 2 - 3j], dtype=torch.complex128)
    psi /= psi.norm()
    initial = torch.zeros(4, dtype=psi.dtype)
    initial[::2] = psi
    circuit = fq.Circuit(2, inputs=initial, dtype=psi.dtype).cx(0, 1)
    plan = build_tensor_network(circuit, dtype=psi.dtype)
    density = torch.outer(psi, psi.conj())
    expected = torch.einsum("abij,ij->ab", superoperator, density)
    result = tensor_network_discard_qubit(plan, 1)
    torch.testing.assert_close(result[0], expected, rtol=0, atol=2e-12)
    assert result[0, 0, 1] == result[0, 1, 0] == 0


def test_discarding_the_only_wire_returns_the_trace() -> None:
    circuit = fq.Circuit(1, dtype=torch.complex128).h(0).s(0)
    plan = build_tensor_network(circuit, dtype=torch.complex128)
    result = tensor_network_discard_qubit(plan, 0)
    torch.testing.assert_close(result, torch.ones((1, 1, 1), dtype=torch.complex128))
