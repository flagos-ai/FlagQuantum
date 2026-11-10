"""Measurement copy leaves match source tensors, channels and branch amplitudes."""

from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.simulation.tensor_network.conjugate_pairs import tensor_conjugate_pairs
from flagquantum.simulation.tensor_network.measurement_copy import (
    z_measurement_copy_node,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_superoperator_binding import (
    contract_register_superoperator,
)
from flagquantum.simulation.tensor_network.trace_indices import trace_conjugate_indices
from tests.unit.test_register_tensor import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
@pytest.mark.parametrize("discard", [False, True])
def test_source_data_axis_order_channel_replacement_and_complex_branches(
    dtype: torch.dtype, discard: bool
) -> None:
    from qualtran import BloqBuilder, CBit, QBit, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import Discard, MeasureZ
    from qualtran.simulation.tensor import DiscardInd

    items = MeasureZ().my_tensors({"q": "input"}, {"c": "result"})
    assert items[0].inds[:2] == (("input", 0), ("result", 0))
    assert isinstance(items[1], DiscardInd)
    node = z_measurement_copy_node(3, 17, 41, torch.empty((), dtype=dtype))
    assert node.labels == (3, 17, 41) and node.tensor.dtype == dtype
    assert node.tensor.device.type == "cpu" and not node.tensor.requires_grad
    np.testing.assert_array_equal(node.tensor.numpy(), items[0].data)
    # The third source leg is a fresh hidden environment, identified by its sentinel.
    assert items[0].inds[2] == items[1].ind_tuple
    bb, inputs = BloqBuilder.from_signature(
        SourceSignature(
            [SourceRegister("q", QBit(), side=Side.LEFT)]
            + ([] if discard else [SourceRegister("c", CBit(), side=Side.RIGHT)])
        )
    )
    c = bb.add(MeasureZ(), q=inputs["q"])
    if discard:
        bb.add(Discard(), c=c)
        original = bb.finalize()
    else:
        original = bb.finalize(c=c)
    sig = native_signature(original.signature)
    labels = (41, 17) if discard else (41,)
    out = {} if discard else {"c": ((17,),)}

    def consume(leaf: TensorNetworkNode) -> torch.Tensor:
        return contract_register_superoperator(
            trace_conjugate_indices(tensor_conjugate_pairs((leaf,)), labels),
            sig,
            {"q": ((3,),)},
            out,
        )

    channel = consume(node)
    np.testing.assert_array_equal(
        channel.numpy(), original.tensor_contract(superoperator=True)
    )
    replacement = TensorNetworkNode(
        torch.as_tensor(items[0].data.real, dtype=dtype), node.labels
    )
    torch.testing.assert_close(consume(replacement), channel)
    theta = torch.tensor(0.29, dtype=torch.float64, requires_grad=True)
    state = torch.tensor([1 + 2j, 3 - 1j, -2 + 1j, 1 + 4j], dtype=torch.complex128)
    state = state / state.norm() * torch.exp(1j * theta)
    amplitudes = torch.einsum(
        "qce,qs->ces", node.tensor.to(torch.complex128), state.reshape(2, 2)
    )
    for outcome in (0, 1):
        torch.testing.assert_close(
            amplitudes[outcome, outcome], state.reshape(2, 2)[outcome]
        )
        assert torch.count_nonzero(amplitudes[outcome, 1 - outcome]) == 0
    (gradient,) = torch.autograd.grad(amplitudes.real.sum(), (theta,))
    torch.testing.assert_close(gradient, -amplitudes.imag.sum())
    density = torch.outer(state.detach(), state.detach().conj()).reshape(2, 2, 2, 2)
    # Keep an entangled spectator; this checks complex coherences within every outcome.
    if discard:
        reduced = torch.einsum("ij,isjt->st", channel.to(torch.complex128), density)
        expected = torch.einsum("isis->", density)  # Only its trace is used below.
        torch.testing.assert_close(torch.trace(reduced), expected)
        torch.testing.assert_close(
            reduced,
            state.detach().reshape(2, 2).T @ state.detach().reshape(2, 2).conj(),
        )
    else:
        result = torch.einsum("abij,isjt->asbt", channel.to(torch.complex128), density)
        for outcome in (0, 1):
            branch = amplitudes[outcome, outcome].detach()
            torch.testing.assert_close(
                result[outcome, :, outcome, :], torch.outer(branch, branch.conj())
            )
            assert torch.count_nonzero(result[outcome, :, 1 - outcome, :]) == 0


@pytest.mark.parametrize(
    "labels",
    [
        (-1, 2, 3),
        (1, -2, 3),
        (1, 2, -3),
        (True, 2, 3),
        (1, False, 3),
        (1, 2, True),
        (1.0, 2, 3),
        (1, 1, 3),
        (1, 2, 1),
        (1, 2, 2),
    ],
)
def test_invalid_labels_fail_before_tensor_construction(
    labels: tuple[Any, Any, Any],
) -> None:
    with pytest.raises(ValueError, match="measurement copy labels"):
        z_measurement_copy_node(*labels, torch.empty(()))


def test_coherent_copy_requires_environment_trace_to_remove_outcome_coherence() -> None:
    node = z_measurement_copy_node(3, 17, 41, torch.empty((), dtype=torch.complex128))
    plus = torch.ones(2, dtype=torch.complex128) / np.sqrt(2)
    copied = torch.einsum("qce,q->ce", node.tensor, plus)
    torch.testing.assert_close(
        copied, torch.eye(2, dtype=torch.complex128) / np.sqrt(2)
    )
    assert (
        torch.linalg.matrix_rank(torch.outer(copied.flatten(), copied.flatten().conj()))
        == 1
    )
    reduced = copied @ copied.conj().T
    torch.testing.assert_close(reduced, torch.eye(2, dtype=torch.complex128) / 2)
    assert torch.linalg.matrix_rank(reduced) == 2
