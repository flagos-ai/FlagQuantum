"""X measurement tensors preserve original environment basis and branch phases."""

from collections import Counter

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_measurement import measure_x_word_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.simulation.matrices import H_MATRIX
from flagquantum.simulation.tensor_network.conjugate_pairs import tensor_conjugate_pairs
from flagquantum.simulation.tensor_network.measurement_copy import (
    x_measurement_node,
    z_measurement_copy_node,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.path_search import _contract_nodes_greedy
from flagquantum.simulation.tensor_network.register_superoperator_binding import (
    contract_register_superoperator,
)
from flagquantum.simulation.tensor_network.trace_indices import trace_conjugate_indices
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_tensor import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
@pytest.mark.parametrize("discard", [False, True])
def test_original_x_tensor_composed_z_channel_replacement_and_all_complex_branches(
    dtype: torch.dtype, discard: bool
) -> None:
    from qualtran import BloqBuilder, CBit, QBit, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import Discard, MeasureX

    atol = 2e-6 if dtype in {torch.float32, torch.complex64} else 2e-14
    source_leaf, sentinel = MeasureX().my_tensors({"q": "in"}, {"c": "out"})
    assert source_leaf.inds == (("in", 0), sentinel.ind_tuple, ("out", 0))
    node = x_measurement_node(3, 17, 41, torch.empty((), dtype=dtype))
    assert node.labels == (3, 41, 17) and node.tensor.dtype == dtype
    np.testing.assert_array_equal(node.tensor.numpy(), source_leaf.data)
    bb, inputs = BloqBuilder.from_signature(
        SourceSignature(
            [SourceRegister("q", QBit(), side=Side.LEFT)]
            + ([] if discard else [SourceRegister("c", CBit(), side=Side.RIGHT)])
        )
    )
    c = bb.add(MeasureX(), q=inputs["q"])
    if discard:
        bb.add(Discard(), c=c)
        original = bb.finalize()
    else:
        original = bb.finalize(c=c)
    sig = native_signature(original.signature)
    traced = (41, 17) if discard else (41,)
    out = {} if discard else {"c": ((17,),)}

    def consume(nodes: tuple[TensorNetworkNode, ...]) -> torch.Tensor:
        return contract_register_superoperator(
            trace_conjugate_indices(tensor_conjugate_pairs(nodes), traced),
            sig,
            {"q": ((3,),)},
            out,
        )

    channel = consume((node,))
    np.testing.assert_allclose(
        channel.numpy(), original.tensor_contract(superoperator=True), atol=atol, rtol=0
    )
    replacement = TensorNetworkNode(
        torch.as_tensor(source_leaf.data.real, dtype=dtype), node.labels
    )
    torch.testing.assert_close(consume((replacement,)), channel, atol=atol, rtol=0)
    # Independent unfused H -> Z copy -> H on the explicit environment.
    h = H_MATRIX.real.to(dtype=dtype)
    raw = (
        TensorNetworkNode(h, (3, 59)),
        z_measurement_copy_node(59, 17, 67, torch.empty((), dtype=dtype)),
        TensorNetworkNode(h, (67, 41)),
    )
    contracted, _ = _contract_nodes_greedy(raw, (3, 41, 17))
    torch.testing.assert_close(contracted, node.tensor, atol=atol, rtol=0)
    torch.testing.assert_close(consume(raw), channel, atol=atol, rtol=0)
    # Omitting the environmental H changes the raw instrument but not the traced channel.
    unrotated = (
        TensorNetworkNode(h, (3, 59)),
        z_measurement_copy_node(59, 17, 41, torch.empty((), dtype=dtype)),
    )
    torch.testing.assert_close(consume(unrotated), channel, atol=atol, rtol=0)

    theta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    phase = torch.exp(1j * theta)
    state = torch.stack((phase, 2 + phase * 0, 1j * phase, -3j + phase * 0)).to(
        torch.complex128
    )
    state = state / state.norm()
    instrument = torch.einsum(
        "qec,qs->ces", node.tensor.to(torch.complex128), state.reshape(2, 2)
    )
    expected = torch.einsum(
        "qec,qs->ces", torch.from_numpy(source_leaf.data), state.reshape(2, 2)
    )
    torch.testing.assert_close(instrument, expected, atol=atol, rtol=0)
    (grad,) = torch.autograd.grad(instrument.real.sum(), (theta,), retain_graph=True)
    (source_grad,) = torch.autograd.grad(expected.real.sum(), (theta,))
    torch.testing.assert_close(grad, source_grad, atol=atol, rtol=0)
    # Physical H+MeasureZ retains Z on the consumed wire. Rotate that wire only
    # for comparing full source instrument amplitudes, never as an extra gate.
    bound = bind_register_program(
        ProgramRecord((measure_x_word_function("main"),)), ((0,),), clean=(2,)
    )
    assert Counter(i.name for i in bound.circuit.instructions) == {"h": 1, "measure": 1}
    gates = (
        Circuit(3)
        .append(FieldOperation(3, bound.circuit.instructions))
        .to_ir()
        .instructions
    )
    assert gates == bound.circuit.instructions
    initial = torch.zeros(8, dtype=torch.complex128)
    initial[::2] = state.detach()
    total = torch.zeros((2, 2), dtype=torch.complex128)
    for outcome in (0, 1):
        branch = forced_branch(initial, gates, (outcome,)).reshape(2, 2, 2)
        assert torch.count_nonzero(branch[:, :, 1]) == 0
        restored = H_MATRIX @ branch[:, :, 0]
        torch.testing.assert_close(
            restored, instrument[outcome].detach(), atol=atol, rtol=0
        )
        total += restored @ restored.conj().T
    torch.testing.assert_close(
        torch.trace(total), torch.ones((), dtype=torch.complex128), atol=atol, rtol=0
    )


def test_x_node_reuses_z_label_refusals() -> None:
    for labels in ((-1, 2, 3), (1, 1, 3), (True, 2, 3)):
        with pytest.raises(ValueError, match="measurement copy labels"):
            x_measurement_node(*labels, torch.empty(()))
