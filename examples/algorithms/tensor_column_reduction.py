"""Compose explicit squeeze/C/R preprocessing before the same register consumer."""

import torch

from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.column_reduction import reduce_tensor_columns
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.rank_simplification import (
    simplify_tensor_ranks,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    contract_register_tensors,
)
from flagquantum.simulation.tensor_network.simplification_axes import (
    squeeze_tensor_axes,
)


def main() -> None:
    """Keep a phase-trainable sparse ket while removing numerical internal axes."""
    phase = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    ket = torch.tensor([0, 1], dtype=torch.complex128) * torch.exp(1j * phase)
    matrix = torch.tensor([[1, 2], [3, 4]], dtype=torch.complex128)
    nodes = (
        TensorNetworkNode(ket.reshape(1, 2), (7, 0)),
        TensorNetworkNode(matrix, (0, 1)),
    )
    signature = Signature(
        (Register("q", RegisterType("bit", 1), side=RegisterSide.RIGHT),)
    )
    outgoing = {"q": ((1,),)}
    expected = contract_register_tensors(nodes, signature, {}, outgoing)
    squeezed = squeeze_tensor_axes(nodes, (1,))
    columns = reduce_tensor_columns(squeezed, (1,), atol=0)
    reduced = simplify_tensor_ranks(columns, (1,))
    actual = contract_register_tensors(reduced, signature, {}, outgoing)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    derivative = torch.autograd.grad(actual.imag.sum(), phase)[0]
    torch.testing.assert_close(derivative, 7 * torch.cos(phase))
    print(
        f"Squeeze/C/R: {sum(len(n.labels) for n in nodes)} -> {sum(len(n.labels) for n in reduced)} axis occurrences"
    )
    print("Output: exp(i * phase) * [3, 4]; fixed-support gradient preserved.")
    print("No physical gates, measurements or work qubits are allocated.")


if __name__ == "__main__":
    main()
