"""Compose explicit A/D/C/R before the existing native register consumer."""

import torch

from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.column_reduction import reduce_tensor_columns
from flagquantum.simulation.tensor_network.diagonal_reduction import (
    gauge_tensor_antidiagonals,
    reduce_tensor_diagonals,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.rank_simplification import (
    simplify_tensor_ranks,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    contract_register_tensors,
)
from flagquantum.simulation.tensor_network.register_tensor_indices import (
    register_contraction_groups,
)
from flagquantum.simulation.tensor_network.simplification_axes import (
    squeeze_tensor_axes,
)


def main() -> None:
    """Reduce a phase-trainable sparse numerical network; this is not gate synthesis."""
    phase = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    nodes = (
        TensorNetworkNode(torch.tensor([1.0, 2.0], dtype=torch.complex128), (0,)),
        TensorNetworkNode(
            torch.tensor([[0.0, 1.0], [2.0, 0.0]], dtype=torch.complex128)
            * torch.exp(1j * phase),
            (0, 1),
        ),
        TensorNetworkNode(
            torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.complex128), (1, 2)
        ),
    )
    signature = Signature(
        (Register("q", RegisterType("bit", 1), side=RegisterSide.RIGHT),)
    )
    outgoing = {"q": ((2,),)}
    groups = register_contraction_groups(nodes, signature, {}, outgoing)
    expected = contract_register_tensors(nodes, signature, {}, outgoing)
    selected = squeeze_tensor_axes(nodes, (2,))
    selected = gauge_tensor_antidiagonals(selected, (2,), atol=0)
    selected = reduce_tensor_diagonals(selected, (2,), atol=0)
    selected = reduce_tensor_columns(selected, (2,), atol=0)
    selected = simplify_tensor_ranks(selected, (2,))
    actual = contract_register_tensors(
        selected, signature, {}, outgoing, output_groups=groups
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.imag.sum(), phase)[0], 39 * torch.cos(phase)
    )
    print(f"A/D/C/R: {len(nodes)} -> {len(selected)} tensors; 5 -> 1 axis occurrences")
    print("Output: exp(i * phase) * [17, 22]; fixed-support gradient preserved.")
    print("Numerical flips are not physical X gates; no work qubits are allocated.")


if __name__ == "__main__":
    main()
