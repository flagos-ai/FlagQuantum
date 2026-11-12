"""Reduce a 13-output numerical network without changing its dense consumer."""

import torch

from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.rank_simplification import (
    simplify_tensor_ranks,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    contract_register_tensors,
)


def wide_state_network(
    width: int = 13,
) -> tuple[tuple[TensorNetworkNode, ...], tuple[int, ...]]:
    """Return a bounded complex state tensor followed by two one-bit transfers.

    This is a numerical example, not a preparation circuit or gate estimate.
    Output labels are MSB-first. Storage is exponential and bounded to 14 bits.
    """
    if type(width) is not int or not 2 <= width <= 14:
        raise ValueError("wide state example requires an integer width from 2 to 14")
    generator = torch.Generator().manual_seed(5)
    initial = torch.randn((2,) * width, dtype=torch.complex128, generator=generator)
    middle = torch.randn((2, 2), dtype=torch.complex128, generator=generator)
    final = torch.randn((2, 2), dtype=torch.complex128, generator=generator)
    labels = tuple(range(width - 1))
    return (
        TensorNetworkNode(initial, (*labels, width)),
        TensorNetworkNode(middle, (width, width + 1)),
        TensorNetworkNode(final, (width + 1, width + 2)),
    ), (*labels, width + 2)


def main() -> None:
    """Use the same register contraction before and after explicit R selection."""
    nodes, outputs = wide_state_network()
    signature = Signature(
        (Register("q", RegisterType("integer", 13), side=RegisterSide.RIGHT),)
    )
    outgoing = {"q": (outputs,)}
    original = contract_register_tensors(nodes, signature, {}, outgoing)
    reduced = simplify_tensor_ranks(nodes, outputs)
    result = contract_register_tensors(reduced, signature, {}, outgoing)
    torch.testing.assert_close(result, original, rtol=0, atol=2e-14)
    print(
        f"Rank pass: {len(nodes)} -> {len(reduced)} tensors; {result.numel()} complex outputs"
    )
    print("No physical gates or work qubits are allocated by this numerical pass.")


if __name__ == "__main__":
    main()
