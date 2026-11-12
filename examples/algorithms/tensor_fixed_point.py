"""Explicit ADCR fixed point before the existing native register consumer."""

import torch

from examples.algorithms.tensor_rank_simplification import wide_state_network
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.full_simplification import (
    simplify_tensor_network,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    contract_register_tensors,
)
from flagquantum.simulation.tensor_network.register_tensor_indices import (
    register_contraction_groups,
)


def main() -> None:
    """Retain full complex values and a fixed-support phase gradient at 13 bits."""
    from dataclasses import replace

    nodes, outputs = wide_state_network(13)
    phase = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    nodes = (
        replace(nodes[0], tensor=nodes[0].tensor * torch.exp(1j * phase)),
        *nodes[1:],
    )
    signature = Signature(
        (Register("q", RegisterType("integer", 13), side=RegisterSide.RIGHT),)
    )
    outgoing = {"q": (outputs,)}
    groups = register_contraction_groups(nodes, signature, {}, outgoing)
    reduced = simplify_tensor_network(nodes, outputs, atol=0)
    actual = contract_register_tensors(
        reduced, signature, {}, outgoing, output_groups=groups
    )
    expected = contract_register_tensors(nodes, signature, {}, outgoing)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-14)
    torch.testing.assert_close(
        torch.autograd.grad(actual.imag.sum(), phase, retain_graph=True)[0],
        torch.autograd.grad(expected.imag.sum(), phase)[0],
    )
    print(
        f"ADCR fixed point: {len(nodes)} -> {len(reduced)} tensors; {actual.numel()} complex outputs"
    )
    print("Explicit preprocessing; no width estimator or physical gates added.")


if __name__ == "__main__":
    main()
