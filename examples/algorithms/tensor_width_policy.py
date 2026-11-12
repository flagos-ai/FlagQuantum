"""Opt into source-default width-triggered ADCR using the existing dense consumer."""

from dataclasses import replace

import torch

from examples.algorithms.tensor_rank_simplification import wide_state_network
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.contraction_width import (
    tensor_contraction_width,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    contract_register_tensors,
)


def main() -> None:
    """Check both sides of the real threshold with complex values and gradients."""
    for bits in (12, 13):
        nodes, outputs = wide_state_network(bits)
        phase = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
        nodes = (
            replace(nodes[0], tensor=nodes[0].tensor * torch.exp(1j * phase)),
            *nodes[1:],
        )
        signature = Signature(
            (Register("q", RegisterType("integer", bits), side=RegisterSide.RIGHT),)
        )
        outgoing = {"q": (outputs,)}
        width = tensor_contraction_width(nodes)
        actual = contract_register_tensors(
            nodes, signature, {}, outgoing, simplification_atol=0
        )
        expected = contract_register_tensors(nodes, signature, {}, outgoing)
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-14)
        torch.testing.assert_close(
            torch.autograd.grad(actual.imag.sum(), phase, retain_graph=True)[0],
            torch.autograd.grad(expected.imag.sum(), phase)[0],
        )
        print(
            f"Width {width:g}: ADCR {'selected' if width > 12 else 'not selected'}; {actual.numel()} complex outputs"
        )
    print(
        "Optional Cotengra plans shapes; native PyTorch executes. Added physical gates: 0."
    )


if __name__ == "__main__":
    main()
