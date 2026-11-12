"""Supply a typed CNOT leaf from the existing native unitary numerical owner."""

from collections.abc import Mapping

import torch

from flagquantum.compiler.registers.arithmetic.addition_component import (
    addition_component_program,
)
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.component_tensor_expansion import (
    expand_components_for_tensor,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import FunctionRecord
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.component_dense import (
    contract_component_tensors,
)
from flagquantum.simulation.tensor_network.component_tensor_providers import (
    ComponentTensorProviders,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_tensor import (
    register_tensor_from_unitary,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
)
from flagquantum.simulation.unitary import get_unitary

Connections = Mapping[str, tuple[tuple[int, ...], ...]]


def main() -> None:
    """Replace one explicit leaf supplier without changing its typed graph."""
    record = addition_component_program(1)
    circuit = CircuitIR(2, (Instruction("cx", (0, 1)),))
    like = torch.empty((), dtype=torch.complex128)
    unitary = get_unitary(circuit, dtype=like.dtype, device=like.device)

    def supply(
        function: FunctionRecord, incoming: Connections, outgoing: Connections
    ) -> tuple[TensorNetworkNode, ...]:
        assert function.registers is not None
        signature = Signature(function.registers)
        return (
            bind_register_tensor(
                register_tensor_from_unitary(unitary, signature),
                signature,
                incoming,
                outgoing,
            ),
        )

    providers = ComponentTensorProviders({"add_cx": supply})
    selected, _ = expand_components_for_tensor(
        record, providers.has_custom_tensor, full_flatten=False
    )
    order = tuple(i for i, _, _ in iter_component_calls(selected))
    result = contract_component_tensors(selected, order, providers.supply, like)
    torch.testing.assert_close(result, unitary, atol=0, rtol=0)
    print(result)


if __name__ == "__main__":
    main()
