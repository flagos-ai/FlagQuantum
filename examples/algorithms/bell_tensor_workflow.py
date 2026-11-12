# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Composition adapted from Qualtran v0.7.0 simulation/tensor.ipynb.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/tensor.ipynb

"""Bell preparation, explicit custom tensors and zero initialization in native IR."""

from collections.abc import Callable, Mapping
from dataclasses import replace

import torch

from examples.algorithms.classical_tensor_crosscheck import crosscheck_program
from flagquantum.compiler.registers.component_basis_vectors import (
    integer_basis_function,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.component_tensor_expansion import (
    expand_component_root_for_tensor,
)
from flagquantum.compiler.registers.layout.component_workspace import (
    allocate_word_function,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.component_tensor_nodes import (
    component_tensor_nodes_with_labels,
)
from flagquantum.simulation.tensor_network.component_tensor_providers import (
    ComponentTensorProviders,
)
from flagquantum.simulation.tensor_network.controlled_not import controlled_not_nodes
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_tensor import (
    register_tensor_from_unitary,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
    contract_register_tensors,
)
from flagquantum.simulation.tensor_network.register_zero_state import (
    register_zero_state_nodes,
)
from flagquantum.simulation.unitary import get_unitary

Connections = Mapping[str, tuple[tuple[int, ...], ...]]


def _definitions() -> tuple[FunctionRecord, ...]:
    bit = RegisterType("bit", 1)
    h = bit_gate_function(
        "h", (Register("q", bit),), CircuitIR(1, (Instruction("h", (0,)),))
    )
    zero = integer_basis_function("zero", bit, 0)
    # Reuse the canonical ket's allocation/split/join scaffold. Its single X
    # becomes H: this is PlusState's actual preparation, not a reset or matrix.
    one = integer_basis_function("plus", bit, 1)
    plus = replace(
        one,
        blocks=tuple(
            replace(
                b,
                operations=tuple(
                    replace(op, opcode="h") if op.opcode == "x" else op
                    for op in b.operations
                ),
            )
            for b in one.blocks
        ),
    )
    cnot = replace(crosscheck_program("cnot").functions[0], name="cnot")
    return h, zero, plus, cnot


def bell_program(*, prepared: bool) -> ProgramRecord:
    """Build the notebook's BellState or its two-input H/CNOT circuit.

    Prepared outputs q0/q1 are owned logical results, not zero scratch.
    The unitary uses source port names q1/q2 in MSB-first order. Physical
    allocation remains explicit in the definitions and needs a caller's pool.
    """
    bit = RegisterType("bit", 1)
    names = ("q0", "q1") if prepared else ("q1", "q2")
    side = RegisterSide.RIGHT if prepared else RegisterSide.THRU
    builder = ComponentProgramBuilder(
        Signature(Register(n, bit, side=side) for n in names),
        _definitions(),
        name="bell",
    )
    if prepared:
        left = builder.call_dict("plus", {})["val"]
        right = builder.call_dict("zero", {})["val"]
    else:
        left = builder.call_dict("h", {"q": builder.inputs[names[0]]})["q"]
        right = builder.inputs[names[1]]
    result = builder.call_dict("cnot", {"ctrl": left, "target": right})
    return builder.finalize({names[0]: result["ctrl"], names[1]: result["target"]})


def bell_providers(
    record: ProgramRecord,
    like: torch.Tensor,
    *,
    factorized_cnot: bool = True,
) -> ComponentTensorProviders[[Connections, Connections, Callable[[], int]]]:
    """Bind explicit primitive definitions; swap CNOT's supplied representation.

    Dense CNOT is the notebook's MyCNOT matrix supplied through the native
    unitary owner; factored CNOT uses the exact COPY/XOR owner. Both execute
    the same physical body. No whole-Bell tensor is registered: unexpanded
    Bell roots fail with the ordinary missing-provider diagnostic.
    """
    expected = {f.name: f for f in _definitions()}
    zeros = {
        f.name: f
        for f in record.functions
        if f.registers is not None
        and len(f.registers) == 1
        and f == allocate_word_function(f.name, f.registers[0].dtype, False)
    }
    h = get_unitary(
        CircuitIR(1, (Instruction("h", (0,)),)), dtype=like.dtype, device=like.device
    )
    cx = get_unitary(
        CircuitIR(2, (Instruction("cx", (0, 1)),)), dtype=like.dtype, device=like.device
    )

    def supply(
        function: FunctionRecord,
        incoming: Connections,
        outgoing: Connections,
        fresh_label: Callable[[], int],
    ) -> tuple[TensorNetworkNode, ...]:
        if function != (expected | zeros)[function.name]:
            raise ValueError("Bell example requires its declared primitive definition")
        assert function.registers is not None
        signature = Signature(function.registers)
        if function.name == "zero" or function.name in zeros:
            return register_zero_state_nodes(signature, outgoing, like)
        if function.name == "cnot" and factorized_cnot:
            return controlled_not_nodes(incoming, outgoing, fresh_label, like)
        matrix = cx if function.name == "cnot" else h
        return (
            bind_register_tensor(
                register_tensor_from_unitary(matrix, signature),
                signature,
                incoming,
                outgoing,
            ),
        )

    return ComponentTensorProviders(dict.fromkeys((*expected, *zeros), supply))


def contract_bell_program(
    record: ProgramRecord,
    like: torch.Tensor,
    *,
    entry: str = "bell",
    factorized_cnot: bool = True,
    full_flatten: bool = True,
) -> torch.Tensor:
    """Wrap and expand the chosen root, then wire and contract explicit leaves."""
    providers = bell_providers(record, like, factorized_cnot=factorized_cnot)
    selected, _ = expand_component_root_for_tensor(
        record, providers.has_custom_tensor, entry, full_flatten=full_flatten
    )
    order = tuple(i for i, _, _ in iter_component_calls(selected, "tensor_root"))
    nodes, incoming, outgoing = component_tensor_nodes_with_labels(
        selected, order, providers.supply, like, "tensor_root"
    )
    return contract_register_tensors(
        nodes, Signature(selected.functions[0].registers or ()), incoming, outgoing
    )


def main() -> None:
    """Run both original Bell paths and the dense custom-CNOT replacement."""
    from flagquantum.compiler.registers.component_zero_initialization import (
        initialize_component_from_zero,
    )

    like = torch.empty((), dtype=torch.complex128)
    state = contract_bell_program(bell_program(prepared=True), like)
    unitary = bell_program(prepared=False)
    matrix = contract_bell_program(unitary, like)
    initialized = initialize_component_from_zero(unitary, "bell")
    from_zero = contract_bell_program(initialized, like, entry="from_zero")
    dense = contract_bell_program(unitary, like, factorized_cnot=False)
    torch.testing.assert_close(state, from_zero, rtol=0, atol=0)
    torch.testing.assert_close(matrix, dense, rtol=0, atol=0)
    print(f"Bell state: {state.tolist()}; circuit shape {tuple(matrix.shape)}")


if __name__ == "__main__":
    main()
