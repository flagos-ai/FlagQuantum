"""Explicit native tensor recipes for the factorized addition example.

Compiler selects the graph; these bindings supply its numerical leaves. No
whole-word Add matrix, foreign simulator or provider-name heuristic is used.
"""

from collections.abc import Mapping

import torch

from flagquantum.compiler.registers.component_rebindings import (
    join_word_function,
    split_word_function,
)
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.classical_and import and_classical_values
from flagquantum.simulation.classical_controlled_bits import cnot_classical_values
from flagquantum.simulation.tensor_network.classical_action_tensor import (
    classical_action_tensor,
)
from flagquantum.simulation.tensor_network.component_tensor_providers import (
    ComponentTensorProviders,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_basis_vectors import (
    register_basis_vector_nodes,
)
from flagquantum.simulation.tensor_network.register_identity import (
    register_bit_wiring_nodes,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
)

Connections = Mapping[str, tuple[tuple[int, ...], ...]]


def addition_tensor_providers(
    record: ProgramRecord,
    vectors: Mapping[str, tuple[int, bool]],
    like: torch.Tensor,
) -> ComponentTensorProviders[[Connections, Connections]]:
    """Bind this example's explicitly supported leaves in one immutable mapping.

    Vectors map exact preparation/effect names to (unsigned value, is_state).
    The caller owns their physical clean-release promise. Split/join declarations
    must equal the existing Compiler word-rebinding definitions; all other unknown
    functions remain unbound and fail if selected. Numerical data retains like's
    dtype/device. This helper is scoped to addition_component_program graphs.
    """
    bit = RegisterType("bit", 1)
    and_signature = Signature(
        (Register("ctrl", bit, (2,)), Register("target", bit, side=RegisterSide.RIGHT))
    )
    cnot_signature = Signature((Register("ctrl", bit), Register("target", bit)))
    compute = torch.as_tensor(
        classical_action_tensor(
            and_signature,
            lambda **values: tuple(and_classical_values(values["ctrl"]).values()),
            operation_name="and",
        ),
        dtype=like.dtype,
        device=like.device,
    ).reshape(8, 4)
    cnot = torch.as_tensor(
        classical_action_tensor(
            cnot_signature,
            lambda **values: cnot_classical_values(values["ctrl"], values["target"]),
            operation_name="cx",
        ),
        dtype=like.dtype,
        device=like.device,
    )
    matrices = {"add_cx": cnot, "add_and": compute, "add_and_adjoint": compute.mH}
    basis = dict(vectors)
    wiring = {
        function.name
        for function in record.functions
        if function.registers is not None
        and len(function.registers) == 2
        and len(function.blocks) == 1
        and function.blocks[0].operations
        and all(
            operation.opcode in {"split_register", "join_register"}
            for operation in function.blocks[0].operations
        )
        and function
        in (
            split_word_function(function.name, function.registers[0].dtype),
            join_word_function(function.name, function.registers[-1].dtype),
        )
    }
    if (set(basis) & set(matrices)) or (set(basis) & wiring):
        raise ValueError(
            "basis vector names must not override addition or wiring recipes"
        )

    def supply(
        function: FunctionRecord, incoming: Connections, outgoing: Connections
    ) -> tuple[TensorNetworkNode, ...]:
        if function.registers is None:
            raise ValueError("addition tensor recipes require typed functions")
        signature = Signature(function.registers)
        if function.name in basis:
            value, state = basis[function.name]
            return register_basis_vector_nodes(
                signature,
                outgoing if state else incoming,
                {"val": value},
                like,
                state=state,
            )
        if function.name in matrices:
            return (
                bind_register_tensor(
                    matrices[function.name], signature, incoming, outgoing
                ),
            )
        return register_bit_wiring_nodes(signature, incoming, outgoing, like)

    names = (set(matrices) | set(basis) | wiring) & {f.name for f in record.functions}
    return ComponentTensorProviders(dict.fromkeys(sorted(names), supply))
