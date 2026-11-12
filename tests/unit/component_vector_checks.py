"""Shared observation and contraction helpers for classical-vector graph checks.

This module contains no collected scenarios and imports no other test modules.
Compiler constructs the graph; Simulation supplies and contracts its tensors.
"""

from collections.abc import Callable, Mapping

import pytest
import torch

from flagquantum.compiler.registers.component_basis_vectors import (
    integer_basis_function,
)
from flagquantum.compiler.registers.component_classical_vectors import (
    flank_component_with_classical_vectors,
)
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.component_dense import (
    contract_component_tensors,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_basis_vectors import (
    register_basis_vector_nodes,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
)

Connections = Mapping[str, tuple[tuple[int, ...], ...]]


def tracked_flank(
    monkeypatch: pytest.MonkeyPatch,
    record: ProgramRecord,
    inputs: Mapping[str, object],
    outputs: Mapping[str, object] | None = None,
    *,
    evaluate: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
    entry: str = "main",
    name: str = "classical_vectors",
) -> tuple[ProgramRecord, dict[str, tuple[int, bool]]]:
    """Observe exact basis declarations while using the real Compiler factory."""
    import flagquantum.compiler.registers.component_classical_vectors as module

    seen: dict[str, tuple[int, bool]] = {}

    def factory(
        name: str, dtype: RegisterType, value: int, *, state: bool = True
    ) -> FunctionRecord:
        seen[name] = value, state
        return integer_basis_function(name, dtype, value, state=state)

    monkeypatch.setattr(module, "integer_basis_function", factory)
    return (
        flank_component_with_classical_vectors(
            record, inputs, outputs, evaluate=evaluate, entry=entry, name=name
        ),
        seen,
    )


def contract(
    record: ProgramRecord, vectors: Mapping[str, tuple[int, bool]], child: torch.Tensor
) -> torch.Tensor:
    """Contract a classical_vectors wrapper with an explicitly supplied main tensor."""

    def supply(
        function: FunctionRecord, incoming: Connections, outgoing: Connections
    ) -> tuple[TensorNetworkNode, ...]:
        signature = Signature(function.registers or ())
        if function.name in vectors:
            value, state = vectors[function.name]
            return register_basis_vector_nodes(
                signature,
                outgoing if state else incoming,
                {"val": value},
                child,
                state=state,
            )
        assert function.name == "main"
        return (bind_register_tensor(child, signature, incoming, outgoing),)

    positions = tuple(
        i for i, _, _ in iter_component_calls(record, "classical_vectors")
    )
    return contract_component_tensors(
        record, positions, supply, child, "classical_vectors"
    )
