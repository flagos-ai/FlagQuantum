"""Compose the four classical/tensor examples using native executable graphs.

Small primitive leaves use native unitary matrices, retaining their full phase.
Add uses its existing factorized graph and logical AND tensors, never a whole-word
truth table. Compiler selects calls; Simulation supplies and contracts the data.
"""

from collections.abc import Mapping

import numpy as np
import torch

from examples.algorithms.factorized_addition_tensors import addition_tensor_providers
from flagquantum.compiler.registers.arithmetic.addition_component import (
    addition_component_program,
)
from flagquantum.compiler.registers.component_basis_vectors import (
    integer_basis_function_data,
)
from flagquantum.compiler.registers.component_classical_vectors import (
    flank_component_with_classical_vectors,
)
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.component_tensor_expansion import (
    expand_components_for_tensor,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.classical_addition import register_add_classical_values
from flagquantum.simulation.classical_controlled_bits import (
    cnot_classical_values,
    toffoli_classical_values,
)
from flagquantum.simulation.classical_pauli_values import x_classical_value
from flagquantum.simulation.tensor_network.component_dense import (
    contract_component_tensors,
)
from flagquantum.simulation.tensor_network.component_tensor_providers import (
    ComponentTensorProviders,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_basis_vectors import (
    register_basis_vector_nodes,
)
from flagquantum.simulation.tensor_network.register_tensor import (
    register_tensor_from_unitary,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
)
from flagquantum.simulation.unitary import get_unitary

Connections = Mapping[str, tuple[tuple[int, ...], ...]]


def crosscheck_program(kind: str, width: int = 5) -> ProgramRecord:
    """Build the real X/CNOT/Toffoli primitive or factorized unsigned Add graph."""
    if kind == "add":
        return addition_component_program(width)
    bit = RegisterType("bit", 1)
    registers: tuple[Register, ...]
    if kind == "xgate":
        registers = (Register("q", bit),)
        instruction = Instruction("x", (0,))
    elif kind == "cnot":
        registers = (Register("ctrl", bit), Register("target", bit))
        instruction = Instruction("cx", (0, 1))
    elif kind == "toffoli":
        registers = (Register("ctrl", bit, (2,)), Register("target", bit))
        instruction = Instruction("ccx", (0, 1, 2))
    else:
        raise ValueError(f"unknown classical cross-check example {kind!r}")
    return ProgramRecord(
        (
            bit_gate_function(
                "main", registers, CircuitIR(len(instruction.wires), (instruction,))
            ),
        )
    )


def _integer(values: Mapping[str, object], name: str) -> int:
    value = values[name]
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer basis value")
    return int(value)


def crosscheck_values(
    kind: str, values: Mapping[str, object], width: int = 5
) -> dict[str, object]:
    """Reuse existing phase-free classical callbacks to predict the output bras."""
    if kind == "xgate":
        return {"q": x_classical_value(_integer(values, "q"))}
    if kind == "cnot":
        ctrl, target = cnot_classical_values(
            _integer(values, "ctrl"), _integer(values, "target")
        )
        return {"ctrl": ctrl, "target": target}
    if kind == "toffoli":
        controls, target = toffoli_classical_values(
            np.asarray(values["ctrl"]), _integer(values, "target")
        )
        return {"ctrl": controls, "target": target}
    if kind == "add":
        a, b = register_add_classical_values(
            _integer(values, "a"), _integer(values, "b"), width
        )
        return {"a": a, "b": b}
    raise ValueError(f"unknown classical cross-check example {kind!r}")


def crosscheck_providers(
    record: ProgramRecord, kind: str, like: torch.Tensor
) -> ComponentTensorProviders[[Connections, Connections]]:
    """Compose declared primitive/AND recipes and canonical integer vectors.

    The primitive unitary is obtained from the actual stored main body after
    Compiler physical binding. Add retains its factorized suppliers. No reference
    library, test fixture, constructor interception or name-derived value is used.
    """
    if kind == "add":
        return addition_tensor_providers(record, None, like)
    if kind not in {"xgate", "cnot", "toffoli"}:
        raise ValueError(f"unknown classical cross-check example {kind!r}")
    root = next(function for function in record.functions if function.name == "main")
    if root.registers is None:
        raise ValueError("cross-check primitives require typed ports")
    words = []
    offset = 0
    for register in Signature(root.registers).lefts():
        for _ in register.all_idxs():
            width = int(register.bitsize)
            words.append(tuple(range(offset, offset + width)))
            offset += width
    bound = bind_register_program(record, tuple(words))
    unitary = get_unitary(bound.circuit, dtype=like.dtype, device=like.device)
    vectors = {
        f.name: data
        for f in record.functions
        if (data := integer_basis_function_data(f)) is not None
    }

    def supply(
        function: FunctionRecord, incoming: Connections, outgoing: Connections
    ) -> tuple[TensorNetworkNode, ...]:
        assert function.registers is not None
        signature = Signature(function.registers)
        if function.name == "main":
            return (
                bind_register_tensor(
                    register_tensor_from_unitary(unitary, signature),
                    signature,
                    incoming,
                    outgoing,
                ),
            )
        value, state = vectors[function.name]
        return register_basis_vector_nodes(
            signature,
            outgoing if state else incoming,
            {"val": value},
            like,
            state=state,
        )

    return ComponentTensorProviders(dict.fromkeys(("main", *vectors), supply))


def prepare_classical_crosscheck(
    kind: str,
    inputs: Mapping[str, object],
    outputs: Mapping[str, object] | None = None,
    *,
    width: int = 5,
) -> ProgramRecord:
    """Prepare the closed typed graph without selecting numerical suppliers."""
    record = crosscheck_program(kind, width)
    return flank_component_with_classical_vectors(
        record,
        inputs,
        outputs,
        evaluate=lambda values: crosscheck_values(kind, values, width),
    )


def contract_classical_crosscheck(
    kind: str,
    inputs: Mapping[str, object],
    outputs: Mapping[str, object] | None = None,
    *,
    width: int = 5,
) -> torch.Tensor:
    """Contract the real prepared component with inferred or explicit effects.

    This small diagnostic uses the existing greedy contraction path. For wide
    Add performance, use its documented explicit cotengra path instead. The
    returned scalar is a complex amplitude, not a normalized probability.
    """
    closed = prepare_classical_crosscheck(kind, inputs, outputs, width=width)
    like = torch.empty((), dtype=torch.complex128)
    providers = crosscheck_providers(closed, kind, like)
    selected, _ = expand_components_for_tensor(
        closed, providers.has_custom_tensor, "classical_vectors", full_flatten=False
    )
    order = tuple(i for i, _, _ in iter_component_calls(selected, "classical_vectors"))
    return contract_component_tensors(
        selected, order, providers.supply, like, "classical_vectors"
    )


def main() -> None:
    """Run the original four concrete numerical examples without foreign objects."""
    examples: tuple[tuple[str, dict[str, object]], ...] = (
        ("xgate", {"q": 0}),
        ("cnot", {"ctrl": 1, "target": 0}),
        ("toffoli", {"ctrl": [1, 1], "target": 0}),
        ("add", {"a": 2, "b": 3}),
    )
    for kind, inputs in examples:
        result = contract_classical_crosscheck(kind, inputs)
        torch.testing.assert_close(
            result, torch.ones((), dtype=result.dtype), atol=2e-12, rtol=0
        )
        print(f"{kind}: {result.item()}")
    wrong = contract_classical_crosscheck("xgate", {"q": 0}, {"q": 0})
    assert wrong.item() == 0


if __name__ == "__main__":
    main()
