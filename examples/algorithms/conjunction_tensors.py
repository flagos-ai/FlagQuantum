# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Composition follows Qualtran v0.7.0 MultiAnd._decompose_via_tree.

"""Compose real predicate bodies and explicit source-ordered And tensor leaves."""

from collections.abc import Mapping

import numpy as np
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.control.component_predicate import (
    control_predicate_program,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.control.multi_control import multi_and_stages
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.component_dense import (
    contract_component_tensors,
)
from flagquantum.simulation.tensor_network.component_tensor_providers import (
    ComponentTensorProviders,
)
from flagquantum.simulation.tensor_network.conjunction import conjunction_tensor_node
from flagquantum.simulation.tensor_network.models import TensorNetworkNode

Connections = Mapping[str, tuple[tuple[int, ...], ...]]


def conjunction_program(values: tuple[int, ...]) -> ProgramRecord:
    """Return ctrl[n] -> ctrl[n], junk[n-2], target with live prefix outputs.

    Row-major bit arrays match the original MultiAnd ports (n >= 3); n=2
    represents And without an empty junk register. Each leaf reuses the existing
    executable predicate constructor. No duplicated physical gate emitter or
    foreign object enters this example. Returned logical outputs belong to the
    caller and require explicit later erasure.
    """
    stages = tuple(multi_and_stages(values))
    bit = RegisterType("bit", 1)
    patterns = tuple(dict.fromkeys(stage[3] for stage in stages))
    leaves = {
        pattern: control_predicate_program(
            (Register("ctrl", bit, (2,)),),
            (ControlRegister(1, pattern),),
            name=f"and_{pattern[0]}{pattern[1]}",
        )
        for pattern in patterns
    }
    n = len(values)
    signature = Signature(
        (
            Register("ctrl", bit, (n,)),
            *(
                (Register("junk", bit, (n - 2,), side=RegisterSide.RIGHT),)
                if n > 2
                else ()
            ),
            Register("target", bit, side=RegisterSide.RIGHT),
        )
    )
    builder = ComponentProgramBuilder(
        signature,
        (f for leaf in leaves.values() for f in leaf.functions),
        name="multi_and",
    )
    wires = dict(enumerate(np.asarray(builder.inputs["ctrl"]).flat))
    for left, right, target, pattern in stages:
        result = builder.call_dict(
            leaves[pattern].functions[0].name,
            {
                "ctrl": np.array([wires[left], wires[right]], dtype=object),
            },
        )
        controls = np.asarray(result["ctrl"])
        wires[left], wires[right] = controls[0], controls[1]
        output = result["target"]
        assert isinstance(output, RegisterBinding)
        wires[target] = output
    outputs = {
        "ctrl": np.array([wires[i] for i in range(n)], dtype=object),
        "target": wires[2 * n - 2],
    }
    if n > 2:
        outputs["junk"] = np.array(
            [wires[i] for i in range(n, 2 * n - 2)], dtype=object
        )
    return builder.finalize(outputs)


def conjunction_providers(
    values: tuple[int, ...],
    like: torch.Tensor,
) -> ComponentTensorProviders[[Connections, Connections]]:
    """Bind exact known predicate definitions to their activation patterns."""
    record = conjunction_program(values)
    definitions = {f.name: f for f in record.functions}
    patterns = {f"and_{p[0]}{p[1]}": p for *_, p in multi_and_stages(values)}

    def supply(
        function: FunctionRecord, incoming: Connections, outgoing: Connections
    ) -> tuple[TensorNetworkNode, ...]:
        if function != definitions[function.name]:
            raise ValueError("conjunction example requires its declared predicate body")
        return (
            conjunction_tensor_node(
                incoming, outgoing, like, values=patterns[function.name]
            ),
        )

    return ComponentTensorProviders(dict.fromkeys(patterns, supply))


def main() -> None:
    """Contract the original notebook's four-control, three-leaf ladder."""
    values = (1, 1, 1, 1)
    record = conjunction_program(values)
    like = torch.empty((), dtype=torch.complex128)
    providers = conjunction_providers(values, like)
    order = tuple(i for i, _, _ in iter_component_calls(record, "multi_and"))
    result = contract_component_tensors(
        record, order, providers.supply, like, "multi_and"
    )
    assert result.shape == (128, 16)
    print(
        f"MultiAnd: {len(order)} source-ordered And tensors, dense shape {tuple(result.shape)}"
    )


if __name__ == "__main__":
    main()
