# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Workflow adapted from Qualtran v0.7.0 simulation/tensor.ipynb.

"""Select logical And instances and inspect the existing native contraction plan."""

import torch

from examples.algorithms.conjunction_tensors import (
    conjunction_program,
    conjunction_providers,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.component_tensor_nodes import (
    component_tensor_nodes,
)
from flagquantum.simulation.tensor_network.contraction import _contraction_profile
from flagquantum.simulation.tensor_network.models import TensorNetworkContractionProfile
from flagquantum.simulation.tensor_network.register_axis_groups import (
    register_tensor_label_groups,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    contract_register_tensors,
)


def select_conjunction_instances(
    values: tuple[int, ...],
    like: torch.Tensor,
    *,
    keep_instance: int = 2,
) -> tuple[ProgramRecord, tuple[int, ...]]:
    """Wrap MultiAnd and retain the chosen occurrence plus declared atomic leaves.

    The original notebook's ``i != 2`` retains an And, which is already atomic
    for tensor contraction. Native predicate bodies have physical decompositions;
    explicit providers identify the same logical stopping boundary. Occurrence
    IDs come from the existing expansion owner, not positions or parsed names.
    Keeping root zero intentionally leaves an unsupplied MultiAnd and its normal
    missing-provider failure. No execution graph, numerical planner or foreign
    object is introduced into Compiler.
    """
    record = conjunction_program(values)
    providers = conjunction_providers(values, like)
    builder = ComponentProgramBuilder(
        Signature(record.functions[0].registers or ()),
        record.functions,
        name="tensor_root",
    )
    wrapped = builder.finalize(builder.call_dict("multi_and", builder.inputs))
    return expand_component_calls(
        wrapped,
        "tensor_root",
        predicate=lambda instance, operation: instance != keep_instance
        and not providers.has_custom_tensor(operation.callee or ""),
    )


def contract_conjunction_with_info(
    values: tuple[int, ...],
    like: torch.Tensor,
    *,
    keep_instance: int = 2,
) -> tuple[torch.Tensor, TensorNetworkContractionProfile]:
    """Return dense isometry and the authoritative native shape-only profile.

    Reuse existing graph wiring, output/input grouping, greedy contraction and
    its profile. Cost is the native product-of-union-dimensions estimate, not
    opt_einsum FLOPs, physical gates, process RAM or a performance measurement.
    Peak counts intermediate tensor elements including dense output. Planning
    does not contract data; returned data retains dtype/device and autograd.
    """
    selected, _ = select_conjunction_instances(
        values, like, keep_instance=keep_instance
    )
    providers = conjunction_providers(values, like)
    order = tuple(i for i, _, _ in iter_component_calls(selected, "tensor_root"))
    nodes, incoming, outgoing = component_tensor_nodes(
        selected, order, providers.supply, like, "tensor_root"
    )
    signature = Signature(selected.functions[0].registers or ())
    groups = register_tensor_label_groups(signature, incoming, outgoing)
    outputs = tuple(label for group in groups for label in group)
    profile = _contraction_profile(nodes, outputs, strategy="greedy")
    dense = contract_register_tensors(nodes, signature, incoming, outgoing)
    return dense, profile


def main() -> None:
    """Inspect the original three-And network with instance two retained."""
    values = (1, 1, 1, 1)
    like = torch.empty((), dtype=torch.complex128)
    matrix, profile = contract_conjunction_with_info(values, like)
    _, instances = select_conjunction_instances(values, like)
    assert instances == (1, 2, 3)
    print(f"MultiAnd instances: {instances}; dense shape {tuple(matrix.shape)}")
    print(
        f"Native estimate: cost={profile.estimated_cost}, peak={profile.peak_size} elements, steps={profile.n_steps}"
    )


if __name__ == "__main__":
    main()
