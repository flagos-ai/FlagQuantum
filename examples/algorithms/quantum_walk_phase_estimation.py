# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 phase_estimation/phase_estimation_of_quantum_walk.ipynb
# at bcb17bb25360ed68ce3bc084645db3e929799389.

"""Preserve the notebook's first-wire-controlled schedule and separate costs."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from examples.algorithms.thc_chemistry_resources import _flame_graph_model
from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation
from flagquantum.algorithms.primitives.alias_preparation import (
    source_dense_alias_preparation_cost,
)
from flagquantum.algorithms.primitives.hubbard_select import SelectHubbard
from flagquantum.algorithms.primitives.hubbard_walk import HubbardWalk
from flagquantum.algorithms.primitives.identity_prepare import IdentityPrepare
from flagquantum.algorithms.primitives.ising_walk import ising_walk
from flagquantum.algorithms.primitives.luis_perina_state import LuisPerinaState
from flagquantum.algorithms.primitives.qft import qft
from flagquantum.algorithms.primitives.qft_resource_model import (
    source_textbook_qft_cost,
)
from flagquantum.algorithms.primitives.qubitization_phase_estimation import (
    QubitizationPhaseEstimation,
)
from flagquantum.compiler._component_trace.call_graph import modeled_call_graph
from flagquantum.compiler._component_trace.gate_budget import legacy_gate_budget
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
    estimate_component_resources,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.lookup.bounded_pauli_select import (
    source_bounded_pauli_select_cost,
)
from flagquantum.compiler.registers.preparation.prepare_reflection import (
    source_reflection_declared_cost,
)
from flagquantum.compiler.registers.primitives.circuit_binding import (
    bind_primitive_circuit,
)
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.flame_graph import iter_flame_graph_data


def get_resource_state(m: int) -> NDArray[np.float64]:
    """Return the source ideal sine tensor for explicit input initialization."""
    if type(m) is not int or m < 0:
        raise ValueError("The resource-state width must be a nonnegative integer")
    denominator = 1 + 2**m
    return np.asarray(
        np.sqrt(2 / denominator) * np.sin(np.pi * (1 + np.arange(2**m)) / denominator),
        dtype=np.float64,
    )


def phase_estimation(
    operation: QubitizationPhaseEstimation,
    *,
    phase: tuple[int, ...],
    target: tuple[int, ...],
    workspace: tuple[int, ...] = (),
    classical_bits: tuple[int, ...] = (),
    adjoint: bool = False,
) -> _BoundBlockOperation:
    """Bind the original notebook schedule after explicit ideal-window injection.

    The first phase wire controls one walk. Later wires control reflection
    pairs around powers 2,4,... . QFT retains the original phase-word order.
    The caller initializes the ideal resource tensor separately. Its physical
    preparation is excluded, matching the original StatePreparationChannel.
    """

    def transform(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        reverse: bool,
    ) -> tuple[Instruction, ...]:
        if work or slots:
            raise ValueError("The notebook's direct QFT requires no work or slots")
        return bind_primitive_circuit(
            qft(len(phase), inverse=not reverse).to_ir(),
            tuple(reversed(ports["q"])),
        )

    bound = replace(
        operation,
        preparation=IdentityPrepare.from_bitsizes((len(phase),)).bind(),
        qft_inverse=transform,
        qft_inv_signature=Signature.build(q=len(phase)),
    )
    return bound.make_on(
        phase=tuple(reversed(phase)),
        target=target,
        walk_workspace=workspace,
        walk_classical_bits=classical_bits,
        adjoint=adjoint,
    )


def hubbard_walk_models(
    side: int, t: float, u: float
) -> tuple[ComponentCost, ComponentCost, ComponentCost]:
    """Return existing walk, positive-controlled walk and zero-reflection models."""
    walk = HubbardWalk.from_lattice(side, t, u)
    prepare = walk.prepare.source_resource_model()
    width = sum(bits for _, bits in walk.source_layout)
    selected = width - 2 * side**2
    reflection = source_reflection_declared_cost(
        selected, 2, prepare, prepare, control_value=0, global_phase=-1
    )
    positive = source_reflection_declared_cost(
        selected, 2, prepare, prepare, control_value=1, global_phase=-1
    )
    select = SelectHubbard(side, controlled=True).source_resource_model()
    controlled = ComponentCost(
        "ControlledWalk(1)",
        width + 1,
        calls=(CostCall(select), CostCall(positive, bystander_qubits=2 * side**2)),
    )
    return walk.source_resource_model(), controlled, reflection


def ising_walk_models(
    sites: int, epsilon: float
) -> tuple[ComponentCost, ComponentCost, ComponentCost]:
    """Reuse native alias preparation and specialized Pauli SELECT accounting."""
    walk, _ = ising_walk(sites, epsilon)
    prepare = source_dense_alias_preparation_cost(walk.table)
    selector = replace(walk.selector, control_val=1)
    selection = selector.selection_bitsize
    target = selector.target_bitsize
    select = source_bounded_pauli_select_cost(
        selector._instruction(
            tuple(range(1, selection + 1)),
            tuple(range(selection + 1, selection + target + 1)),
            (0,),
        )
    )
    controlled = walk.walk.resource_model(
        select, prepare, prepare, control_registers=(ControlRegister(1, (1,)),)
    )
    reflection = source_reflection_declared_cost(
        selection,
        walk.walk.prepare.junk_width,
        prepare,
        prepare,
        control_value=0,
        global_phase=-1,
    )
    return walk.resource_model(), controlled, reflection


def phase_estimation_model(
    models: tuple[ComponentCost, ComponentCost, ComponentCost],
    m: int,
    *,
    include_window_and_qft: bool = False,
) -> ComponentCost:
    """Preserve source circuit-cost omissions and the separate full-QPE model."""
    if type(m) is not int or m < 1:
        raise ValueError("Phase estimation requires positive integer phase width")
    walk, controlled, reflection = models
    prepare = ComponentCost("Resource state (injected, excluded)", m)
    transform = ComponentCost("Inverse QFT (excluded)", m)
    if include_window_and_qft:
        counts = LuisPerinaState(m).source_gate_cost().asdict()
        prepare = ComponentCost(
            "LuisPerinaState", m, GateCost(**{k: int(v) for k, v in counts.items()})
        )
        transform = source_textbook_qft_cost(m)
    calls = [
        CostCall(prepare, bystander_qubits=walk.qubits),
        CostCall(controlled, bystander_qubits=m - 1),
    ]
    for index in range(1, m):
        power = ComponentCost(
            f"Walk^{2**index}", walk.qubits, calls=(CostCall(walk, 2**index),)
        )
        reflection_call = CostCall(
            reflection, bystander_qubits=m - 1 + walk.calls[-1].bystander_qubits
        )
        calls.extend(
            (reflection_call, CostCall(power, bystander_qubits=m), reflection_call)
        )
    calls.append(CostCall(transform, bystander_qubits=walk.qubits))
    return ComponentCost(
        "QuantumWalkPhaseEstimation", m + walk.qubits, calls=tuple(calls)
    )


def legacy_flame_rows(model: ComponentCost) -> list[str]:
    """Fold the source historical T metric, retaining rotations separately."""
    root = _flame_graph_model(model)
    graph = modeled_call_graph(root)
    nodes = {root, *(node for edge in graph.direct_calls for node in edge)}
    totals = {
        node: int(legacy_gate_budget(estimate_component_resources(node).gates)["t"])
        for node in nodes
    }
    folded: Counter[str] = Counter()
    for row in iter_flame_graph_data(
        root, graph.direct_calls, totals, title=lambda node: node.name
    ):
        path, weight = row.rsplit("\t", 1)
        folded[path] += int(weight)
    return [f"{path}\t{weight}" for path, weight in folded.items()]
