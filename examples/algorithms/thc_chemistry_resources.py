# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 THC tutorial and notebook helpers.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/thc

"""Source fixture, call, cost, and register views for the native THC tutorial."""

from collections import Counter
from dataclasses import replace
from html import escape

import numpy as np
from numpy.typing import NDArray

from flagquantum.algorithms.primitives.thc_chemistry_prepare import PrepareTHCChemistry
from flagquantum.algorithms.primitives.thc_uniform_preparation import (
    THCUniformPreparation,
)
from flagquantum.compiler._component_trace.call_graph import modeled_call_graph
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    PhaseCall,
    ResourceCall,
    arbitrary_clifford_cost_key,
    generalize_control_values,
    generalize_rotation,
    ignore_allocation,
    ignore_register_partition,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
    estimate_component_resources,
)
from flagquantum.drawer.flame_graph import iter_flame_graph_data
from flagquantum.drawer.register_score import (
    RegisterScore,
    RegisterStep,
    layout_register_score,
)


def generalize_thc_resource_call(call: ResourceCall) -> ResourceCall | None:
    """Apply the tutorial's view rules to typed native graph nodes."""
    for rule in (
        ignore_register_partition,
        ignore_allocation,
        generalize_rotation,
        generalize_control_values,
    ):
        generalized = rule(call)
        if generalized is None:
            return None
        call = generalized
    if isinstance(call, CliffordCall) and call.width in (1, 2):
        return arbitrary_clifford_cost_key(call.width)
    if isinstance(call, PhaseCall) and call.opcode == "s":
        return arbitrary_clifford_cost_key(1)
    return call


def lookup_graph_label(
    selection_widths: tuple[int, ...],
    target_widths: tuple[int, ...],
    *,
    block_sizes: tuple[int, ...] | None = None,
    clean: bool = False,
) -> str:
    """Set lookup annotations on the current graph without changing its nodes."""
    if block_sizes is None:
        return f"QROM(selection_bitsizes={selection_widths}, target_bitsizes={target_widths})"
    kind = "QROAMClean" if clean else "SelectSwapQROM"
    return f"{kind}(target_bitsizes={target_widths}, block_sizes={block_sizes})"


def thc_tutorial_integrals() -> (
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
):
    """Reproduce the seeded eight-factor, four-spatial-orbital source fixture."""
    rng = np.random.RandomState(7)
    one = rng.normal(size=(4, 4))
    central = rng.normal(size=(8, 8))
    leaves = rng.normal(size=(8, 4))
    return np.linalg.eigvalsh(0.5 * (one + one.T)), leaves, 0.5 * (central + central.T)


def uniform_t_breakdown(model: ComponentCost) -> dict[str, int]:
    """Use the same stopping categories as the source classifier."""
    groups: Counter[str] = Counter(
        rotations=11 * model.gates.rotation, toffoli=4 * model.gates.toffoli
    )
    for call in model.calls:
        child = call.component
        family = {
            "ReflectionUsingPrepare": "reflection",
            "MultiControlX": "multi_control_pauli",
        }.get(child.name, "arithmetic")
        groups[family] += (
            call.repetitions * estimate_component_resources(child).gates.total_t_count()
        )
    return dict(groups)


def tutorial_call_model(model: ComponentCost, *, width: int) -> ComponentCost:
    """Show the named immediate source operations that the model groups locally."""
    gates: tuple[tuple[str, int, GateCost, int], ...]
    if model.name == "UniformSuperpositionTHC":
        gates = (
            ("OnEach(H)", width, GateCost(clifford=width), 6),
            ("Ry (both directions)", 1, GateCost(rotation=1), 2),
            ("Toffoli", 3, GateCost(toffoli=1), 4),
            ("XGate", 1, GateCost(clifford=1), 1),
        )
    elif model.name == "PrepareTHC":
        gates = (
            ("OnEach(H)", width, GateCost(clifford=width), 1),
            ("XGate", 1, GateCost(clifford=1), 2),
            ("CZ", 2, GateCost(clifford=1), 2),
            ("Hadamard", 1, GateCost(clifford=1), 3),
        )
    elif model.name == "SelectTHC":
        gates = (("XGate", 1, GateCost(clifford=1), 1),)
    else:
        raise ValueError(f"Unknown THC tutorial root {model.name}")
    return replace(
        model,
        gates=GateCost(),
        calls=(
            *model.calls,
            *(
                CostCall(ComponentCost(name, size, cost), n)
                for name, size, cost, n in gates
            ),
        ),
    )


def _flame_graph_model(model: ComponentCost) -> ComponentCost:
    """Expose exclusive cost alongside child calls so the folded weights close."""
    if not model.calls:
        return model
    calls = tuple(
        replace(call, component=_flame_graph_model(call.component))
        for call in model.calls
    )
    if model.gates.total_t_count():
        calls += (
            CostCall(ComponentCost(f"{model.name} local", model.qubits, model.gates)),
        )
    return replace(model, gates=GateCost(), calls=calls)


def folded_thc_costs(model: ComponentCost) -> Counter[tuple[str, ...]]:
    """Return folded path masses from the native resource graph."""
    root = _flame_graph_model(model)
    graph = modeled_call_graph(root)
    nodes = {root, *(node for edge in graph.direct_calls for node in edge)}
    budgets = {
        node: estimate_component_resources(node).gates.total_t_count() for node in nodes
    }
    folded: Counter[tuple[str, ...]] = Counter()
    for row in iter_flame_graph_data(
        root, graph.direct_calls, budgets, title=lambda node: node.name
    ):
        labels, weight = row.rsplit("\t", 1)
        folded[tuple(labels.split(";"))] += int(weight)
    return folded


def thc_flame_svg(model: ComponentCost) -> str:
    """Render the folded stacks as a self-contained, weighted flame graph."""
    folded = folded_thc_costs(model)
    total = sum(folded.values())
    if not total:
        raise ValueError("The THC flame graph requires positive T-equivalent costs")
    scale = 960 / total
    row_height = 24
    depth = max(map(len, folded))
    height = depth * row_height + 8
    spans: dict[tuple[str, ...], tuple[int, int]] = {}
    start = 0
    for path, weight in sorted(folded.items()):
        for length in range(1, len(path) + 1):
            prefix = path[:length]
            left, _ = spans.get(prefix, (start, start))
            spans[prefix] = left, start + weight
        start += weight
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" role="img" aria-label="T-equivalent call stacks">'
    ]
    for path, (left, right) in spans.items():
        x, width = left * scale, (right - left) * scale
        y = (depth - len(path)) * row_height
        label = path[-1]
        room = max(0, int((width - 9) / 7))
        fill = ("#f7cd82", "#e99b49", "#f2b46a")[len(path) % 3]
        text = escape(label if len(label) <= room else label[: max(0, room - 1)] + "…")
        parts.append(
            f"<g><title>{escape(label)} · {right - left} T equivalents</title>"
            f'<rect x="{x:.3f}" y="{y}" width="{width:.3f}" height="22" fill="{fill}" stroke="white"/>'
        )
        if room >= 3:
            parts.append(
                f'<text x="{x + 4:.3f}" y="{y + 15}" font-family="sans-serif" font-size="12">{text}</text>'
            )
        parts.append("</g>")
    return "".join((*parts, "</svg>"))


def _step(name: str, *ports: str) -> RegisterStep:
    return RegisterStep(name, ports, ports)


def uniform_register_score(uniform: THCUniformPreparation) -> RegisterScore:
    """Lay out the frozen uniform component and four temporary flag lifetimes."""
    widths = dict(uniform.layout)
    inputs = tuple(widths)
    widths.update(lt=1, le=1, gt=1, mark=1)
    checks = (
        _step("ν < M", "nu", "lt"),
        _step("μ ≤ ν", "mu", "nu", "le"),
        _step("ν = M+1", "nu", "nu_eq_mp1"),
        _step("μ > N/2", "mu", "gt"),
    )
    toffoli = _step("CCX", "nu_eq_mp1", "gt", "mark")
    hadamards = (_step("H", "mu"), _step("H", "nu"))
    return layout_register_score(
        widths,
        inputs,
        (
            RegisterStep("Allocate", outputs=("lt", "le", "gt", "mark")),
            *hadamards,
            _step("Ry", "rot"),
            *checks,
            toffoli,
            _step("Reflect", "rot", "lt", "le", "mark"),
            *checks,
            toffoli,
            _step("Ry†", "rot"),
            *hadamards,
            _step("Reflect", "mu", "nu", "rot"),
            *hadamards,
            *checks,
            toffoli,
            _step("Success", "lt", "le", "mark", "succ"),
            toffoli,
            checks[0],
            checks[1],
            checks[3],
            _step("X", "mark"),
            RegisterStep("Free", ("lt", "le", "gt", "mark")),
        ),
    )


def prepare_register_score(prepare: PrepareTHCChemistry) -> RegisterScore:
    """Show the source alias bindings and retained grouped lookup outputs."""
    widths = dict((*prepare.selection_layout, *prepare.junk_layout))
    inputs = tuple(reg.name for reg in prepare.source_signature.lefts())
    banks = tuple(name for name, _ in prepare.data.lookup_output_layout)
    compare = _step("Compare", "keep", "sigma", "less_than")
    return layout_register_score(
        widths,
        inputs,
        (
            _step("Uniform", "mu", "nu", "succ", "nu_eq_mp1", "rot"),
            _step("Index", "mu", "nu", "s"),
            RegisterStep("QROAM", ("s",), ("s", *banks)),
            _step("H", "sigma"),
            compare,
            _step("CZ", "alt_theta", "less_than"),
            _step("X", "less_than"),
            _step("CZ", "theta", "less_than"),
            _step("X", "less_than"),
            _step("CSwap", "less_than", "alt_mu", "mu"),
            _step("CSwap", "less_than", "alt_nu", "nu"),
            compare,
            *(_step("H", name) for name in ("plus_a", "plus_b", "plus_mn")),
            _step("Predicate", "nu_eq_mp1", "plus_a", "extra_ctrl"),
            _step("CSwap", "extra_ctrl", "mu", "nu"),
        ),
    )
