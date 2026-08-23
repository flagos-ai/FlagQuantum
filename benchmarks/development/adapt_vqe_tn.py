"""Deterministic qubit-ADAPT-VQE workload for FlagQuantum TN evaluation.

This is a development benchmark, not scalability evidence.  It keeps the
algorithm, operator pool, Hamiltonian, and stopping rules independent of the
selected FlagQuantum simulation backend so a TensorCircuit-NG implementation
can consume the same JSON workload contract.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from math import prod
from pathlib import Path
from time import perf_counter

import torch

from adapt_vqe_tn_contract import initial_angles, product_state_energy

import flagquantum as fq
from flagquantum.simulation.tensor_contraction import (
    _canonicalize_unit_extent_nodes,
    _linearize_contraction_tree,
    _slice_nodes,
    _tree_from_steps,
)


@dataclass(frozen=True)
class PoolOperator:
    kind: str
    wires: tuple[int, ...]


def serialize_slicing_plan(plan: fq.TensorNetworkSlicingPlan) -> dict:
    return asdict(plan)


def deserialize_slicing_plan(payload: dict) -> fq.TensorNetworkSlicingPlan:
    data = dict(payload)
    tuple_fields = ("sliced_labels", "slice_shape")
    for field in tuple_fields:
        data[field] = tuple(data[field])
    data["contraction_path"] = tuple(
        fq.PairContractionStep(
            **{
                **step,
                "left_labels": tuple(step["left_labels"]),
                "right_labels": tuple(step["right_labels"]),
                "output_labels": tuple(step["output_labels"]),
                "output_shape": tuple(step["output_shape"]),
            }
        )
        for step in data["contraction_path"]
    )
    return fq.TensorNetworkSlicingPlan(**data)


def reslice_binary_axes(
    plan: fq.TensorNetworkExpectationPlan,
    slicing: fq.TensorNetworkSlicingPlan,
    *,
    target_slices: int,
) -> fq.TensorNetworkSlicingPlan:
    """Select dimension-two axes on a fixed external path."""

    if slicing.sliced_labels:
        raise ValueError("binary reslicing requires an unsliced external path")
    dims = {
        label: int(node.tensor.shape[index])
        for node in plan.nodes
        for index, label in enumerate(node.labels)
    }
    candidates = {
        label
        for node in plan.nodes
        for label in node.labels
        if dims[label] == 2 and label not in plan.output_labels
    }
    selected = []

    def economics(labels):
        label_set = set(labels)
        peak = max(
            (
                prod(dims[label] for label in step.output_labels if label not in label_set)
                for step in slicing.contraction_path
            ),
            default=1,
        )
        per_slice_cost = sum(
            max(
                1,
                step.estimated_cost
                // prod(
                    dims[label]
                    for label in label_set
                    if label in step.left_labels or label in step.right_labels
                ),
            )
            for step in slicing.contraction_path
        )
        return peak, per_slice_cost

    while 2 ** len(selected) < target_slices:
        choices = []
        for label in candidates - set(selected):
            peak, cost = economics((*selected, label))
            choices.append(((peak, cost, label), label))
        if not choices:
            raise ValueError("cannot create requested binary slices")
        selected.append(min(choices)[1])
    source_nodes = plan.nodes
    source_outputs = plan.output_labels
    if slicing.canonicalize_unit_extent_labels:
        source_nodes, source_outputs = _canonicalize_unit_extent_nodes(
            source_nodes, source_outputs
        )
    tree = _tree_from_steps(
        source_nodes, slicing.contraction_path
    )
    subnodes = _slice_nodes(source_nodes, {label: 0 for label in selected})
    contraction_path = _linearize_contraction_tree(
        tree, subnodes, source_outputs
    )
    peak = max((step.intermediate_size for step in contraction_path), default=1)
    per_slice_cost = sum(step.estimated_cost for step in contraction_path)
    n_slices = 2 ** len(selected)
    total_cost = per_slice_cost * n_slices
    return replace(
        slicing,
        sliced_labels=tuple(selected),
        slice_shape=(2,) * len(selected),
        n_slices=n_slices,
        per_slice_cost=per_slice_cost,
        total_estimated_cost=total_cost,
        peak_size=peak,
        recomputation_factor=(
            total_cost / slicing.baseline_estimated_cost
            if slicing.baseline_estimated_cost
            else 1.0
        ),
        peak_bytes=peak * slicing.element_size_bytes,
        contraction_path=contraction_path,
        contraction_path_source=f"{slicing.contraction_path_source}+binary_reslice",
    )


_PAULI_PRODUCT = {
    ("x", "x"): (1.0, "i"),
    ("x", "y"): (1.0j, "z"),
    ("x", "z"): (-1.0j, "y"),
    ("y", "x"): (-1.0j, "z"),
    ("y", "y"): (1.0, "i"),
    ("y", "z"): (1.0j, "x"),
    ("z", "x"): (1.0j, "y"),
    ("z", "y"): (-1.0j, "x"),
    ("z", "z"): (1.0, "i"),
}


def grid_edges(rows: int, cols: int) -> tuple[tuple[int, int], ...]:
    edges = []
    for row in range(rows):
        for col in range(cols):
            wire = row * cols + col
            if col + 1 < cols:
                edges.append((wire, wire + 1))
            if row + 1 < rows:
                edges.append((wire, wire + cols))
    return tuple(edges)


def operator_pool(rows: int, cols: int) -> tuple[PoolOperator, ...]:
    qubits = rows * cols
    edges = grid_edges(rows, cols)
    return tuple(
        [PoolOperator("ry", (wire,)) for wire in range(qubits)]
        + [PoolOperator("rxx", edge) for edge in edges]
        + [PoolOperator("rzz", edge) for edge in edges]
        + [PoolOperator("ryx", edge) for edge in edges]
        + [PoolOperator("rxy", edge) for edge in edges]
    )


def hamiltonian(rows: int, cols: int) -> fq.Hamiltonian:
    qubits = rows * cols
    terms = [fq.pauli_term(-1.0, "ZZ", edge) for edge in grid_edges(rows, cols)]
    terms.extend(fq.pauli_term(-0.7, "X", (wire,)) for wire in range(qubits))
    terms.extend(
        fq.pauli_term(0.11 * (-1.0 if wire % 2 else 1.0), "Z", (wire,))
        for wire in range(qubits)
    )
    return fq.Hamiltonian(terms)


def build_circuit(
    rows: int,
    cols: int,
    cycles: int,
    extra_entanglers: int,
    operators: tuple[object, ...],
    parameters: torch.Tensor,
    initial_state: str = "mean_field",
) -> fq.Circuit:
    qubits = rows * cols
    circuit = fq.Circuit(
        qubits,
        device=parameters.device,
        dtype=torch.complex128,
    )
    for wire, angle in enumerate(initial_angles(rows, cols, initial_state)):
        if initial_state == "legacy_h_ry":
            circuit.h(wire)
        if initial_state == "legacy_h_ry" and angle:
            circuit.ry(wire, theta=parameters.new_tensor(angle))
    edges = grid_edges(rows, cols)
    for cycle in range(cycles):
        for position, edge in enumerate(edges):
            if position % 2 == cycle % 2:
                circuit.cx(*edge)
    bridge_edges = tuple(
        edge for position, edge in enumerate(edges) if position % 2 == cycles % 2
    )
    if extra_entanglers > len(bridge_edges):
        raise ValueError(
            f"extra_entanglers={extra_entanglers} exceeds available "
            f"bridge edges={len(bridge_edges)}"
        )
    for edge in bridge_edges[:extra_entanglers]:
        circuit.cx(*edge)
    if initial_state != "legacy_h_ry":
        # The CNOT scaffold acts on |0...0> and is therefore an identity here;
        # prepare the audited product-state warm start after that scaffold.
        for wire, angle in enumerate(initial_angles(rows, cols, initial_state)):
            if angle:
                circuit.ry(wire, theta=parameters.new_tensor(angle))
    for operator, parameter in zip(operators, parameters):
        if not isinstance(operator, PoolOperator):
            raise TypeError("unexpected ADAPT-VQE pool operator")
        if operator.kind in {"ryx", "rxy"}:
            y_position = 0 if operator.kind == "ryx" else 1
            y_wire = operator.wires[y_position]
            circuit.sdg(y_wire)
            circuit.rxx(*operator.wires, theta=parameter)
            circuit.s(y_wire)
        else:
            getattr(circuit, operator.kind)(*operator.wires, theta=parameter)
    return circuit


def generator_ops(operator: PoolOperator) -> dict[int, str]:
    if operator.kind == "ryx":
        return {operator.wires[0]: "y", operator.wires[1]: "x"}
    if operator.kind == "rxy":
        return {operator.wires[0]: "x", operator.wires[1]: "y"}
    names = {"ry": "y", "rxx": "x", "rzz": "z"}
    return {wire: names[operator.kind] for wire in operator.wires}


def commutator_gradient_hamiltonian(
    operator: PoolOperator, target: fq.Hamiltonian
) -> fq.Hamiltonian | None:
    """Return the exact i/2 [P, H] gradient observable for exp(-i theta P/2)."""

    generator = generator_ops(operator)
    combined: dict[tuple[tuple[int, str], ...], complex] = {}
    for term in target.terms:
        target_ops = dict(term.ops)
        anticommutes = sum(
            generator.get(wire) is not None
            and target_ops.get(wire) is not None
            and generator[wire] != target_ops[wire]
            for wire in generator.keys() | target_ops.keys()
        )
        if anticommutes % 2 == 0:
            continue
        phase = 1.0 + 0.0j
        product: dict[int, str] = {}
        for wire in sorted(generator.keys() | target_ops.keys()):
            left = generator.get(wire, "i")
            right = target_ops.get(wire, "i")
            if left == "i":
                name = right
            elif right == "i":
                name = left
            else:
                local_phase, name = _PAULI_PRODUCT[(left, right)]
                phase *= local_phase
            if name != "i":
                product[wire] = name
        key = tuple(product.items())
        combined[key] = combined.get(key, 0.0j) + 1.0j * complex(term.coefficient) * phase
    terms = []
    for ops, coefficient in combined.items():
        if abs(coefficient) <= 1e-15:
            continue
        if abs(coefficient.imag) > 1e-12:
            raise ValueError("commutator gradient produced a non-Hermitian coefficient")
        terms.append(fq.pauli_term(coefficient.real, dict(ops)))
    return fq.Hamiltonian(terms) if terms else None


def tn_energy(
    circuit: fq.Circuit,
    target: fq.Hamiltonian,
    *,
    max_intermediate_bytes: int | None = None,
) -> torch.Tensor:
    plan = fq.build_tensor_network_hamiltonian_expectation(circuit, target)
    return plan.contract(
        strategy="auto_sliced" if max_intermediate_bytes else "greedy",
        max_intermediate_bytes=max_intermediate_bytes,
    ).real


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--extra-entanglers", type=int, default=0)
    parser.add_argument(
        "--initial-state",
        choices=("mean_field", "zero", "legacy_h_ry"),
        default="mean_field",
    )
    parser.add_argument("--adapt-iterations", type=int, default=2)
    parser.add_argument("--optimization-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-8)
    parser.add_argument("--pool-limit", type=int)
    parser.add_argument("--backend", choices=("statevector", "tn"), default="tn")
    parser.add_argument(
        "--screening", choices=("append", "commutator"), default="commutator"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-intermediate-gib", type=float)
    parser.add_argument("--planner", choices=("native", "cotengra"), default="native")
    parser.add_argument("--path-repeats", type=int, default=4)
    parser.add_argument("--path-method", default="greedy")
    parser.add_argument("--planner-seed", type=int, default=0)
    parser.add_argument("--target-slices", type=int)
    parser.add_argument("--binary-target-slices", type=int)
    parser.add_argument("--max-slices", type=int, default=4096)
    parser.add_argument("--max-recomputation-factor", type=float, default=64.0)
    parser.add_argument("--path-minimize", choices=("flops", "size", "write", "combo"), default="write")
    parser.add_argument("--planning-only", action="store_true")
    parser.add_argument("--screening-only", action="store_true")
    parser.add_argument("--plan-artifact-input", type=Path)
    parser.add_argument("--plan-artifact-output", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if (
        args.rows <= 0
        or args.cols <= 0
        or args.cycles < 0
        or args.extra_entanglers < 0
    ):
        raise ValueError("grid dimensions must be positive and cycles non-negative")
    device = torch.device(args.device)
    max_intermediate_bytes = (
        None
        if args.max_intermediate_gib is None
        else int(args.max_intermediate_gib * 2**30)
    )
    if max_intermediate_bytes is not None and max_intermediate_bytes <= 0:
        raise ValueError("max-intermediate-gib must be positive")
    target = hamiltonian(args.rows, args.cols)
    pool = operator_pool(args.rows, args.cols)
    if args.pool_limit is not None:
        if args.pool_limit <= 0:
            raise ValueError("pool-limit must be positive")
        if args.pool_limit < len(pool):
            indices = tuple(
                round(index * (len(pool) - 1) / max(1, args.pool_limit - 1))
                for index in range(args.pool_limit)
            )
            pool = tuple(pool[index] for index in indices)

    def builder(operators, parameters):
        return build_circuit(
            args.rows,
            args.cols,
            args.cycles,
            args.extra_entanglers,
            operators,
            parameters,
            args.initial_state,
        )

    planning_seconds = 0.0
    slicing_cache = {}
    topology_slicing_cache = {}
    slicing_summaries = {}
    artifact_plans = (
        json.loads(args.plan_artifact_input.read_text(encoding="utf-8"))["plans"]
        if args.plan_artifact_input is not None
        else {}
    )

    def contract_tn(plan, *, reuse_topology_path=False):
        nonlocal planning_seconds
        if max_intermediate_bytes is None:
            return plan.contract(strategy="greedy").real
        key = (
            tuple((node.labels, tuple(node.tensor.shape)) for node in plan.nodes),
            plan.output_labels,
            args.planner,
            max_intermediate_bytes,
        )
        artifact_signature = {
            "nodes": [
                {"labels": node.labels, "shape": tuple(node.tensor.shape)}
                for node in plan.nodes
            ],
            "output_labels": plan.output_labels,
        }
        topology_key = (
            tuple(node.labels for node in plan.nodes),
            plan.output_labels,
            args.planner,
            max_intermediate_bytes,
        )
        if reuse_topology_path and topology_key in topology_slicing_cache:
            slicing = topology_slicing_cache[topology_key]
            dims = {
                label: int(node.tensor.shape[index])
                for node in plan.nodes
                for index, label in enumerate(node.labels)
            }
            slice_shape = tuple(dims[label] for label in slicing.sliced_labels)
            adapted = replace(
                slicing,
                slice_shape=slice_shape,
                n_slices=prod(slice_shape) if slice_shape else 1,
            )
            return plan.contract_slicing_plan(adapted).real
        if key not in slicing_cache:
            planning_started = perf_counter()
            artifact_key = str(len(slicing_summaries))
            if artifact_key in artifact_plans:
                artifact = artifact_plans[artifact_key]
                if artifact["signature"] != json.loads(
                    json.dumps(artifact_signature)
                ):
                    raise ValueError("cotengra plan artifact topology mismatch")
                slicing = deserialize_slicing_plan(artifact["slicing"])
            elif args.planner == "cotengra":
                element_size = max(node.tensor.element_size() for node in plan.nodes)
                slicing = plan.cotengra_slicing_plan(
                    target_peak_elements=max_intermediate_bytes // element_size,
                    max_repeats=args.path_repeats,
                    minimize=args.path_minimize,
                    methods=(args.path_method,),
                    seed=args.planner_seed,
                    target_slices=args.target_slices,
                )
                if args.binary_target_slices is not None:
                    slicing = reslice_binary_axes(
                        plan,
                        slicing,
                        target_slices=args.binary_target_slices,
                    )
            else:
                slicing = plan.slicing_plan(
                    max_intermediate_bytes=max_intermediate_bytes
                )
            planning_seconds += perf_counter() - planning_started
            slicing.validate_economics(
                max_slices=args.max_slices,
                max_recomputation_factor=args.max_recomputation_factor,
            )
            slicing_cache[key] = slicing
            topology_slicing_cache[topology_key] = slicing
            slicing_summaries[str(len(slicing_summaries))] = slicing.summary()
        return plan.contract_slicing_plan(slicing_cache[key]).real

    energy_function = (
        (
            lambda circuit: contract_tn(
                fq.build_tensor_network_hamiltonian_expectation(circuit, target)
            )
        )
        if args.backend == "tn"
        else (lambda circuit: target.expectation(circuit))
    )

    def screening_function(operators, parameters, candidates):
        circuit = builder(operators, parameters)
        observables = tuple(
            commutator_gradient_hamiltonian(candidate, target)
            for candidate in candidates
        )
        if args.backend == "tn":
            if args.planner == "cotengra" and max_intermediate_bytes:
                values = [
                    contract_tn(
                        fq.build_tensor_network_hamiltonian_expectation(
                            circuit, observable
                        )
                    ).reshape(-1)[0]
                    for observable in observables
                    if observable is not None
                ]
            else:
                nonzero = tuple(
                    observable for observable in observables if observable
                )
                values = (
                    contract_tn(
                        fq.build_tensor_network_hamiltonian_expectations(
                            circuit, nonzero
                        )
                    ).reshape(-1)
                    if nonzero
                    else ()
                )
            value_iterator = iter(values)
            gradients = [
                float(next(value_iterator)) if observable is not None else 0.0
                for observable in observables
            ]
        else:
            gradients = [
                0.0
                if observable is None
                else float(observable.expectation(circuit).sum())
                for observable in observables
            ]
        return gradients

    if args.planning_only:
        if args.backend != "tn" or max_intermediate_bytes is None:
            raise ValueError(
                "planning-only requires TN backend and max-intermediate-gib"
            )
        parameters = torch.empty(0, dtype=torch.float64, device=device)
        circuit = builder((), parameters)
        energy_plan = fq.build_tensor_network_hamiltonian_expectation(circuit, target)
        contract_plans = [("energy", energy_plan)]
        if args.screening == "commutator":
            observables = tuple(
                filter(
                    None,
                    (
                        commutator_gradient_hamiltonian(candidate, target)
                        for candidate in pool
                    ),
                )
            )
            if args.planner == "cotengra":
                # Scalar commutator MPOs reuse the full-Hamiltonian path. Their
                # MPO bonds are smaller, so the energy plan is a conservative
                # peak-memory bound for screening.
                pass
            else:
                contract_plans.append(
                    (
                        "screening_block",
                        fq.build_tensor_network_hamiltonian_expectations(
                            circuit, observables
                        ),
                    )
                )
        preflight = {}
        serialized_plans = {}
        for name, plan in contract_plans:
            key = (
                tuple((node.labels, tuple(node.tensor.shape)) for node in plan.nodes),
                plan.output_labels,
                args.planner,
                max_intermediate_bytes,
            )
            planning_started = perf_counter()
            element_size = max(node.tensor.element_size() for node in plan.nodes)
            slicing = (
                plan.cotengra_slicing_plan(
                    target_peak_elements=max_intermediate_bytes // element_size,
                    max_repeats=args.path_repeats,
                    minimize=args.path_minimize,
                    methods=(args.path_method,),
                    seed=args.planner_seed,
                    target_slices=args.target_slices,
                )
                if args.planner == "cotengra"
                else plan.slicing_plan(max_intermediate_bytes=max_intermediate_bytes)
            )
            if args.binary_target_slices is not None:
                slicing = reslice_binary_axes(
                    plan,
                    slicing,
                    target_slices=args.binary_target_slices,
                )
            if (
                args.planner != "cotengra"
                and
                args.target_slices is not None
                and slicing.n_slices < args.target_slices
            ):
                slicing = plan.reslice_external_plan(
                    slicing, target_slices=args.target_slices
                )
            seconds = perf_counter() - planning_started
            slicing.validate_economics(
                max_slices=args.max_slices,
                max_recomputation_factor=args.max_recomputation_factor,
            )
            slicing_cache[key] = slicing
            preflight[name] = {**slicing.summary(), "planning_seconds": seconds}
            serialized_plans[str(len(serialized_plans))] = {
                "signature": {
                    "nodes": [
                        {
                            "labels": node.labels,
                            "shape": tuple(node.tensor.shape),
                        }
                        for node in plan.nodes
                    ],
                    "output_labels": plan.output_labels,
                },
                "slicing": serialize_slicing_plan(slicing),
            }
        payload = {
            "schema_version": 1,
            "workload": "qubit_adapt_vqe_grid_tfim_planning_preflight",
            "implementation": "FlagQuantum",
            "rows": args.rows,
            "cols": args.cols,
            "qubits": args.rows * args.cols,
            "cycles": args.cycles,
            "extra_entanglers": args.extra_entanglers,
            "initial_state": args.initial_state,
            "dtype": "complex128",
            "operator_pool_size": len(pool),
            "planner": args.planner,
            "path_method": args.path_method,
            "planner_seed": args.planner_seed,
            "target_peak_bytes": max_intermediate_bytes,
            "target_slices": args.target_slices,
            "binary_target_slices": args.binary_target_slices,
            "plans": preflight,
        }
        rendered = json.dumps(payload, indent=2, sort_keys=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        if args.plan_artifact_output is not None:
            args.plan_artifact_output.parent.mkdir(parents=True, exist_ok=True)
            args.plan_artifact_output.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "qubits": args.rows * args.cols,
                        "cycles": args.cycles,
                        "planner": args.planner,
                        "plans": serialized_plans,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        print(rendered)
        return

    if args.screening_only:
        parameters = torch.empty(0, dtype=torch.float64, device=device)
        circuit = builder((), parameters)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = perf_counter()
        initial_energy = float(energy_function(circuit).detach().sum())
        gradients = tuple(screening_function((), parameters, pool))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - started
        selected_index = max(
            range(len(gradients)), key=lambda index: abs(gradients[index])
        )
        payload = {
            "schema_version": 1,
            "workload": "qubit_adapt_vqe_grid_tfim_screening",
            "implementation": "FlagQuantum",
            "backend": "tensor_network",
            "rows": args.rows,
            "cols": args.cols,
            "qubits": args.rows * args.cols,
            "cycles": args.cycles,
            "extra_entanglers": args.extra_entanglers,
            "initial_state": args.initial_state,
            "dtype": "complex128",
            "parameter_dtype": "float64",
            "operator_pool_size": len(pool),
            "planner": args.planner,
            "screening_method": args.screening,
            "initial_energy": initial_energy,
            "pool_gradients": gradients,
            "selected_pool_index": selected_index,
            "selected_gradient": gradients[selected_index],
            "execution_seconds": elapsed,
            "path_planning_seconds": planning_seconds,
            "slicing_plans": slicing_summaries,
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "max_intermediate_bytes": max_intermediate_bytes,
            "scalability_claim_allowed": False,
        }
        rendered = json.dumps(payload, indent=2, sort_keys=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = perf_counter()
    result = fq.run_adapt_vqe(
        builder,
        pool,
        energy_function=energy_function,
        screening_function=(
            screening_function if args.screening == "commutator" else None
        ),
        max_adapt_iterations=args.adapt_iterations,
        optimization_steps=args.optimization_steps,
        lr=args.learning_rate,
        gradient_tolerance=args.gradient_tolerance,
        dtype=torch.float64,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    payload = {
        "schema_version": 1,
        "workload": "qubit_adapt_vqe_grid_tfim",
        "implementation": "FlagQuantum",
        "backend": "tensor_network" if args.backend == "tn" else "statevector",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "rows": args.rows,
        "cols": args.cols,
        "qubits": args.rows * args.cols,
        "cycles": args.cycles,
        "extra_entanglers": args.extra_entanglers,
        "initial_state": args.initial_state,
        "initial_product_state_energy": product_state_energy(
            args.rows,
            args.cols,
            initial_angles(args.rows, args.cols, args.initial_state),
        ),
        "dtype": "complex128",
        "parameter_dtype": "float64",
        "operator_pool": [asdict(operator) for operator in pool],
        "operator_pool_size": len(pool),
        "screening_method": args.screening,
        "hamiltonian_terms": target.n_terms,
        "adapt_iterations_requested": args.adapt_iterations,
        "adapt_iterations_completed": result.n_adapt_iterations,
        "optimization_steps_per_iteration": args.optimization_steps,
        "learning_rate": args.learning_rate,
        "selected_pool_indices": result.selected_pool_indices,
        "initial_energy": result.initial_energy,
        "final_energy": float(result.energy),
        "parameters": result.parameters.cpu().tolist(),
        "converged": result.converged,
        "execution_seconds": elapsed,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "max_intermediate_bytes": max_intermediate_bytes,
        "planner": args.planner,
        "path_method": args.path_method,
        "path_repeats": args.path_repeats,
        "path_minimize": args.path_minimize,
        "planner_seed": args.planner_seed,
        "path_planning_seconds": planning_seconds,
        "slicing_plans": slicing_summaries,
        "iterations": [asdict(record) for record in result.iterations],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
