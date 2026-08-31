"""TensorCircuit-NG termwise-TN baseline for the qubit-ADAPT-VQE workload."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import tensorcircuit as tc
import tensornetwork as tn
from tensorcircuit.experimental import DistributedContractor
from tensorcircuit.quantum import QuOperator

from adapt_vqe_tn_contract import initial_angles, product_state_energy


@dataclass(frozen=True)
class PoolOperator:
    kind: str
    wires: tuple[int, ...]


@dataclass(frozen=True)
class HamiltonianTerm:
    coefficient: float
    pauli: str
    wires: tuple[int, ...]


_PAULI_PRODUCT = {
    ("X", "X"): (1.0, "I"),
    ("X", "Y"): (1.0j, "Z"),
    ("X", "Z"): (-1.0j, "Y"),
    ("Y", "X"): (-1.0j, "Z"),
    ("Y", "Y"): (1.0, "I"),
    ("Y", "Z"): (1.0j, "X"),
    ("Z", "X"): (1.0j, "Y"),
    ("Z", "Y"): (-1.0j, "X"),
    ("Z", "Z"): (1.0, "I"),
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


def hamiltonian(rows: int, cols: int) -> tuple[HamiltonianTerm, ...]:
    qubits = rows * cols
    terms = [HamiltonianTerm(-1.0, "ZZ", edge) for edge in grid_edges(rows, cols)]
    terms.extend(HamiltonianTerm(-0.7, "X", (wire,)) for wire in range(qubits))
    terms.extend(
        HamiltonianTerm(0.11 * (-1.0 if wire % 2 else 1.0), "Z", (wire,))
        for wire in range(qubits)
    )
    return tuple(terms)


def commutator_gradient_terms(
    operator: PoolOperator, terms: tuple[HamiltonianTerm, ...]
) -> tuple[HamiltonianTerm, ...]:
    if operator.kind == "ryx":
        generator = {operator.wires[0]: "Y", operator.wires[1]: "X"}
    elif operator.kind == "rxy":
        generator = {operator.wires[0]: "X", operator.wires[1]: "Y"}
    else:
        generator_name = {"ry": "Y", "rxx": "X", "rzz": "Z"}[operator.kind]
        generator = {wire: generator_name for wire in operator.wires}
    combined: dict[tuple[tuple[int, str], ...], complex] = {}
    for term in terms:
        target = dict(zip(term.wires, term.pauli))
        anticommutes = sum(
            generator.get(wire) is not None
            and target.get(wire) is not None
            and generator[wire] != target[wire]
            for wire in generator.keys() | target.keys()
        )
        if anticommutes % 2 == 0:
            continue
        phase = 1.0 + 0.0j
        product = {}
        for wire in sorted(generator.keys() | target.keys()):
            left = generator.get(wire, "I")
            right = target.get(wire, "I")
            if left == "I":
                name = right
            elif right == "I":
                name = left
            else:
                local_phase, name = _PAULI_PRODUCT[(left, right)]
                phase *= local_phase
            if name != "I":
                product[wire] = name
        key = tuple(product.items())
        combined[key] = combined.get(key, 0.0j) + 1.0j * term.coefficient * phase
    result = []
    for ops, coefficient in combined.items():
        if abs(coefficient) <= 1e-15:
            continue
        if abs(coefficient.imag) > 1e-12:
            raise ValueError("commutator gradient produced a non-Hermitian coefficient")
        result.append(
            HamiltonianTerm(
                coefficient.real,
                "".join(name for _, name in ops),
                tuple(wire for wire, _ in ops),
            )
        )
    return tuple(result)


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
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--target-size", type=int, default=2**28)
    parser.add_argument("--path-repeats", type=int, default=4)
    parser.add_argument(
        "--mpo-compression", choices=("direct", "tt-svd"), default="direct"
    )
    parser.add_argument(
        "--screening", choices=("append", "commutator"), default="commutator"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def hamiltonian_mpo(
    terms: tuple[HamiltonianTerm, ...],
    qubits: int,
    *,
    compression: str = "direct",
) -> QuOperator:
    matrices = {
        "I": jnp.eye(2, dtype=jnp.complex128),
        "X": jnp.asarray([[0, 1], [1, 0]], dtype=jnp.complex128),
        "Y": jnp.asarray([[0, -1j], [1j, 0]], dtype=jnp.complex128),
        "Z": jnp.asarray([[1, 0], [0, -1]], dtype=jnp.complex128),
    }
    products = []
    for term in terms:
        local = [matrices["I"]] * qubits
        for name, wire in zip(term.pauli, term.wires):
            local[wire] = matrices[name]
        products.append(tuple(local))
    if qubits == 1:
        dense = sum(term.coefficient * product[0] for term, product in zip(terms, products))
        return QuOperator.from_tensor(dense, out_axes=(0,), in_axes=(1,))
    if compression == "tt-svd":
        term_count = len(terms)
        physical_dim = 4
        local_by_wire = tuple(
            np.stack(tuple(np.asarray(product[wire]) for product in products))
            for wire in range(qubits)
        )
        coefficients = np.asarray(
            [term.coefficient for term in terms], dtype=np.complex128
        )
        cores = [
            (coefficients[:, None, None] * local_by_wire[0]).reshape(
                1, term_count, physical_dim
            )
        ]
        diagonal = np.eye(term_count, dtype=np.complex128)
        for wire in range(1, qubits - 1):
            core = diagonal[:, :, None, None] * local_by_wire[wire][:, None, :, :]
            cores.append(core.reshape(term_count, term_count, physical_dim))
        cores.append(local_by_wire[-1].reshape(term_count, 1, physical_dim))
        for wire in range(qubits - 1):
            left_rank, right_rank, _ = cores[wire].shape
            matrix = cores[wire].transpose(0, 2, 1).reshape(
                left_rank * physical_dim, right_rank
            )
            left, singular, right = np.linalg.svd(matrix, full_matrices=False)
            rank = max(1, int(np.count_nonzero(singular > 1e-14 * singular[0])))
            transfer = singular[:rank, None] * right[:rank]
            cores[wire] = left[:, :rank].reshape(
                left_rank, physical_dim, rank
            ).transpose(0, 2, 1)
            next_right = cores[wire + 1].shape[1]
            cores[wire + 1] = (
                transfer @ cores[wire + 1].reshape(right_rank, -1)
            ).reshape(rank, next_right, physical_dim)
        tensors = [
            core.reshape(core.shape[0], core.shape[1], 2, 2)
            for core in cores
        ]
        nodes = [tn.Node(jnp.asarray(tensors[0].squeeze(0)), name="hamiltonian:0")]
        nodes.extend(
            tn.Node(jnp.asarray(tensors[wire]), name=f"hamiltonian:{wire}")
            for wire in range(1, qubits - 1)
        )
        nodes.append(
            tn.Node(
                jnp.asarray(tensors[-1].squeeze(1)),
                name=f"hamiltonian:{qubits - 1}",
            )
        )
        nodes[0][0] ^ nodes[1][0]
        for site in range(1, qubits - 1):
            nodes[site][1] ^ nodes[site + 1][0]
        out_edges = [nodes[0][1]]
        in_edges = [nodes[0][2]]
        for node in nodes[1:-1]:
            out_edges.append(node[2])
            in_edges.append(node[3])
        out_edges.append(nodes[-1][1])
        in_edges.append(nodes[-1][2])
        return QuOperator(out_edges, in_edges, ref_nodes=nodes)
    bond = len(terms)
    nodes = []
    first = jnp.stack(
        tuple(term.coefficient * product[0] for term, product in zip(terms, products))
    )
    nodes.append(tn.Node(first, name="hamiltonian:0"))
    for site in range(1, qubits - 1):
        core = jnp.zeros((bond, bond, 2, 2), dtype=jnp.complex128)
        core = core.at[jnp.arange(bond), jnp.arange(bond)].set(
            jnp.stack(tuple(product[site] for product in products))
        )
        nodes.append(tn.Node(core, name=f"hamiltonian:{site}"))
    last = jnp.stack(tuple(product[-1] for product in products))
    nodes.append(tn.Node(last, name=f"hamiltonian:{qubits - 1}"))
    nodes[0][0] ^ nodes[1][0]
    for site in range(1, qubits - 1):
        nodes[site][1] ^ nodes[site + 1][0]
    out_edges = [nodes[0][1]]
    in_edges = [nodes[0][2]]
    for node in nodes[1:-1]:
        out_edges.append(node[2])
        in_edges.append(node[3])
    out_edges.append(nodes[-1][1])
    in_edges.append(nodes[-1][2])
    return QuOperator(out_edges, in_edges, ref_nodes=nodes)


def main() -> None:
    args = arguments()
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    devices = jax.devices()[: args.devices]
    if len(devices) != args.devices:
        raise ValueError("requested more JAX devices than are available")
    qubits = args.rows * args.cols
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
    terms = hamiltonian(args.rows, args.cols)
    edges = grid_edges(args.rows, args.cols)
    selected_indices: list[int] = []
    parameters = jnp.empty((0,), dtype=jnp.float64)
    records = []
    contractor_build_seconds = 0.0
    contractor_count = 0
    contractor_cache = {}

    def contractors_for(
        selected: tuple[int, ...],
        values,
        observable_terms: tuple[HamiltonianTerm, ...] = terms,
    ):
        nonlocal contractor_build_seconds, contractor_count
        observable_key = tuple(
            (term.coefficient, term.pauli, term.wires) for term in observable_terms
        )
        cache_key = (selected, observable_key)
        if cache_key in contractor_cache:
            return contractor_cache[cache_key]
        operators = tuple(pool[index] for index in selected)
        mpo = hamiltonian_mpo(
            observable_terms, qubits, compression=args.mpo_compression
        )
        def nodes_fn(theta, selected_operators=operators):
            circuit = tc.Circuit(qubits)
            for wire, angle in enumerate(
                initial_angles(args.rows, args.cols, args.initial_state)
            ):
                if args.initial_state == "legacy_h_ry":
                    circuit.h(wire)
                if args.initial_state == "legacy_h_ry" and angle:
                    circuit.ry(wire, theta=jnp.asarray(angle, dtype=jnp.float64))
            for cycle in range(args.cycles):
                for position, edge in enumerate(edges):
                    if position % 2 == cycle % 2:
                        circuit.cnot(*edge)
            extra_parity = args.cycles % 2
            extra_edges = tuple(
                edge
                for position, edge in enumerate(edges)
                if position % 2 == extra_parity
            )[: args.extra_entanglers]
            for edge in extra_edges:
                circuit.cnot(*edge)
            if args.initial_state != "legacy_h_ry":
                for wire, angle in enumerate(
                    initial_angles(args.rows, args.cols, args.initial_state)
                ):
                    if angle:
                        circuit.ry(wire, theta=jnp.asarray(angle, dtype=jnp.float64))
            for operator, parameter in zip(selected_operators, theta):
                if operator.kind in {"ryx", "rxy"}:
                    y_position = 0 if operator.kind == "ryx" else 1
                    y_wire = operator.wires[y_position]
                    circuit.sdg(y_wire)
                    circuit.rxx(*operator.wires, theta=parameter)
                    circuit.s(y_wire)
                else:
                    getattr(circuit, operator.kind)(*operator.wires, theta=parameter)
            state = circuit.get_quvector()
            return (state.adjoint() @ mpo @ state).nodes

        started = perf_counter()
        result = DistributedContractor(
            nodes_fn,
            values,
            devices=devices,
            cotengra_options={
                "slicing_reconf_opts": {"target_size": args.target_size},
                "max_repeats": args.path_repeats,
                "minimize": "write",
                "parallel": False,
                "progbar": False,
            },
        )
        contractor_build_seconds += perf_counter() - started
        contractor_count += 1
        contractor_cache[cache_key] = result
        return result

    def value_and_grad(
        selected: tuple[int, ...],
        values,
        observable_terms: tuple[HamiltonianTerm, ...] = terms,
    ):
        value = jnp.asarray(0.0, dtype=jnp.float64)
        gradient = jnp.zeros_like(values)
        contractor = contractors_for(selected, values, observable_terms)
        term_value, term_gradient = contractor.value_and_grad(values)
        value = value + jnp.real(term_value)
        gradient = gradient + jnp.real(term_gradient)
        value.block_until_ready()
        gradient.block_until_ready()
        return value, gradient

    started = perf_counter()
    initial_energy, _ = value_and_grad((), parameters)
    for _ in range(args.adapt_iterations):
        screening_started = perf_counter()
        available = [index for index in range(len(pool)) if index not in selected_indices]
        gradients = []
        for pool_index in available:
            if args.screening == "append":
                candidate = jnp.concatenate(
                    (parameters, jnp.zeros((1,), dtype=jnp.float64))
                )
                _, candidate_gradient = value_and_grad(
                    (*selected_indices, pool_index), candidate
                )
                gradients.append(float(candidate_gradient[-1]))
            else:
                gradient_terms = commutator_gradient_terms(pool[pool_index], terms)
                if not gradient_terms:
                    gradients.append(0.0)
                else:
                    gradient, _ = value_and_grad(
                        tuple(selected_indices), parameters, gradient_terms
                    )
                    gradients.append(float(gradient))
        best_position = max(range(len(available)), key=lambda index: abs(gradients[index]))
        best_index = available[best_position]
        best_gradient = gradients[best_position]
        screening_seconds = perf_counter() - screening_started
        if abs(best_gradient) <= args.gradient_tolerance:
            break
        energy_before, _ = value_and_grad(tuple(selected_indices), parameters)
        selected_indices.append(best_index)
        parameters = jnp.concatenate((parameters, jnp.zeros((1,), dtype=jnp.float64)))
        first_moment = jnp.zeros_like(parameters)
        second_moment = jnp.zeros_like(parameters)
        history = []
        optimization_started = perf_counter()
        for step in range(1, args.optimization_steps + 1):
            value, gradient = value_and_grad(tuple(selected_indices), parameters)
            history.append(float(value))
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * jnp.square(gradient)
            corrected_first = first_moment / (1.0 - 0.9**step)
            corrected_second = second_moment / (1.0 - 0.999**step)
            parameters = parameters - args.learning_rate * corrected_first / (
                jnp.sqrt(corrected_second) + 1e-8
            )
        optimization_seconds = perf_counter() - optimization_started
        energy_after, _ = value_and_grad(tuple(selected_indices), parameters)
        full_gradients = [0.0] * len(pool)
        for pool_index, gradient in zip(available, gradients):
            full_gradients[pool_index] = gradient
        records.append(
            {
                "selected_pool_index": best_index,
                "selected_gradient": best_gradient,
                "pool_gradients": full_gradients,
                "energy_before": float(energy_before),
                "energy_after": float(energy_after),
                "optimization_history": history,
                "screening_seconds": screening_seconds,
                "optimization_seconds": optimization_seconds,
            }
        )
    final_energy, _ = value_and_grad(tuple(selected_indices), parameters)
    execution_seconds = perf_counter() - started
    payload = {
        "schema_version": 1,
        "workload": "qubit_adapt_vqe_grid_tfim",
        "implementation": "TensorCircuit-NG 1.8.0 DistributedContractor",
        "backend": "tensor_network",
        "distribution_semantics": (
            "single_device_fast_path" if args.devices == 1 else "manual_sliced_tensor_contraction"
        ),
        "scalability_claim_allowed": False,
        "rows": args.rows,
        "cols": args.cols,
        "qubits": qubits,
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
        "devices": args.devices,
        "operator_pool": [asdict(operator) for operator in pool],
        "operator_pool_size": len(pool),
        "screening_method": args.screening,
        "mpo_compression": args.mpo_compression,
        "planner": "cotengra",
        "path_repeats": args.path_repeats,
        "path_minimize": "write",
        "target_peak_elements": args.target_size,
        "hamiltonian_terms": len(terms),
        "adapt_iterations_requested": args.adapt_iterations,
        "adapt_iterations_completed": len(records),
        "optimization_steps_per_iteration": args.optimization_steps,
        "learning_rate": args.learning_rate,
        "selected_pool_indices": selected_indices,
        "initial_energy": float(initial_energy),
        "final_energy": float(final_energy),
        "parameters": list(map(float, parameters)),
        "execution_seconds": execution_seconds,
        "contractor_build_seconds": contractor_build_seconds,
        "contractor_count": contractor_count,
        "iterations": records,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
