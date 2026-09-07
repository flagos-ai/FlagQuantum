"""Independent MPS/DMRG reference for the 48q grid-TFIM benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import quimb.tensor as qtn
import torch
from adapt_vqe_tn_contract import grid_edges, mean_field_angles

from flagquantum.simulation.tensor_network.entrypoints import _compress_pauli_sum_mpo


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--bond-dims", type=int, nargs="+", default=(32, 64, 128, 256))
    parser.add_argument("--cutoff", type=float, default=1e-12)
    parser.add_argument("--max-sweeps", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--initial-bond", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def exact_mpo(rows: int, cols: int) -> qtn.MatrixProductOperator:
    qubits = rows * cols
    identity = torch.eye(2, dtype=torch.complex128)
    paulis = {
        "X": torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
        "Z": torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128),
    }
    terms = [(-1.0, {left: "Z", right: "Z"}) for left, right in grid_edges(rows, cols)]
    terms += [(-0.7, {wire: "X"}) for wire in range(qubits)]
    terms += [
        (0.11 * (-1.0 if wire % 2 else 1.0), {wire: "Z"})
        for wire in range(qubits)
    ]
    products = tuple(
        tuple(paulis.get(ops.get(wire), identity) for wire in range(qubits))
        for _, ops in terms
    )
    coefficients = torch.tensor([coefficient for coefficient, _ in terms], dtype=torch.complex128)
    cores = _compress_pauli_sum_mpo(coefficients, products)
    arrays = [cores[0].squeeze(0).numpy()]
    arrays.extend(core.numpy() for core in cores[1:-1])
    arrays.append(cores[-1].squeeze(1).numpy())
    return qtn.MatrixProductOperator(arrays, shape="lrud")


def main() -> None:
    args = arguments()
    started = perf_counter()
    mpo = exact_mpo(args.rows, args.cols)
    if args.initial_bond == 1:
        angles = mean_field_angles(args.rows, args.cols)
        product_arrays = [
            np.asarray([np.cos(angle / 2), np.sin(angle / 2)], dtype=np.complex128)
            for angle in angles
        ]
        initial = qtn.MPS_product_state(product_arrays)
        initial_state = "mean_field_product"
    else:
        initial = qtn.MPS_rand_state(
            args.rows * args.cols,
            bond_dim=args.initial_bond,
            dtype="complex128",
            seed=args.seed,
        )
        initial_state = "seeded_random_mps"
    dmrg = qtn.DMRG2(
        mpo,
        bond_dims=args.bond_dims,
        cutoffs=args.cutoff,
        p0=initial,
    )
    converged = dmrg.solve(
        tol=args.tolerance,
        max_sweeps=args.max_sweeps,
        verbosity=1,
    )
    applied = mpo.apply(dmrg.state)
    energy_squared = float(np.real(applied.H @ applied))
    energy = float(np.real(dmrg.energy))
    variance = max(0.0, energy_squared - energy * energy)
    payload = {
        "schema_version": 1,
        "workload": "grid_tfim_dmrg_reference",
        "implementation": "quimb.DMRG2",
        "rows": args.rows,
        "cols": args.cols,
        "qubits": args.rows * args.cols,
        "dtype": "complex128",
        "bond_dims": args.bond_dims,
        "cutoff": args.cutoff,
        "tolerance": args.tolerance,
        "max_sweeps": args.max_sweeps,
        "initial_state": initial_state,
        "initial_bond": args.initial_bond,
        "seed": args.seed,
        "converged": bool(converged),
        "energy": energy,
        "energy_variance": variance,
        "sweep_energies": [float(np.real(value)) for value in dmrg.energies],
        "max_bond_reached": int(dmrg.state.max_bond()),
        "execution_seconds": perf_counter() - started,
        "reference_status": "variational_upper_bound_not_exact_theory",
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
