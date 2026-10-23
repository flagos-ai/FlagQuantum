"""Measure statevector/MPS/tensor-network agreement over widths and depths.

The architectural invariant is that backend selection may change execution but
not the meaning of the program or its result contract. For the approximate
representations that is a quantitative claim: a truncated MPS discards weight
and a contracted network accumulates error in an order the dense path never
takes. One circuit cannot characterize it, so this sweep drives one
parameterized workload through every available representation over a grid and
records the spread per cell.

Two properties of the measurement matter more than the grid's size.

First, the spread is reported **relative** to the dense energy as well as
absolutely. The energy of this Hamiltonian grows with width, so an absolute
spread grows with width by normalization alone; a plot of it would show a trend
that is arithmetic rather than physics.

Second, each cell records the **observed** maximum bond dimension beside the
cap it was given. A cell whose observed bond never reaches its cap was not
truncated, so its spread measures floating-point noise and says nothing about
agreement under approximation. Reading such a cell as evidence of agreement is
the specific error this sweep exists to prevent: at the defaults of
``examples/vqe_switch_sv_mps_tn.py`` (depth 2, ``max_bond`` 16) the bond reaches
only 4, so the spread it reports is complex64 epsilon.

Examples
--------
Measure the checked-in grid and seal it. The payload is a single-device result,
so it belongs under ``benchmarks/results/local/``::

    python benchmarks/representation_agreement_sweep.py \
        --output benchmarks/results/local/representation_agreement_cpu_grid.json

A fast subgrid, which is what the conformance test drives::

    python benchmarks/representation_agreement_sweep.py \
        --widths 4,6 --depths 2,6 --structures nearest_neighbour \
        --output /tmp/agreement-smoke.json

Repeat the identical seeded angles with double-precision amplitudes::

    python benchmarks/representation_agreement_sweep.py --dtype complex128 \
        --output /tmp/agreement-complex128.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

import flagquantum as fq
import flagquantum.runtime as fqr
from flagquantum.simulation.tensor_network import tensor_network_expectations
from flagquantum.simulation.tensor_network.state import TensorNetworkState

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The representations the invariant is claimed over. ``statevector`` is first
#: because it is the exact reference the other two are compared against.
MODES = ("statevector", "mps", "tensor_network")

STRUCTURES = ("nearest_neighbour", "all_to_all")


def _integers(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def _modes(value: str) -> tuple[str, ...]:
    requested = tuple(item.strip() for item in value.split(","))
    unknown = [item for item in requested if item not in MODES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown representation(s) {unknown}; choose from {MODES}"
        )
    if "statevector" not in requested:
        raise argparse.ArgumentTypeError(
            "statevector is the exact reference the spread is measured against "
            "and cannot be dropped"
        )
    return requested


def _structures(value: str) -> tuple[str, ...]:
    requested = tuple(item.strip() for item in value.split(","))
    unknown = [item for item in requested if item not in STRUCTURES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown interaction structure(s) {unknown}; choose from {STRUCTURES}"
        )
    return requested


def build_ansatz(parameters: torch.Tensor, structure: str) -> fq.Circuit:
    """Build the workload at one width, depth, and interaction structure.

    The rotation layer is identical across structures so that only the
    entangling pattern differs: ``nearest_neighbour`` keeps the chain an MPS
    represents natively, while ``all_to_all`` couples every pair and is the
    regime where a chain representation and a contraction order would be
    expected to matter.
    """
    depth, n_qubits, _ = parameters.shape
    dtype = torch.complex128 if parameters.dtype == torch.float64 else torch.complex64
    circuit = fq.Circuit(n_qubits, device=parameters.device, dtype=dtype)
    for qubit in range(n_qubits):
        circuit.h(qubit)
    for layer in range(depth):
        for qubit in range(n_qubits):
            circuit.ry(qubit, parameters[layer, qubit, 0])
            circuit.rz(qubit, parameters[layer, qubit, 1])
        if structure == "nearest_neighbour":
            for qubit in range(n_qubits - 1):
                circuit.cx(qubit, qubit + 1)
        else:
            for control in range(n_qubits):
                for target in range(control + 1, n_qubits):
                    circuit.cx(control, target)
    return circuit


def exact_max_bond(n_qubits: int) -> int:
    """Return the bond dimension an exact open chain of this width needs.

    The widest cut of an ``n``-qubit chain splits it in half, so an untruncated
    MPS needs ``2 ** (n // 2)`` there. A cap at or above this value cannot
    truncate anything, however close the observed bond comes to it.
    """
    return 1 << (n_qubits // 2)


def _expectation(
    target: Any, n_qubits: int, *, x: tuple[int, ...] = (), z: tuple[int, ...] = ()
) -> torch.Tensor:
    if hasattr(target, "expectation_ps"):
        return target.expectation_ps(x=x, z=z).sum()
    state_circuit = fq.Circuit(
        n_qubits, device=target.device, dtype=target.dtype, inputs=target
    )
    return state_circuit.expectation_ps(x=x, z=z).sum()


def energy(target: Any, n_qubits: int) -> torch.Tensor:
    """Open-boundary transverse-field Ising energy.

    This is the observable ``examples/vqe_switch_sv_mps_tn.py`` trains, so a
    cell of this sweep measures the same quantity the single-point result
    reports.
    """
    if isinstance(target, TensorNetworkState) and target.max_intermediate_bytes is None:
        # The same bra/ket is shared by every Ising term. Contract the complete
        # observable batch once instead of rebuilding it for each Pauli product.
        observables: list[dict[str, tuple[int, ...]]] = [
            {"z": (qubit, qubit + 1)} for qubit in range(n_qubits - 1)
        ] + [{"x": (qubit,)} for qubit in range(n_qubits)]
        values = tensor_network_expectations(
            target.plan,
            observables,
            contraction_strategy=target.contraction_strategy,
        )
        coefficients = values.new_tensor([-1.0] * (n_qubits - 1) + [-0.7] * n_qubits)
        return (values * coefficients).sum()
    value = sum(
        -_expectation(target, n_qubits, z=(qubit, qubit + 1))
        for qubit in range(n_qubits - 1)
    )
    return value + sum(
        -0.7 * _expectation(target, n_qubits, x=(qubit,)) for qubit in range(n_qubits)
    )


def measure_cell(
    *,
    n_qubits: int,
    depth: int,
    structure: str,
    max_bond: int,
    seed: int,
    device: str,
    modes: tuple[str, ...] = MODES,
    dtype: torch.dtype = torch.complex64,
) -> dict[str, Any]:
    """Measure one grid cell: the same parameters through every representation.

    The parameters are drawn once per cell from a seeded generator and shared by
    every run, so a disagreement is a property of the representation rather than
    of a different starting point.
    """
    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError("dtype must be torch.complex64 or torch.complex128")
    generator = torch.Generator(device=device).manual_seed(seed)
    # Draw once in float32, including the scale, then widen exactly. Both
    # precision runs therefore execute identical angles rather than different
    # random streams or differently rounded input programs.
    parameters = (
        0.05
        * torch.randn(
            depth, n_qubits, 2, device=device, generator=generator, dtype=torch.float32
        )
    ).to(dtype=torch.float64 if dtype == torch.complex128 else torch.float32)
    failures: dict[str, str] = {}

    # Whether the cap actually discarded weight. An observed bond equal to the
    # cap does not settle it: a shallow circuit stops growing on its own, and at
    # depth 2 and width 8 the bond is 4 whether the cap is 4, 16, or 256. So the
    # same circuit runs once more with one doubling of headroom -- a state that
    # would have grown past the cap exceeds it here, and one that would not
    # reports the same bond. Probing at the exact bond would answer the same
    # question at exponential cost, since an exact chain of 20 qubits needs 1024.
    headroom_cap = min(exact_max_bond(n_qubits), 2 * max_bond)
    headroom_bond: int | None = None
    if "mps" in modes:
        try:
            with torch.no_grad():
                headroom_state = fqr.run_native(
                    build_ansatz(parameters, structure),
                    mode="mps",
                    max_bond=headroom_cap,
                )
                headroom_bond = int(max(headroom_state.bond_dims))
        except Exception as exc:
            headroom_bond = None
            failures["headroom_probe"] = (
                f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            )

    energies: dict[str, float | None] = {}
    observed_bond: int | None = None
    for mode in modes:
        options = {"max_bond": max_bond} if mode == "mps" else {}
        try:
            with torch.no_grad():
                target = fqr.run_native(
                    build_ansatz(parameters, structure), mode=mode, **options
                )
                if mode == "mps":
                    observed_bond = int(max(target.bond_dims))
                energies[mode] = float(energy(target, n_qubits))
        except Exception as exc:
            energies[mode] = None
            failures[mode] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"

    measured = [value for value in energies.values() if value is not None]
    reference = energies["statevector"]
    digits = 17 if dtype == torch.complex128 else 9
    # Preserve double precision in the record; nine digits suffice only for
    # complex64. Compute both spreads from the unrounded measurements.
    energies = {
        mode: None if value is None else float(f"{value:.{digits}g}")
        for mode, value in energies.items()
    }
    # The dense path is the exact reference the relative spread is divided by,
    # named explicitly rather than taken positionally.
    spread = max(measured) - min(measured) if len(measured) > 1 else None
    relative = (
        spread / abs(reference)
        if spread is not None and reference not in (None, 0.0)
        else None
    )
    spread = None if spread is None else float(f"{spread:.6g}")
    relative = None if relative is None else float(f"{relative:.6g}")
    cell: dict[str, Any] = {
        "n_qubits": n_qubits,
        "depth": depth,
        "structure": structure,
        "max_bond_cutoff": max_bond,
        "observed_max_bond": observed_bond,
        # The cap the headroom probe used, and the bond it reached. The cutoff
        # is `min(exact_max_bond(n_qubits), 2 * max_bond)` and the exact bond is
        # `2 ** (n_qubits // 2)`, so both are recomputable from the three fields
        # above; only the measured bond is recorded.
        "headroom_max_bond": headroom_bond,
        # The cap discarded weight only if the state grew past it once given
        # room. A cell where it did not was represented exactly at this cap, so
        # its spread is complex64 noise rather than evidence about agreement
        # under approximation.
        "mps_truncated": (
            None
            if observed_bond is None or headroom_bond is None
            else headroom_bond > max_bond
        ),
        "energies": energies,
        "spread_absolute": spread,
        "spread_relative": relative,
    }
    # A mode that raised is named with its error; a cell that measured every
    # requested mode carries no `failures` key at all. Recording an empty one in
    # every cell spends the benchmark-results ratchet budget on the absence of
    # news, and `summarize` reads the key's presence rather than its emptiness.
    if failures:
        cell["failures"] = failures
    return cell


def summarize(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Report the bound separately over truncated and untruncated cells.

    Mixing the two would let the untruncated cells, whose spread sits at
    complex64 epsilon, dominate a bound that is supposed to describe behavior
    under approximation.
    """
    truncated = [
        cell
        for cell in cells
        if cell["mps_truncated"] and cell["spread_relative"] is not None
    ]
    untruncated = [
        cell
        for cell in cells
        if cell["mps_truncated"] is False and cell["spread_relative"] is not None
    ]

    def bound(group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {"cells": 0}
        worst = max(group, key=lambda cell: cell["spread_relative"])
        return {
            "cells": len(group),
            # Already rounded on the cell; carried through unchanged so the
            # summary and the cell it names cannot disagree.
            "max_spread_relative": worst["spread_relative"],
            "max_spread_absolute": worst["spread_absolute"],
            "worst_cell": {
                key: worst[key]
                for key in ("n_qubits", "depth", "structure", "max_bond_cutoff")
            },
        }

    incomplete = [
        {key: cell[key] for key in ("n_qubits", "depth", "structure", "failures")}
        for cell in cells
        if "failures" in cell
    ]
    return {
        "cells_total": len(cells),
        "truncated": bound(truncated),
        "untruncated": bound(untruncated),
        "incomplete_cells": incomplete,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # The default grid is bounded by the tensor-network path, not by the
    # physics. That path re-contracts the whole network once per Pauli term and
    # an energy needs 2n-1 of them, so its cost grows with depth and, on the
    # all-to-all structure, with width squared: one term measured 0.6s, 3.3s and
    # 10.7s at depths 2, 4 and 6 on the chain at width 6, and a single term at
    # width 12 all-to-all depth 4 did not finish inside two minutes.
    # Statevector and MPS finish every one of these cells in under a second.
    #
    # So the chain is swept to width 20 while all-to-all stops at 8, and the
    # deeper and wider all-to-all cells are left to a run that can afford them
    # -- `--structures all_to_all --widths 12,16 --modes statevector,mps` is
    # affordable today and says nothing about the tensor-network path, which is
    # the honest trade rather than a silent omission.
    parser.add_argument("--widths", type=_integers, default=(4, 8, 12, 16, 20))
    parser.add_argument("--depths", type=_integers, default=(2, 4))
    parser.add_argument(
        "--all-to-all-max-width",
        type=int,
        default=8,
        help=(
            "widest all-to-all cell to attempt; its tensor-network contraction "
            "grows with width squared and is the sweep's cost ceiling"
        ),
    )
    parser.add_argument("--structures", type=_structures, default=STRUCTURES)
    parser.add_argument("--max-bond", type=_integers, default=(4, 16))
    parser.add_argument(
        "--modes",
        type=_modes,
        default=MODES,
        help=(
            "representations to compare; dropping tensor_network makes a wide "
            "grid affordable, at the cost of saying nothing about that path"
        ),
    )
    parser.add_argument("--seed", type=int, default=260720)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--dtype", choices=("complex64", "complex128"), default="complex64"
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    started = time.perf_counter()
    cells: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for structure in arguments.structures:
        for n_qubits in arguments.widths:
            if (
                structure == "all_to_all"
                and n_qubits > arguments.all_to_all_max_width
                and "tensor_network" in arguments.modes
            ):
                # Recorded rather than dropped: a reader has to be able to see
                # which part of the grid was not measured and why.
                skipped.append(
                    {
                        "structure": structure,
                        "n_qubits": n_qubits,
                        "reason": (
                            "all-to-all tensor-network contraction exceeds the "
                            "CPU budget above width "
                            f"{arguments.all_to_all_max_width}"
                        ),
                    }
                )
                continue
            for depth in arguments.depths:
                for max_bond in arguments.max_bond:
                    cell = measure_cell(
                        n_qubits=n_qubits,
                        depth=depth,
                        structure=structure,
                        max_bond=max_bond,
                        seed=arguments.seed,
                        device=arguments.device,
                        modes=arguments.modes,
                        dtype=getattr(torch, arguments.dtype),
                    )
                    cells.append(cell)
                    print(
                        json.dumps(
                            {
                                key: cell[key]
                                for key in (
                                    "structure",
                                    "n_qubits",
                                    "depth",
                                    "max_bond_cutoff",
                                    "observed_max_bond",
                                    "mps_truncated",
                                    "spread_relative",
                                )
                            }
                        ),
                        flush=True,
                    )

    payload = {
        "schema": "flagquantum.representation_agreement_sweep.v1",
        # The benchmark's identity, which the contract in
        # tests/benchmark_contract/test_benchmark_contract.py requires of every
        # unsealed payload beside its scalability blockers.
        "benchmark": "representation_agreement_sweep",
        "artifact_class": "measured_local_run",
        "claim": (
            "backend selection changes execution but not the result contract; "
            "this payload states the regime that was measured and the bound the "
            "spread held to inside it"
        ),
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        # The layout contract every checked-in non-release payload answers to,
        # enforced by tests/test_distributed_scalability_audit.py: a payload
        # under results/local must declare its evidence class and name why it
        # cannot carry a scalability claim, so a reader cannot mistake a
        # single-device measurement for a sharded one.
        "non_release_evidence": True,
        "benchmark_evidence_class": "local_non_release",
        "scalability_blockers": [
            "single-process CPU agreement measurement across representations; "
            "it shards no workload and is not distributed scalability evidence"
        ],
        "modes": list(arguments.modes),
        "grid": {
            "widths": list(arguments.widths),
            "depths": list(arguments.depths),
            "structures": list(arguments.structures),
            "max_bond_cutoffs": list(arguments.max_bond),
        },
        "observable": "open-boundary transverse-field Ising, h=0.7",
        "parameter_initialization": (
            "0.05 * float32 randn, one seeded draw shared per cell; "
            "complex128 widens the same float32 angles exactly"
        ),
        "seed": arguments.seed,
        "device": arguments.device,
        "dtype": arguments.dtype,
        "summary": summarize(cells),
        "skipped_configurations": skipped,
        "cells": cells,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "command": " ".join(sys.argv),
        "python": platform.python_version(),
        "torch": torch.__version__,
        # The published revision this measurement ran against. Recording HEAD
        # instead would name the working commit, which is rewritten by a rebase
        # or an amend before the payload is sealed; the evidence-revision check
        # requires a revision a ref of this repository reaches, and an orphaned
        # hash is exactly what it rejects.
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "origin/main"], cwd=REPO_ROOT, text=True, timeout=15
        ).strip(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, timeout=15
        ).strip(),
        # Whether the measuring code itself was committed when it ran. A whole
        # working tree cannot be reported clean here -- writing this payload
        # dirties the tree it would describe -- but the script that produced the
        # numbers can be, and that is the part a reader needs to re-run.
        "measuring_source_committed": not subprocess.check_output(
            ["git", "status", "--porcelain", "--", str(Path(__file__).resolve())],
            cwd=REPO_ROOT,
            text=True,
            timeout=15,
        ).strip(),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(arguments.output), **payload["summary"]}, indent=2))
    return 2 if payload["summary"]["incomplete_cells"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
