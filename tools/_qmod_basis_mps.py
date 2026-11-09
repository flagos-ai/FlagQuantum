"""Bounded computational-basis MPS profile for pinned Qmod acceptance."""

from __future__ import annotations

import math
import warnings

import torch

from flagquantum.compiler._construction.selection import select_register_program
from flagquantum.compiler.resource_estimation import estimate_resources
from flagquantum.ecosystem._qmod.adapter import QmodImport
from flagquantum.simulation.mps.entrypoints import run_mps

_BASIS_GATE_WIDTH = {"i": 1, "x": 1, "z": 1, "cx": 2, "ccx": 3}


def probe_basis_mps(imported: QmodImport, memory_bytes: int) -> dict[str, str | int]:
    """Verify and execute an all-zero-entry basis permutation with an exact sign and no truncation."""
    if imported.workflow is not None:
        return {"stage": "planning_refused", "detail": "expected a static Qmod program"}
    selection = select_register_program(
        imported.program,
        imported.functions,
        include_boolean=True,
    )
    row = selection.selected
    if row is None or row.target is None or row.arithmetic is None:
        return {
            "stage": "planning_refused",
            "detail": "no selected circuit for the basis MPS profile",
        }
    circuit = row.target.submitted
    if (
        circuit.measurements
        or circuit.observables
        or any(
            _BASIS_GATE_WIDTH.get(gate.name) != len(gate.wires)
            or gate.matrix is not None
            or gate.params
            or getattr(gate, "parameter_slots", None)
            or gate.metadata
            for gate in circuit.instructions
        )
    ):
        return {
            "stage": "planning_refused",
            "detail": "basis MPS requires a closed I/X/Z/CX/CCX circuit",
        }

    resources = estimate_resources(circuit)
    storage = 32 * circuit.n_wires
    if storage > memory_bytes:
        return {
            "stage": "planning_refused",
            "detail": f"basis MPS tensors require {storage} bytes, limit is {memory_bytes}",
        }
    try:
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            state = run_mps(circuit, dtype=torch.complex128)
    except (ValueError, RuntimeError, RuntimeWarning) as error:
        return {"stage": "execution_failed", "detail": str(error)}
    if state.max_bond != 1 or state.truncation_records:
        return {
            "stage": "execution_failed",
            "detail": "basis MPS rank-one proof failed",
        }
    bits = []
    phase = 1.0 + 0j
    for site in state.tensors:
        zero, one = (complex(value) for value in site.reshape(2))
        if (zero == 0) == (one == 0):
            return {"stage": "execution_failed", "detail": "non-basis MPS site"}
        bit = zero == 0
        bits.append(str(int(bit)))
        phase *= one if bit else zero
    if not math.isclose(abs(phase), 1.0, abs_tol=1e-12, rel_tol=0):
        return {"stage": "execution_failed", "detail": "basis MPS norm is not one"}
    result: dict[str, str | int] = {
        "stage": "execution_smoke_passed",
        "execution_profile": "basis_mps",
        "qubits": resources.n_qubits,
        "operations": resources.n_operations,
        "depth": resources.depth,
        "workspace": row.arithmetic.peak_workspace,
        "state_bytes": storage,
        "basis_state": "".join(bits),
    }
    if any(gate.name == "z" for gate in circuit.instructions):
        sign = 1 if phase.real >= 0 else -1
        if not math.isclose(abs(phase - sign), 0.0, abs_tol=1e-12):
            return {
                "stage": "execution_failed",
                "detail": "basis MPS sign proof failed",
            }
        result["basis_sign"] = sign
    return result
