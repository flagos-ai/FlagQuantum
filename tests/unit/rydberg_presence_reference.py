"""Disposable proposal 092 composition; never a native source or sampled plan.

Physical consumers access this module only through their four factories.
Stage tests inspect its disposable format. Delete it after domain replacement.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import Any

import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.core.ir import IRValidationError
from flagquantum.core.rydberg import RydbergProgram
from flagquantum.errors import CapabilityError, ExecutionError, SerializationError
from flagquantum.lindblad import _execution_settings, _observables
from flagquantum.lindblad._plan import _decode_request
from flagquantum.operators import SuperOperator
from flagquantum.runtime import measurements
from flagquantum.runtime._terminal_instruments import _instrument_probabilities
from flagquantum.runtime.execution_plan_contract import (
    ExecutionPlanContractError,
    canonical_hash,
    canonical_json,
)
from flagquantum.runtime.result import ExecutionResult, MeasurementResult
from flagquantum.simulation.lindblad import EvolutionValidationError
from flagquantum.simulation.lindblad_generator import Liouvillian
from flagquantum.simulation.lindblad_integrators import advance_interval
from flagquantum.simulation.lindblad_request import (
    _normalize_collapse_operators,
    _solve_max_iterations,
)


class ReferenceSource(RydbergProgram):
    """Test frontend holding an observation outside its unchanged source 1.0."""

    observation: tuple[str, float, float, int] | None = None

    def measure_presence(
        self, atom: str, *, at: float, efficiency: float
    ) -> ReferenceSource:
        if not isinstance(atom, str) or any(
            type(value) not in (int, float) for value in (at, efficiency)
        ):
            raise TypeError("presence requires an atom ID and real Python scalars")
        if (
            atom not in self.atom_ids
            or self.observation is not None
            or not math.isfinite(at)
            or not math.isfinite(efficiency)
            or at <= 0
            or not 0 <= efficiency <= 1
        ):
            raise IRValidationError("invalid or repeated presence observation")
        index = len(self.to_ir().functions[0].blocks[0].operations)
        self.observation = (atom, float(at), float(efficiency), index)
        return self


@dataclass(frozen=True)
class ReferencePlan:
    """An immutable test recipe containing an authentic old numeric plan."""

    text: str

    @property
    def identity(self) -> str:
        value: str = json.loads(self.text)["identity"]
        return value

    def to_json(self, *, indent: int | None = None) -> str:
        return (
            self.text
            if indent is None
            else json.dumps(json.loads(self.text), indent=indent)
        )


def _trainable(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return value.requires_grad
    if isinstance(value, dict):
        return any(_trainable(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return any(_trainable(v) for v in value)
    return False


def plan(
    source: ReferenceSource,
    initial_state: torch.Tensor,
    times: torch.Tensor,
    *,
    options: fq.ExecutionOptions | None = None,
    **settings: Any,
) -> ReferencePlan:
    if options is None or options.shots != 1 or type(options.shots) is not int:
        raise CapabilityError("presence requires explicit shots=1 and seed")
    if options.seed is None:
        raise CapabilityError("presence requires explicit shots=1 and seed")
    if type(options.seed) is not int:
        raise TypeError("seed must be a Python integer")
    if not 0 <= options.seed <= 2**64 - 1:
        raise ValueError("seed must be in [0, 2**64-1]")
    numeric_options = replace(options, shots=None, seed=None)
    device, dtype, batch, gradients = _execution_settings(numeric_options)
    if (
        torch.device(device).type != "cpu"
        or dtype != torch.complex128
        or batch not in (None, 1)
        or gradients
        or initial_state.device.type != "cpu"
        or initial_state.requires_grad
        or initial_state.ndim > 2
        or initial_state.ndim == 2
        and initial_state.shape[0] != initial_state.shape[1]
    ):
        raise CapabilityError(
            "presence supports one detached CPU complex128 trajectory"
        )
    observation = source.observation
    if observation is None:
        raise CapabilityError("test presence recipe requires an observation")
    if times.requires_grad:
        raise CapabilityError("presence time grids cannot require gradients")
    if _trainable(settings.get("parameter_values")) or _trainable(
        _observables(settings.get("outputs"))
    ):
        raise CapabilityError("presence controls and outputs cannot require gradients")
    atom, at, efficiency, operation_index = observation
    matches = (times - at).abs() <= 1e-12
    if int(matches.sum()) != 1 or not bool(matches[1:-1].any()):
        raise EvolutionValidationError(
            "invalid_native_request",
            "presence needs one internal grid boundary",
            field="times",
        )
    native = fql.plan(source, initial_state, times, options=numeric_options, **settings)
    payload = native.to_dict()
    reserved = {"mid_circuit_measurement"}
    if payload["request"]["return_density_matrices"]:
        reserved.add("density_matrix")
    if any(
        name in reserved
        for name in (item["name"] for item in payload["request"]["observables"])
    ):
        raise EvolutionValidationError(
            "duplicate_output_name", "reserved presence output name", field="outputs"
        )
    recipe = {
        "schema": "test.rydberg_presence_recipe",
        "version": "1.0",
        "native_plan": payload,
        "observation": [atom, at, efficiency, operation_index],
        "seed": options.seed,
        "rng_torch_version": str(torch.__version__),
    }
    recipe["identity"] = canonical_hash(recipe)
    return read(canonical_json(recipe))


def read(text: str) -> ReferencePlan:
    recipe = json.loads(text)
    if (
        set(recipe)
        != {
            "schema",
            "version",
            "native_plan",
            "observation",
            "seed",
            "rng_torch_version",
            "identity",
        }
        or recipe["schema"] != "test.rydberg_presence_recipe"
        or recipe["version"] != "1.0"
        or recipe["identity"]
        != canonical_hash({k: v for k, v in recipe.items() if k != "identity"})
    ):
        raise SerializationError(
            "invalid test recipe; this is not a native sampled plan"
        )
    if recipe["rng_torch_version"] != str(torch.__version__):
        raise ExecutionPlanContractError(
            "environment_incompatible",
            "test recipe requires the captured Torch RNG build",
        )
    fql.LindbladPlan.from_dict(recipe["native_plan"])
    return ReferencePlan(canonical_json(recipe))


def run(plan: ReferencePlan) -> ExecutionResult:
    recipe = json.loads(read(plan.text).text)
    native = recipe["native_plan"]
    request = _decode_request(native["request"], native["decision"], version="1.5")
    atom, at, efficiency, source_index = recipe["observation"]
    roster = [entry[0] for entry in native["program"]["physical_context"]["atoms"]]
    count = len(roster)
    local = (
        torch.diag(
            torch.tensor([1, 1, math.sqrt(1 - efficiency)], dtype=torch.complex128)
        ),
        torch.diag(torch.tensor([0, 0, math.sqrt(efficiency)], dtype=torch.complex128)),
    )
    operators = []
    for operator in local:
        full = torch.ones((1, 1), dtype=torch.complex128)
        for name in roster:
            full = torch.kron(
                full, operator if name == atom else torch.eye(3, dtype=torch.complex128)
            )
        operators.append(full)
    grid = torch.tensor(request["times"], dtype=torch.float64)
    boundary = int(torch.nonzero((grid - at).abs() <= 1e-12)[0, 0])
    rho = request["initial_state"]
    collapse = _normalize_collapse_operators(
        request["collapse_operators"],
        n_wires=None,
        dim=3**count,
        dtype=torch.complex128,
        device=torch.device("cpu"),
        qubit_space=False,
        dimensions=(3,) * count,
    )
    states = [rho]
    event = None
    for index, matrix in enumerate(request["bound_hamiltonians"]):
        generator = Liouvillian(matrix, collapse, hilbert_dimension=3**count)
        rho, residual = advance_interval(
            generator.derivative,
            rho,
            grid[index + 1] - grid[index],
            method=request["method"],
            solve_tolerance=request["solve_tolerance"] or 1e-12,
            solve_max_iterations=_solve_max_iterations(3**count),
        )
        if residual > (request["solve_tolerance"] or 1e-12):
            raise ExecutionError("test recipe integration failed to converge")
        if index + 1 == boundary:
            branches: list[tuple[str, tuple[torch.Tensor, ...]]] = [
                (name, (op,))
                for name, op in zip(("no_flag", "flag"), operators, strict=True)
            ]
            weights, corrected = _instrument_probabilities(
                rho.unsqueeze(0), branches, 1e-10
            )
            selected = measurements._draw_probability_indices(
                corrected,
                1,
                generator=torch.Generator(device="cpu").manual_seed(recipe["seed"]),
            )
            if (
                selected.shape != (1, 1)
                or selected.dtype != torch.int64
                or selected.device.type != "cpu"
            ):
                raise ExecutionError("presence draw must be one CPU int64 index")
            outcome = int(selected[0, 0])
            if outcome not in (0, 1) or float(weights[0, outcome]) <= 0:
                raise ExecutionError(
                    "presence selected a zero-probability or invalid branch"
                )
            op = operators[outcome]
            rho = SuperOperator.from_kraus((op,)).apply(rho) / weights[0, outcome]
            event = MeasurementResult(
                "mid_circuit_measurement",
                (),
                selected,
                shots=1,
                metadata={
                    "outcome_labels": ["no_flag", "flag"],
                    "probabilities": weights.tolist(),
                    "probability_correction": float((corrected - weights).abs().max()),
                    "atom_id": atom,
                    "time_index": boundary,
                    "time": at,
                    "time_unit": native["program"]["physical_context"]["time_unit"],
                    "stage": "intermediate",
                    "source_operation_index": source_index,
                },
            )
        states.append(rho)
    trajectory = torch.stack(states)
    records = []
    for index, (name, operator) in enumerate(request["observables"].items()):
        values = torch.einsum("tij,ji->t", trajectory, operator).real
        records.append(
            MeasurementResult(
                "expectation",
                (),
                values,
                metadata={
                    "name": name,
                    "fq_output_name": name,
                    "fq_output_index": index,
                    "fq_output_kind": "expectation",
                },
            )
        )
    if request["return_density_matrices"]:
        records.append(
            MeasurementResult(
                "density_matrix",
                (),
                trajectory,
                metadata={"axes": ["time", "basis_row", "basis_column"]},
            )
        )
    assert event is not None
    records.append(event)
    return ExecutionResult(
        state=rho.unsqueeze(0),
        measurements=tuple(records),
        runtime={
            "mode": "density_matrix",
            "backend": "torch",
            "device": "cpu",
            "precision": "complex128",
            "world_size": 1,
            "is_distributed": False,
            "seed": recipe["seed"],
            "shots": 1,
            "time_grid": grid.tolist(),
            "state_axes": ["sample", "basis_row", "basis_column"],
            "expectation_axes": ["time"],
            "test_recipe": True,
        },
    )
