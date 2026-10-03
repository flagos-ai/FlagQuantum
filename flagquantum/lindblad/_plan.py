"""Immutable, serializable execution plans for Lindblad evolution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

import torch

from ..errors import SerializationError
from ..simulation.lindblad import (
    EvolutionPlan,
    _normalize_collapse_terms,
    _normalize_hamiltonian,
    _normalize_initial_state,
    _normalize_observables,
    _normalize_times,
    plan_density_matrix_evolution,
)
from ..simulation.lindblad_integrators import DEFAULT_INTEGRATOR
from ..simulation.matrix_free_hamiltonian import PauliSum

_SCHEMA = "flagquantum.lindblad_plan"
# 1.1 adds ``numerics.solve_tolerance`` to the decision, which version 1.0 did
# not record, and 1.2 widens ``numerics.order`` to admit a scheme that has no
# step-size order. 1.3 records the declared batch axis in the decision, which is
# also what the reported ``trajectory_bytes`` now counts. The stored decision is
# rebuilt from its request and compared, so an older payload cannot be
# reconstructed by this build; refusing it by version reports that fact instead
# of a misleading decision mismatch.
_VERSION = "1.3"


def _request_hamiltonian_matrix(
    hamiltonian: PauliSum | torch.Tensor,
) -> torch.Tensor:
    """Return the dense matrix the serialized request stores.

    The request payload is the one place that still needs ``H`` entry by entry,
    so this is the only remaining ``4**n`` Hamiltonian allocation on the
    planning path. Evolution itself keeps the Pauli sum matrix-free, and an
    in-process plan does not pass through this payload at all.
    """

    return hamiltonian.dense() if isinstance(hamiltonian, PauliSum) else hamiltonian


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identity(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("identity", None)
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def _complex_tensor_payload(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().to(device="cpu", dtype=torch.complex128)
    return {
        "shape": list(tensor.shape),
        "values": torch.stack((tensor.real, tensor.imag), dim=-1).tolist(),
    }


def _complex_tensor(payload: object, *, dtype: torch.dtype, field: str) -> torch.Tensor:
    if not isinstance(payload, Mapping):
        raise SerializationError(f"{field} must be a complex tensor payload")
    shape = payload.get("shape")
    values = payload.get("values")
    if not isinstance(shape, list) or any(type(item) is not int for item in shape):
        raise SerializationError(f"{field}.shape must contain integers")
    try:
        encoded = torch.tensor(values, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise SerializationError(f"{field}.values must be numeric") from exc
    if encoded.ndim < 1 or encoded.shape[-1] != 2:
        raise SerializationError(f"{field}.values must contain real/imaginary pairs")
    tensor = torch.complex(encoded[..., 0], encoded[..., 1]).to(dtype=dtype)
    if list(tensor.shape) != shape:
        raise SerializationError(f"{field}.shape does not match its values")
    return tensor


def _summary(payload: Mapping[str, Any]) -> EvolutionPlan:
    """Rebuild the decision record from a serialized payload.

    ``numerics.order`` is ``None`` for a scheme that has no step-size order, and
    ``batch_size`` sits beside ``trajectory_bytes`` because both count the batch
    axis. A payload that omits the batch axis is read as one member, which is
    what version 1.2 and earlier recorded implicitly.
    """

    numerics = payload.get("numerics")
    if not isinstance(numerics, Mapping):
        raise SerializationError("decision.numerics must be a mapping")
    solve_tolerance = numerics.get("solve_tolerance")
    if solve_tolerance is not None and (
        isinstance(solve_tolerance, bool)
        or not isinstance(solve_tolerance, (int, float))
    ):
        raise SerializationError("decision.numerics.solve_tolerance must be numeric")
    order = numerics.get("order")
    if order is not None and (isinstance(order, bool) or not isinstance(order, int)):
        raise SerializationError("decision.numerics.order must be an integer or null")
    require_gradients = numerics.get("require_gradients", False)
    if type(require_gradients) is not bool:
        raise SerializationError(
            "decision.numerics.require_gradients must be a boolean"
        )
    batch_size = payload.get("batch_size", 1)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise SerializationError("decision.batch_size must be an integer")
    try:
        return EvolutionPlan(
            n_wires=int(payload["n_wires"]),
            n_times=int(payload["n_times"]),
            dimension=int(payload["dimension"]),
            trajectory_bytes=int(payload["trajectory_bytes"]),
            method=str(numerics["method"]),
            order=order,
            step_size=float(numerics["step_size"]),
            device=str(numerics["device"]),
            precision=str(numerics["precision"]),
            solve_tolerance=(
                None if solve_tolerance is None else float(solve_tolerance)
            ),
            batch_size=batch_size,
            require_gradients=require_gradients,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SerializationError("invalid Lindblad plan decision") from exc


@dataclass(frozen=True, slots=True)
class LindbladPlan:
    """A sealed Lindblad request that can be inspected, serialized, and run.

    A plan built in this process also holds the live request it was validated
    from. Running it therefore keeps the matrix-free Hamiltonian and the
    autograd graph of the tensors the caller supplied, instead of rebuilding
    detached tensors from JSON. Nothing about the serialized form depends on
    that live request, and a plan restored by :meth:`from_json` has none.
    """

    _payload_json: str
    _live_request: dict[str, Any] | None = field(
        default=None, compare=False, repr=False
    )

    def to_dict(self) -> dict[str, Any]:
        payload = json.loads(self._payload_json)
        if not isinstance(payload, dict):
            raise SerializationError("stored Lindblad plan must contain an object")
        return cast(dict[str, Any], payload)

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return self._payload_json
        return json.dumps(
            self.to_dict(), sort_keys=True, indent=indent, allow_nan=False
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LindbladPlan:
        if not isinstance(payload, Mapping):
            raise TypeError("Lindblad plan payload must be a mapping")
        checked = dict(payload)
        if checked.get("schema") != _SCHEMA:
            raise SerializationError("invalid Lindblad plan schema")
        if checked.get("version") != _VERSION:
            raise SerializationError("unsupported Lindblad plan version")
        supplied_identity = checked.get("identity")
        if not isinstance(supplied_identity, str) or supplied_identity != _identity(
            checked
        ):
            raise SerializationError(
                "Lindblad plan identity does not match its content"
            )
        request = checked.get("request")
        decision = checked.get("decision")
        if not isinstance(request, Mapping) or not isinstance(decision, Mapping):
            raise SerializationError("Lindblad plan requires request and decision maps")
        decoded = _decode_request(request, decision)
        recomputed = plan_density_matrix_evolution(**decoded)
        if recomputed.to_dict() != dict(decision):
            raise SerializationError(
                "Lindblad plan decision does not match its request"
            )
        return cls(_canonical_json(checked))

    @classmethod
    def from_json(cls, text: str) -> LindbladPlan:
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SerializationError("invalid Lindblad plan JSON") from exc
        if not isinstance(payload, Mapping):
            raise SerializationError("Lindblad plan JSON must contain an object")
        return cls.from_dict(payload)

    @property
    def identity(self) -> str:
        return str(self.to_dict()["identity"])

    @property
    def n_qubits(self) -> int:
        return self._decision.n_qubits

    @property
    def n_times(self) -> int:
        return self._decision.n_times

    @property
    def dimension(self) -> int:
        return self._decision.dimension

    @property
    def trajectory_bytes(self) -> int:
        return self._decision.trajectory_bytes

    @property
    def method(self) -> str:
        return self._decision.method

    @property
    def order(self) -> int | None:
        return self._decision.order

    @property
    def step_size(self) -> float:
        return self._decision.step_size

    @property
    def device(self) -> str:
        return self._decision.device

    @property
    def precision(self) -> str:
        return self._decision.precision

    @property
    def solve_tolerance(self) -> float | None:
        """Return the implicit-solve residual target, or ``None`` if explicit."""

        return self._decision.solve_tolerance

    @property
    def require_gradients(self) -> bool:
        """Return whether the caller made the returned trajectory differentiable."""

        return self._decision.require_gradients

    @property
    def batch_size(self) -> int:
        """Return the declared batch axis the trajectory reports."""

        return self._decision.batch_size

    @property
    def _decision(self) -> EvolutionPlan:
        decision = self.to_dict()["decision"]
        assert isinstance(decision, Mapping)
        return _summary(decision)

    def _execution_request(self) -> dict[str, Any]:
        if self._live_request is not None:
            return dict(self._live_request)
        payload = self.to_dict()
        request = payload["request"]
        decision = payload["decision"]
        assert isinstance(request, Mapping)
        assert isinstance(decision, Mapping)
        return _decode_request(request, decision)


def build_lindblad_plan(
    hamiltonian: Any,
    initial_state: Any,
    n_qubits: int,
    times: Any,
    collapse_operators: Sequence[Any] | None,
    observables: Mapping[str, Any] | None,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    return_density_matrices: bool,
    method: str = DEFAULT_INTEGRATOR,
    solve_tolerance: float | None = None,
    batch_size: int | None = None,
    require_gradients: bool = False,
) -> LindbladPlan:
    summary = plan_density_matrix_evolution(
        hamiltonian,
        initial_state,
        n_qubits,
        times,
        collapse_operators,
        observables,
        device=device,
        dtype=dtype,
        return_density_matrices=return_density_matrices,
        method=method,
        solve_tolerance=solve_tolerance,
        batch_size=batch_size,
        require_gradients=require_gradients,
    )
    resolved_device = torch.device(device)
    dimension = 2**n_qubits
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    tolerance = 1e-5 if dtype == torch.complex64 else 1e-10
    time_grid = _normalize_times(
        times, device=resolved_device, real_dtype=real_dtype
    ).detach()
    hamiltonian_representation = _normalize_hamiltonian(
        hamiltonian,
        n_wires=n_qubits,
        dim=dimension,
        dtype=dtype,
        device=resolved_device,
        tolerance=tolerance,
    )
    density_matrix = _normalize_initial_state(
        initial_state,
        dim=dimension,
        dtype=dtype,
        device=resolved_device,
        tolerance=tolerance,
    )
    collapse_terms = _normalize_collapse_terms(
        collapse_operators,
        n_wires=n_qubits,
        dim=dimension,
        dtype=dtype,
        device=resolved_device,
    )
    observable_matrices = _normalize_observables(
        observables, dim=dimension, dtype=dtype, device=resolved_device
    )
    request = {
        "n_qubits": n_qubits,
        "hamiltonian": _complex_tensor_payload(
            _request_hamiltonian_matrix(hamiltonian_representation)
        ),
        "initial_density_matrix": _complex_tensor_payload(density_matrix),
        "times": time_grid.to(device="cpu", dtype=torch.float64).tolist(),
        "collapse_operators": [
            {"operator": _complex_tensor_payload(operator), "rate": rate}
            for operator, rate in collapse_terms
        ],
        "observables": [
            {"name": name, "operator": _complex_tensor_payload(operator)}
            for name, operator in observable_matrices.items()
        ],
        "return_density_matrices": return_density_matrices,
    }
    payload: dict[str, Any] = {
        "schema": _SCHEMA,
        "version": _VERSION,
        "request": request,
        "decision": summary.to_dict(),
    }
    payload["identity"] = _identity(payload)
    validated = LindbladPlan.from_dict(payload)
    # The live request is the same one the payload describes, holding the
    # tensors as the caller passed them. It is what makes execution keep the
    # matrix-free Hamiltonian and the autograd graph.
    live = {
        "hamiltonian": hamiltonian_representation,
        "initial_state": density_matrix,
        "n_wires": n_qubits,
        "times": time_grid,
        "collapse_operators": list(collapse_terms),
        "observables": dict(observable_matrices),
        "device": resolved_device,
        "dtype": dtype,
        "return_density_matrices": return_density_matrices,
        "method": summary.method,
        "solve_tolerance": summary.solve_tolerance,
        "batch_size": summary.batch_size,
        "require_gradients": summary.require_gradients,
    }
    return replace(validated, _live_request=live)


def _decode_request(
    request: Mapping[str, Any], decision: Mapping[str, Any]
) -> dict[str, Any]:
    summary = _summary(decision)
    dtype = {
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }.get(summary.precision)
    if dtype is None:
        raise SerializationError("unsupported Lindblad plan precision")
    try:
        n_qubits = int(request["n_qubits"])
        times = request["times"]
        return_density_matrices = request["return_density_matrices"]
        raw_collapse = request["collapse_operators"]
        raw_observables = request["observables"]
    except KeyError as exc:
        raise SerializationError(
            f"missing Lindblad plan field {exc.args[0]!r}"
        ) from exc
    if type(return_density_matrices) is not bool:
        raise SerializationError("return_density_matrices must be a boolean")
    if not isinstance(raw_collapse, list) or not isinstance(raw_observables, list):
        raise SerializationError("collapse_operators and observables must be lists")
    collapse_operators = []
    for index, item in enumerate(raw_collapse):
        if not isinstance(item, Mapping):
            raise SerializationError(f"collapse_operators[{index}] must be a mapping")
        try:
            rate = float(item["rate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SerializationError(
                f"collapse_operators[{index}].rate must be numeric"
            ) from exc
        collapse_operators.append(
            (
                _complex_tensor(
                    item.get("operator"),
                    dtype=dtype,
                    field=f"collapse_operators[{index}].operator",
                ),
                rate,
            )
        )
    observables: dict[str, torch.Tensor] = {}
    for index, item in enumerate(raw_observables):
        if not isinstance(item, Mapping):
            raise SerializationError(f"observables[{index}] must be a mapping")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise SerializationError(f"observables[{index}].name must be non-empty")
        if name in observables:
            raise SerializationError(f"duplicate observable name {name!r}")
        observables[name] = _complex_tensor(
            item.get("operator"), dtype=dtype, field=f"observables[{index}].operator"
        )
    return {
        "hamiltonian": _complex_tensor(
            request.get("hamiltonian"), dtype=dtype, field="hamiltonian"
        ),
        "initial_state": _complex_tensor(
            request.get("initial_density_matrix"),
            dtype=dtype,
            field="initial_density_matrix",
        ),
        "n_wires": n_qubits,
        "times": times,
        "collapse_operators": collapse_operators,
        "observables": observables,
        "device": summary.device,
        "dtype": dtype,
        "return_density_matrices": return_density_matrices,
        "method": summary.method,
        "solve_tolerance": summary.solve_tolerance,
        "batch_size": summary.batch_size,
        "require_gradients": summary.require_gradients,
    }


__all__ = ("LindbladPlan",)
