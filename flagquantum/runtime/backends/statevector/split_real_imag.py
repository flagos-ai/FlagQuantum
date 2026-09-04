"""Portable single-device statevector execution using split FP32 storage."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

import torch

from ....core.ir import CircuitIR, ensure_circuit_ir
from ....core.parameters import (
    Parameter,
    ParameterExpression,
    bind_parameter_value,
    parameter_names_in_value,
    value_to_tensor,
)
from ....providers.platform import get_platform_runtime, resolve_platform_device
from ....simulation.split_real_imag_statevector import (
    SPLIT_REAL_IMAG_SUPPORTED_GATES,
)
from ....simulation.split_real_imag_statevector import (
    apply_gate_pair as _apply_gate_pair,
)
from ....simulation.split_real_imag_statevector import (
    fixed_matrix_pair as _fixed_matrix,
)
from ....simulation.split_real_imag_statevector import (
    instruction_matrix_pair as _instruction_matrix_pair,
)

SPLIT_REAL_IMAG_SCHEMA = "flagquantum_split_real_imag_statevector_result_v1"
SPLIT_REAL_IMAG_PARAMETER_SHIFT_GATES = frozenset(
    {"rx", "ry", "rz", "rxx", "ryy", "rzz"}
)


@dataclass(frozen=True)
class SplitRealImagStatevectorResult:
    """One local statevector stored without a complex accelerator dtype."""

    real: torch.Tensor
    imag: torch.Tensor
    circuit_hash: str
    gate_count: int
    provider: str
    operator_profile: str
    operator_profile_hash: str
    operator_evidence_ids: tuple[str, ...]
    executor: str = "split_real_imag_statevector_p0"

    def __post_init__(self) -> None:
        if self.real.dtype != torch.float32 or self.imag.dtype != torch.float32:
            raise TypeError("split statevector storage must remain torch.float32")
        if self.real.shape != self.imag.shape or self.real.device != self.imag.device:
            raise ValueError("split statevector words must share shape and device")

    @property
    def device(self) -> torch.device:
        return self.real.device

    def probabilities(self) -> torch.Tensor:
        """Return probabilities on the execution device using only FP32 ops."""

        return self.real.square() + self.imag.square()

    def cpu_complex128(self) -> torch.Tensor:
        """Materialize a CPU-only diagnostic value for reference comparison."""

        real = self.real.detach().to(device="cpu", dtype=torch.float64)
        imag = self.imag.detach().to(device="cpu", dtype=torch.float64)
        return torch.complex(real, imag)

    def summary(self) -> dict[str, Any]:
        training = self.executor in {
            "split_real_imag_statevector_p1",
            "split_real_imag_statevector_p2_precision",
        }
        selective_double_single = (
            self.executor == "split_real_imag_statevector_p2_precision"
        )
        return {
            "schema": SPLIT_REAL_IMAG_SCHEMA,
            "executor": self.executor,
            "representation": "split_real_imag",
            "distribution_semantics": "single_device_fast_path",
            "device": str(self.device),
            "device_type": self.device.type,
            "provider": self.provider,
            "storage_dtype": "float32",
            "kernel_compute_dtype": "float32",
            "gate_generation_dtype": "float32",
            "complex_accelerator_tensor_materialized": False,
            "flagquantum_host_fallback": False,
            "provider_internal_route_audited": False,
            "observable_supported": training,
            "gradient_supported": training,
            "gradient_method": "parameter_shift" if training else None,
            "reduction_dtype": (
                "double_single_fp32" if selective_double_single else "float32"
            ),
            "native_autograd_supported": False,
            "distributed_supported": False,
            "scalability_claim_allowed": False,
            "runtime_default": False,
            "circuit_hash": self.circuit_hash,
            "gate_count": self.gate_count,
            "amplitude_count": self.real.numel(),
            "state_bytes": self.real.numel()
            * (self.real.element_size() + self.imag.element_size()),
            "operator_profile": self.operator_profile,
            "operator_profile_hash": self.operator_profile_hash,
            "operator_evidence_ids": self.operator_evidence_ids,
        }


@dataclass(frozen=True)
class SplitRealImagConformanceCase:
    depth: int
    max_abs_error: float
    norm_drift: float
    state_infidelity: float
    passed: bool


@dataclass(frozen=True)
class SplitRealImagConformanceReport:
    device: str
    provider: str
    cases: tuple[SplitRealImagConformanceCase, ...]
    passed: bool
    operator_profile: str
    operator_profile_hash: str
    hardware_certification: bool = False
    schema: str = "flagquantum_split_real_imag_conformance_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def require_accepted(self) -> None:
        if not self.passed:
            failed = tuple(case.depth for case in self.cases if not case.passed)
            raise RuntimeError(f"split real/imag conformance failed at depths {failed}")


@dataclass(frozen=True)
class SplitRealImagExpectationResult:
    """P1 Pauli/Hamiltonian expectation evaluated with split FP32 tensors."""

    state: SplitRealImagStatevectorResult
    value: torch.Tensor
    term_expectations: torch.Tensor
    term_count: int

    def summary(self) -> dict[str, Any]:
        summary = self.state.summary()
        summary.update(
            {
                "schema": "flagquantum_split_real_imag_expectation_v1",
                "observable": "pauli_hamiltonian",
                "term_count": self.term_count,
                "expectation_dtype": str(self.value.dtype).removeprefix("torch."),
            }
        )
        return summary


@dataclass(frozen=True)
class SplitRealImagParameterShiftResult:
    """Explicit parameter-shift gradient with a stable named-parameter order."""

    expectation: SplitRealImagExpectationResult
    parameter_order: tuple[str, ...]
    gradient: torch.Tensor
    shifted_evaluations: int
    shift: float = math.pi / 2.0

    def gradient_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: self.gradient[index]
            for index, name in enumerate(self.parameter_order)
        }

    def summary(self) -> dict[str, Any]:
        summary = self.expectation.summary()
        summary.update(
            {
                "schema": "flagquantum_split_real_imag_parameter_shift_v1",
                "gradient_method": "parameter_shift",
                "parameter_order": self.parameter_order,
                "shifted_evaluations": self.shifted_evaluations,
                "gradient_device": str(self.gradient.device),
            }
        )
        return summary


@dataclass(frozen=True)
class SplitRealImagTrainingConformanceCase:
    depth: int
    seed: int
    expectation_abs_error: float
    expectation_rel_error: float
    gradient_relative_error: float
    gradient_cosine_similarity: float
    norm_drift: float
    passed: bool


@dataclass(frozen=True)
class SplitRealImagTrainingConformanceReport:
    device: str
    provider: str
    cases: tuple[SplitRealImagTrainingConformanceCase, ...]
    passed: bool
    operator_profile: str
    operator_profile_hash: str
    hardware_certification: bool = False
    schema: str = "flagquantum_split_real_imag_training_conformance_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def require_accepted(self) -> None:
        if not self.passed:
            failed = tuple(
                (case.depth, case.seed) for case in self.cases if not case.passed
            )
            raise RuntimeError(
                f"split real/imag P1 conformance failed at depth/seed {failed}"
            )


@dataclass(frozen=True)
class _PauliTerm:
    coefficient: Any
    ops: tuple[tuple[int, str], ...]


def _execute_bound_split_statevector(
    ir: CircuitIR,
    *,
    device: str | torch.device,
    profile_name: str,
    preflight: bool,
    executor: str,
) -> SplitRealImagStatevectorResult:
    resolved_device = resolve_platform_device(device)
    platform = get_platform_runtime(resolved_device.type)
    identity = platform.identity()
    if preflight:
        from ...operator_probes import (
            preflight_split_real_imag_statevector_p0,
            preflight_split_real_imag_statevector_p1,
            preflight_split_real_imag_statevector_p2,
        )

        preflight_fn = {
            "split_real_imag_statevector_p0": preflight_split_real_imag_statevector_p0,
            "split_real_imag_statevector_p1": preflight_split_real_imag_statevector_p1,
            "split_real_imag_statevector_p2_precision": preflight_split_real_imag_statevector_p2,
        }.get(profile_name)
        if preflight_fn is None:
            raise ValueError(
                f"unknown split real/imag operator profile {profile_name!r}"
            )
        operator_report = preflight_fn(
            device=resolved_device,
            provider=identity.provider,
        )
        operator_report.require_supported()
    else:
        from ...capabilities import load_operator_profile

        profile = load_operator_profile(profile_name)
        operator_report = None
        operator_profile = profile.name
        operator_profile_hash = profile.profile_hash
        operator_evidence_ids: tuple[str, ...] = ()
    if operator_report is not None:
        operator_profile = operator_report.profile
        operator_profile_hash = operator_report.profile_hash
        operator_evidence_ids = tuple(operator_report.evidence_ids)
    amplitude_count = 2**ir.n_wires
    real = torch.zeros(amplitude_count, dtype=torch.float32, device=resolved_device)
    imag = torch.zeros_like(real)
    real[0] = 1.0
    for instruction in ir.instructions:
        matrix_real, matrix_imag = _instruction_matrix_pair(
            instruction, device=resolved_device
        )
        real, imag = _apply_gate_pair(
            real,
            imag,
            matrix_real,
            matrix_imag,
            instruction.wires,
            n_wires=ir.n_wires,
        )
        if (
            real.device.type != resolved_device.type
            or imag.device.type != resolved_device.type
        ):
            raise RuntimeError("split statevector escaped the requested logical device")
    return SplitRealImagStatevectorResult(
        real=real,
        imag=imag,
        circuit_hash=ir.content_hash,
        gate_count=len(ir.instructions),
        provider=identity.provider,
        operator_profile=operator_profile,
        operator_profile_hash=operator_profile_hash,
        operator_evidence_ids=operator_evidence_ids,
        executor=executor,
    )


def execute_split_real_imag_statevector(
    circuit_or_ir: Any,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    preflight: bool = True,
) -> SplitRealImagStatevectorResult:
    """Execute a forward-only local statevector with split FP32 amplitudes."""

    if dtype != torch.float32:
        raise ValueError("split real/imag P0 supports torch.float32 only")
    if getattr(circuit_or_ir, "_inputs", None) is not None:
        raise NotImplementedError("split real/imag P0 supports |0...0> input only")
    ir: CircuitIR = ensure_circuit_ir(circuit_or_ir)
    batch_size = int(ir.metadata.get("batch_size", 1))
    if batch_size != 1:
        raise NotImplementedError("split real/imag P0 supports batch size one only")
    if ir.observables or ir.measurements:
        raise NotImplementedError(
            "split real/imag P0 returns state only; observables and measurements "
            "are not supported"
        )
    for instruction in ir.instructions:
        if any(
            isinstance(value, torch.Tensor) and value.requires_grad
            for value in instruction.params.values()
        ):
            raise NotImplementedError(
                "split real/imag P0 is forward-only and rejects trainable parameters"
            )
    return _execute_bound_split_statevector(
        ir,
        device=device,
        profile_name="split_real_imag_statevector_p0",
        preflight=preflight,
        executor="split_real_imag_statevector_p0",
    )


def _normalized_bindings(
    parameter_bindings: Mapping[str | Parameter, Any],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_name, value in parameter_bindings.items():
        name = raw_name.name if isinstance(raw_name, Parameter) else str(raw_name)
        if not name or name in normalized:
            raise ValueError(f"duplicate or empty parameter binding {name!r}")
        tensor = value_to_tensor(value)
        if tensor.numel() != 1:
            raise ValueError("split real/imag P1 requires scalar parameter bindings")
        if tensor.is_complex() and bool(torch.any(tensor.imag != 0)):
            raise ValueError("split real/imag P1 gate parameters must be real")
        normalized[name] = (
            tensor.real.detach() if tensor.is_complex() else tensor.detach()
        )
    return normalized


def _parameter_occurrences(
    ir: CircuitIR,
) -> dict[str, tuple[tuple[int, str], ...]]:
    occurrences: dict[str, list[tuple[int, str]]] = {}
    for index, instruction in enumerate(ir.instructions):
        for parameter_name, value in instruction.params.items():
            names = parameter_names_in_value(value)
            if not names:
                if isinstance(value, torch.Tensor) and value.requires_grad:
                    raise ValueError(
                        "split real/imag P1 requires named Parameter bindings; "
                        "anonymous requires_grad gate tensors are ambiguous"
                    )
                continue
            if isinstance(value, ParameterExpression):
                raise NotImplementedError(
                    "split real/imag P1 parameter shift rejects ParameterExpression; "
                    "bind one named scalar directly to each supported rotation"
                )
            if not isinstance(value, Parameter):
                raise NotImplementedError(
                    "split real/imag P1 supports direct named Parameter values only"
                )
            if (
                instruction.name not in SPLIT_REAL_IMAG_PARAMETER_SHIFT_GATES
                or parameter_name != "theta"
            ):
                raise NotImplementedError(
                    "split real/imag P1 parameter shift supports theta on "
                    f"{sorted(SPLIT_REAL_IMAG_PARAMETER_SHIFT_GATES)}, got "
                    f"{instruction.name}.{parameter_name}"
                )
            occurrences.setdefault(value.name, []).append((index, parameter_name))
    return {name: tuple(items) for name, items in occurrences.items()}


def _bind_p1_ir(
    ir: CircuitIR,
    bindings: Mapping[str, Any],
    *,
    shifted_occurrence: tuple[int, str, float] | None = None,
) -> CircuitIR:
    instructions = []
    for index, instruction in enumerate(ir.instructions):
        params: dict[str, Any] = {}
        for parameter_name, value in instruction.params.items():
            if isinstance(value, (Parameter, ParameterExpression)):
                bound = bind_parameter_value(value, bindings)
            else:
                bound = value
            tensor = value_to_tensor(bound)
            if tensor.numel() != 1:
                raise ValueError("split real/imag P1 gate parameters must be scalar")
            if (
                shifted_occurrence is not None
                and (
                    index,
                    parameter_name,
                )
                == shifted_occurrence[:2]
            ):
                tensor = tensor + shifted_occurrence[2]
            params[parameter_name] = tensor.detach()
        instructions.append(replace(instruction, params=params))
    return replace(
        ir,
        instructions=tuple(instructions),
        observables=(),
        measurements=(),
    )


def _normalize_pauli_ops(
    pauli: str | Mapping[int, str], wires: Sequence[int]
) -> tuple[tuple[int, str], ...]:
    if isinstance(pauli, Mapping):
        raw_ops = tuple((int(wire), str(name).lower()) for wire, name in pauli.items())
    else:
        names = str(pauli).lower()
        if len(names) != len(wires):
            raise ValueError("Pauli string length must match its wires")
        raw_ops = tuple(zip((int(wire) for wire in wires), names))
    normalized = []
    seen = set()
    for wire, name in raw_ops:
        if name not in {"i", "x", "y", "z"}:
            raise ValueError("split P1 observables support Pauli I, X, Y, Z only")
        if wire in seen:
            raise ValueError("a split P1 Pauli term cannot repeat a wire")
        seen.add(wire)
        if name != "i":
            normalized.append((wire, name))
    return tuple(sorted(normalized))


def _normalized_observables(
    ir: CircuitIR, observable: Any | None
) -> tuple[_PauliTerm, ...]:
    if observable is not None and ir.observables:
        raise ValueError(
            "provide observables through either FlagQuantum IR or argument"
        )
    if observable is None:
        raw_terms: Sequence[Any] = ir.observables
    elif hasattr(observable, "terms"):
        raw_terms = tuple(observable.terms)
    elif hasattr(observable, "ops") and hasattr(observable, "coefficient"):
        raw_terms = (observable,)
    elif isinstance(observable, Sequence) and not isinstance(observable, (str, bytes)):
        raw_terms = observable
    else:
        raise TypeError(
            "observable must be a Hamiltonian, HamiltonianTerm, term sequence, "
            "or FlagQuantum IR observables"
        )
    if not raw_terms:
        raise ValueError("split real/imag P1 requires at least one observable term")
    terms = []
    for raw in raw_terms:
        if hasattr(raw, "ops") and hasattr(raw, "coefficient"):
            coefficient = raw.coefficient
            ops = tuple((int(wire), str(name).lower()) for wire, name in raw.ops)
        else:
            name = str(raw.name).lower()
            pauli = raw.metadata.get("pauli", name)
            coefficient = raw.coefficient
            ops = _normalize_pauli_ops(pauli, raw.wires)
        coefficient_tensor = value_to_tensor(coefficient)
        if coefficient_tensor.numel() != 1 or coefficient_tensor.requires_grad:
            raise ValueError("split real/imag P1 coefficients must be fixed scalars")
        if coefficient_tensor.is_complex() and bool(
            torch.any(coefficient_tensor.imag != 0)
        ):
            raise ValueError(
                "split real/imag P1 requires real Hamiltonian coefficients"
            )
        for wire, pauli_name in ops:
            if wire < 0 or wire >= ir.n_wires or pauli_name not in {"x", "y", "z"}:
                raise ValueError(
                    "invalid Pauli operator or wire in split P1 observable"
                )
        terms.append(_PauliTerm(coefficient=coefficient_tensor.detach(), ops=ops))
    return tuple(terms)


def _pauli_term_expectation(
    state: SplitRealImagStatevectorResult,
    term: _PauliTerm,
    *,
    n_wires: int,
) -> torch.Tensor:
    transformed_real = state.real
    transformed_imag = state.imag
    for wire, name in term.ops:
        matrix_real, matrix_imag = _fixed_matrix(name, device=state.device)
        transformed_real, transformed_imag = _apply_gate_pair(
            transformed_real,
            transformed_imag,
            matrix_real,
            matrix_imag,
            (wire,),
            n_wires=n_wires,
        )
    base = torch.sum(state.real * transformed_real + state.imag * transformed_imag)
    coefficient = term.coefficient.to(device=state.device, dtype=torch.float32)
    return coefficient * base


def _expectation_from_bound_p1_ir(
    ir: CircuitIR,
    terms: Sequence[_PauliTerm],
    *,
    device: str | torch.device,
    preflight: bool,
) -> SplitRealImagExpectationResult:
    state = _execute_bound_split_statevector(
        ir,
        device=device,
        profile_name="split_real_imag_statevector_p1",
        preflight=preflight,
        executor="split_real_imag_statevector_p1",
    )
    term_values = torch.stack(
        [_pauli_term_expectation(state, term, n_wires=ir.n_wires) for term in terms]
    )
    return SplitRealImagExpectationResult(
        state=state,
        value=torch.sum(term_values),
        term_expectations=term_values,
        term_count=len(terms),
    )


def _validate_p1_execution_scope(circuit_or_ir: Any, ir: CircuitIR) -> None:
    if getattr(circuit_or_ir, "_inputs", None) is not None:
        raise NotImplementedError("split real/imag P1 supports |0...0> input only")
    if int(ir.metadata.get("batch_size", 1)) != 1:
        raise NotImplementedError("split real/imag P1 supports batch size one only")
    if ir.measurements:
        raise NotImplementedError("split real/imag P1 does not support sampling")


def execute_split_real_imag_expectation(
    circuit_or_ir: Any,
    observable: Any | None = None,
    *,
    parameter_bindings: Mapping[str | Parameter, Any] | None = None,
    device: str | torch.device = "cpu",
    preflight: bool = True,
) -> SplitRealImagExpectationResult:
    """Evaluate a P1 Pauli Hamiltonian without complex accelerator tensors."""

    ir = ensure_circuit_ir(circuit_or_ir)
    _validate_p1_execution_scope(circuit_or_ir, ir)
    occurrences = _parameter_occurrences(ir)
    bindings = _normalized_bindings(parameter_bindings or {})
    if set(bindings) != set(occurrences):
        raise ValueError(
            "parameter bindings must exactly match named circuit parameters; "
            f"expected {sorted(occurrences)}, got {sorted(bindings)}"
        )
    terms = _normalized_observables(ir, observable)
    bound_ir = _bind_p1_ir(ir, bindings)
    return _expectation_from_bound_p1_ir(
        bound_ir, terms, device=device, preflight=preflight
    )


def parameter_shift_split_real_imag_gradient(
    circuit_or_ir: Any,
    observable: Any | None = None,
    *,
    parameter_bindings: Mapping[str | Parameter, Any],
    device: str | torch.device = "cpu",
    preflight: bool = True,
) -> SplitRealImagParameterShiftResult:
    """Return explicit P1 gradients using occurrence-wise two-term shifts."""

    ir = ensure_circuit_ir(circuit_or_ir)
    _validate_p1_execution_scope(circuit_or_ir, ir)
    occurrences = _parameter_occurrences(ir)
    bindings = _normalized_bindings(parameter_bindings)
    if set(bindings) != set(occurrences) or not occurrences:
        raise ValueError(
            "parameter bindings must exactly match at least one named circuit "
            f"parameter; expected {sorted(occurrences)}, got {sorted(bindings)}"
        )
    terms = _normalized_observables(ir, observable)
    base_ir = _bind_p1_ir(ir, bindings)
    base = _expectation_from_bound_p1_ir(
        base_ir, terms, device=device, preflight=preflight
    )
    gradients = []
    shifted_evaluations = 0
    for name in sorted(occurrences):
        gradient = torch.zeros((), dtype=torch.float32, device=base.value.device)
        for instruction_index, parameter_name in occurrences[name]:
            plus_ir = _bind_p1_ir(
                ir,
                bindings,
                shifted_occurrence=(instruction_index, parameter_name, math.pi / 2.0),
            )
            minus_ir = _bind_p1_ir(
                ir,
                bindings,
                shifted_occurrence=(instruction_index, parameter_name, -math.pi / 2.0),
            )
            plus = _expectation_from_bound_p1_ir(
                plus_ir, terms, device=device, preflight=False
            ).value
            minus = _expectation_from_bound_p1_ir(
                minus_ir, terms, device=device, preflight=False
            ).value
            gradient = gradient + 0.5 * (plus - minus)
            shifted_evaluations += 2
        gradients.append(gradient)
    return SplitRealImagParameterShiftResult(
        expectation=base,
        parameter_order=tuple(sorted(occurrences)),
        gradient=torch.stack(gradients),
        shifted_evaluations=shifted_evaluations,
    )


def _conformance_ir(depth: int) -> CircuitIR:
    from ....circuit import Circuit

    circuit = Circuit(3, device="cpu", dtype=torch.complex64)
    circuit.h(0)
    for layer in range(depth):
        angle = (layer + 1) * 0.017
        circuit.rx(layer % 3, theta=angle)
        circuit.ry((layer + 2) % 3, theta=-0.31 * angle)
        circuit.rz((layer + 1) % 3, theta=-0.7 * angle)
        circuit.cx(layer % 3, (layer + 1) % 3)
        circuit.rzz((layer + 1) % 3, (layer + 2) % 3, theta=0.23 * angle)
    return circuit.to_ir()


def _state_metrics(
    candidate: torch.Tensor, reference: torch.Tensor
) -> tuple[float, float, float]:
    candidate = candidate.reshape(-1).to(torch.complex128)
    reference = reference.detach().cpu().reshape(-1).to(torch.complex128)
    candidate_norm = torch.linalg.vector_norm(candidate)
    reference_norm = torch.linalg.vector_norm(reference)
    denominator = candidate_norm.square() * reference_norm.square()
    # PyTorch 2.5's Linux aarch64 CPU wheel can return zero from complex128
    # torch.vdot for non-zero inputs.  Keep the CPU reference portable by
    # spelling out the mathematically equivalent inner product.
    overlap = torch.abs(torch.sum(torch.conj(reference) * candidate)).square()
    return (
        float(torch.max(torch.abs(candidate - reference)).item()),
        abs(float(candidate_norm.item()) - 1.0),
        max(0.0, 1.0 - float((overlap / denominator).real.item())),
    )


def run_split_real_imag_conformance(
    device: str | torch.device = "cpu",
    *,
    depths: Sequence[int] = (8, 32, 128),
    max_abs_error: float = 5e-5,
    max_norm_drift: float = 5e-5,
    max_state_infidelity: float = 5e-6,
) -> SplitRealImagConformanceReport:
    """Compare the device-resident split path with CPU complex128."""

    from ....circuit import Circuit

    cases = []
    first_result = None
    for depth in tuple(int(value) for value in depths):
        ir = _conformance_ir(depth)
        result = execute_split_real_imag_statevector(ir, device=device)
        reference = Circuit.from_ir(ir, device="cpu", dtype=torch.complex128).state()
        error, norm_drift, infidelity = _state_metrics(
            result.cpu_complex128(), reference
        )
        passed = bool(
            error <= max_abs_error
            and norm_drift <= max_norm_drift
            and infidelity <= max_state_infidelity
        )
        cases.append(
            SplitRealImagConformanceCase(
                depth=depth,
                max_abs_error=error,
                norm_drift=norm_drift,
                state_infidelity=infidelity,
                passed=passed,
            )
        )
        first_result = first_result or result
    assert first_result is not None
    report = SplitRealImagConformanceReport(
        device=str(first_result.device),
        provider=first_result.provider,
        cases=tuple(cases),
        passed=all(case.passed for case in cases),
        operator_profile=first_result.operator_profile,
        operator_profile_hash=first_result.operator_profile_hash,
    )
    return report


def _training_conformance_ir(depth: int, seed: int) -> CircuitIR:
    from ....circuit import Circuit

    circuit = Circuit(3, device="cpu", dtype=torch.complex64).h(0)
    for layer in range(depth):
        angle = (layer + 1) * 0.013 + seed * 0.001
        circuit.ry(layer % 3, theta=angle)
        circuit.rz((layer + 1) % 3, theta=-0.47 * angle)
        circuit.cx(layer % 3, (layer + 1) % 3)
        circuit.rxx((layer + 1) % 3, (layer + 2) % 3, theta=0.19 * angle)
    circuit.rx(0, theta=Parameter("alpha"))
    circuit.ry(1, theta=Parameter("beta"))
    circuit.rzz(1, 2, theta=Parameter("gamma"))
    return circuit.to_ir()


def _training_conformance_observable() -> Any:
    from ....algorithms import Hamiltonian, pauli_term

    return Hamiltonian(
        (
            pauli_term(0.7, "Z", (0,)),
            pauli_term(-0.4, "XX", (1, 2)),
            pauli_term(0.2, "Y", (2,)),
        )
    )


def _complex128_pauli_expectation(
    ir: CircuitIR, terms: Sequence[_PauliTerm]
) -> torch.Tensor:
    from ....circuit import Circuit
    from ....ops.matrices import GATE_MAT_DICT
    from ....simulation.statevector_ops import _apply_matrix

    state = Circuit.from_ir(ir, device="cpu", dtype=torch.complex128).state()
    batched = state.reshape(1, -1)
    values = []
    for term in terms:
        transformed = batched
        for wire, name in term.ops:
            transformed = _apply_matrix(
                transformed,
                GATE_MAT_DICT[name].to(dtype=torch.complex128),
                (wire,),
                ir.n_wires,
            )
        coefficient = term.coefficient.to(dtype=torch.float64, device="cpu")
        values.append(
            coefficient * torch.real(torch.sum(torch.conj(batched) * transformed))
        )
    return torch.stack(values).sum()


def _complex128_parameter_shift(
    ir: CircuitIR,
    terms: Sequence[_PauliTerm],
    bindings: Mapping[str, Any],
    occurrences: Mapping[str, Sequence[tuple[int, str]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    value = _complex128_pauli_expectation(_bind_p1_ir(ir, bindings), terms)
    gradients = []
    for name in sorted(occurrences):
        gradient = torch.zeros((), dtype=torch.float64)
        for instruction_index, parameter_name in occurrences[name]:
            plus = _complex128_pauli_expectation(
                _bind_p1_ir(
                    ir,
                    bindings,
                    shifted_occurrence=(
                        instruction_index,
                        parameter_name,
                        math.pi / 2.0,
                    ),
                ),
                terms,
            )
            minus = _complex128_pauli_expectation(
                _bind_p1_ir(
                    ir,
                    bindings,
                    shifted_occurrence=(
                        instruction_index,
                        parameter_name,
                        -math.pi / 2.0,
                    ),
                ),
                terms,
            )
            gradient = gradient + 0.5 * (plus - minus)
        gradients.append(gradient)
    return value, torch.stack(gradients)


def run_split_real_imag_training_conformance(
    device: str | torch.device = "cpu",
    *,
    depths: Sequence[int] = (8, 32, 128),
    seeds: Sequence[int] = (0, 7),
    max_expectation_abs_error: float = 5e-5,
    max_expectation_rel_error: float = 5e-4,
    max_gradient_relative_error: float = 2e-3,
    min_gradient_cosine_similarity: float = 0.999,
    max_norm_drift: float = 5e-5,
) -> SplitRealImagTrainingConformanceReport:
    """Certify P1 expectation and parameter-shift gradients against complex128."""

    cases = []
    first_result = None
    observable = _training_conformance_observable()
    for depth in tuple(int(value) for value in depths):
        for seed in tuple(int(value) for value in seeds):
            ir = _training_conformance_ir(depth, seed)
            bindings = {
                "alpha": torch.tensor(0.17 + seed * 0.003, requires_grad=True),
                "beta": torch.tensor(-0.29 + depth * 0.0002, requires_grad=True),
                "gamma": torch.tensor(0.11 - seed * 0.002, requires_grad=True),
            }
            result = parameter_shift_split_real_imag_gradient(
                ir,
                observable,
                parameter_bindings=bindings,
                device=device,
                preflight=first_result is None,
            )
            terms = _normalized_observables(ir, observable)
            normalized_bindings = _normalized_bindings(bindings)
            reference_value, reference_gradient = _complex128_parameter_shift(
                ir,
                terms,
                normalized_bindings,
                _parameter_occurrences(ir),
            )
            candidate_value = result.expectation.value.detach().cpu().to(torch.float64)
            candidate_gradient = result.gradient.detach().cpu().to(torch.float64)
            expectation_abs_error = float(
                torch.abs(candidate_value - reference_value).item()
            )
            expectation_rel_error = expectation_abs_error / max(
                float(torch.abs(reference_value).item()), 1e-12
            )
            gradient_delta = torch.linalg.vector_norm(
                candidate_gradient - reference_gradient
            )
            reference_gradient_norm = torch.linalg.vector_norm(reference_gradient)
            candidate_gradient_norm = torch.linalg.vector_norm(candidate_gradient)
            gradient_relative_error = float(
                (
                    gradient_delta / torch.clamp(reference_gradient_norm, min=1e-12)
                ).item()
            )
            gradient_cosine_similarity = float(
                (
                    torch.dot(candidate_gradient, reference_gradient)
                    / torch.clamp(
                        candidate_gradient_norm * reference_gradient_norm, min=1e-12
                    )
                ).item()
            )
            norm = torch.sqrt(result.expectation.state.probabilities().sum())
            norm_drift = abs(float(norm.detach().cpu().item()) - 1.0)
            passed = bool(
                expectation_abs_error <= max_expectation_abs_error
                and expectation_rel_error <= max_expectation_rel_error
                and gradient_relative_error <= max_gradient_relative_error
                and gradient_cosine_similarity >= min_gradient_cosine_similarity
                and norm_drift <= max_norm_drift
            )
            cases.append(
                SplitRealImagTrainingConformanceCase(
                    depth=depth,
                    seed=seed,
                    expectation_abs_error=expectation_abs_error,
                    expectation_rel_error=expectation_rel_error,
                    gradient_relative_error=gradient_relative_error,
                    gradient_cosine_similarity=gradient_cosine_similarity,
                    norm_drift=norm_drift,
                    passed=passed,
                )
            )
            first_result = first_result or result
    assert first_result is not None
    return SplitRealImagTrainingConformanceReport(
        device=str(first_result.expectation.state.device),
        provider=first_result.expectation.state.provider,
        cases=tuple(cases),
        passed=all(case.passed for case in cases),
        operator_profile=first_result.expectation.state.operator_profile,
        operator_profile_hash=first_result.expectation.state.operator_profile_hash,
    )


__all__ = (
    "SPLIT_REAL_IMAG_SCHEMA",
    "SPLIT_REAL_IMAG_PARAMETER_SHIFT_GATES",
    "SPLIT_REAL_IMAG_SUPPORTED_GATES",
    "SplitRealImagConformanceCase",
    "SplitRealImagConformanceReport",
    "SplitRealImagExpectationResult",
    "SplitRealImagParameterShiftResult",
    "SplitRealImagStatevectorResult",
    "SplitRealImagTrainingConformanceCase",
    "SplitRealImagTrainingConformanceReport",
    "execute_split_real_imag_expectation",
    "execute_split_real_imag_statevector",
    "parameter_shift_split_real_imag_gradient",
    "run_split_real_imag_conformance",
    "run_split_real_imag_training_conformance",
)
