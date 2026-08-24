"""Portable single-device statevector execution using split FP32 storage."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch

from ....core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ....core.parameters import value_to_tensor
from ...platforms import get_platform_runtime, resolve_platform_device

SPLIT_REAL_IMAG_SCHEMA = "flagquantum_split_real_imag_statevector_result_v1"
SPLIT_REAL_IMAG_SUPPORTED_GATES = frozenset(
    {
        "i",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "sx",
        "sxdg",
        "rx",
        "ry",
        "rz",
        "phase",
        "u1",
        "u2",
        "u3",
        "cx",
        "cy",
        "cz",
        "swap",
        "crx",
        "cry",
        "crz",
        "cphase",
        "rxx",
        "ryy",
        "rzz",
    }
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
        return {
            "schema": SPLIT_REAL_IMAG_SCHEMA,
            "executor": "split_real_imag_statevector_p0",
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
            "gradient_supported": False,
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


def _scalar(value: Any, *, device: torch.device) -> torch.Tensor:
    tensor = value_to_tensor(value)
    if tensor.numel() != 1:
        raise ValueError("split real/imag P0 requires scalar gate parameters")
    if tensor.requires_grad:
        raise NotImplementedError(
            "split real/imag P0 is forward-only and rejects trainable parameters"
        )
    if tensor.is_complex():
        if bool(torch.any(tensor.imag != 0)):
            raise ValueError("gate angles must be real")
        tensor = tensor.real
    return tensor.detach().to(device=device, dtype=torch.float32).reshape(())


def _matrix(
    entries: Sequence[Sequence[tuple[Any, Any]]], *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    width = len(entries)
    if width == 0 or any(len(row) != width for row in entries):
        raise ValueError("gate matrix entries must be non-empty and square")
    real = torch.stack(
        [_scalar(item[0], device=device) for row in entries for item in row]
    ).reshape(width, width)
    imag = torch.stack(
        [_scalar(item[1], device=device) for row in entries for item in row]
    ).reshape(width, width)
    return real, imag


def _zero_entries(width: int) -> list[list[tuple[Any, Any]]]:
    return [[(0.0, 0.0) for _ in range(width)] for _ in range(width)]


def _fixed_matrix(
    name: str, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    q = 1.0 / math.sqrt(2.0)
    fixed: Mapping[str, Sequence[Sequence[tuple[Any, Any]]]] = {
        "i": (((1, 0), (0, 0)), ((0, 0), (1, 0))),
        "x": (((0, 0), (1, 0)), ((1, 0), (0, 0))),
        "y": (((0, 0), (0, -1)), ((0, 1), (0, 0))),
        "z": (((1, 0), (0, 0)), ((0, 0), (-1, 0))),
        "h": (((q, 0), (q, 0)), ((q, 0), (-q, 0))),
        "s": (((1, 0), (0, 0)), ((0, 0), (0, 1))),
        "sdg": (((1, 0), (0, 0)), ((0, 0), (0, -1))),
        "t": (((1, 0), (0, 0)), ((0, 0), (q, q))),
        "tdg": (((1, 0), (0, 0)), ((0, 0), (q, -q))),
        "sx": (((0.5, 0.5), (0.5, -0.5)), ((0.5, -0.5), (0.5, 0.5))),
        "sxdg": (((0.5, -0.5), (0.5, 0.5)), ((0.5, 0.5), (0.5, -0.5))),
    }
    if name in fixed:
        return _matrix(fixed[name], device=device)
    entries = _zero_entries(4)
    if name == "cx":
        for row, column in enumerate((0, 1, 3, 2)):
            entries[row][column] = (1, 0)
    elif name == "cy":
        entries[0][0] = entries[1][1] = (1, 0)
        entries[2][3] = (0, -1)
        entries[3][2] = (0, 1)
    elif name == "cz":
        for index, value in enumerate((1, 1, 1, -1)):
            entries[index][index] = (value, 0)
    elif name == "swap":
        for row, column in enumerate((0, 2, 1, 3)):
            entries[row][column] = (1, 0)
    else:
        raise KeyError(name)
    return _matrix(entries, device=device)


def _rotation_pair(angle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    half = angle * 0.5
    return torch.cos(half), torch.sin(half)


def _parameter_matrix(
    instruction: Instruction, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    name = instruction.name
    params = instruction.params
    if name in {"rx", "ry", "rz", "crx", "cry", "crz", "rxx", "ryy", "rzz"}:
        theta = _scalar(params["theta"], device=device)
        c, s = _rotation_pair(theta)
    if name == "rx":
        return _matrix((((c, 0), (0, -s)), ((0, -s), (c, 0))), device=device)
    if name == "ry":
        return _matrix((((c, 0), (-s, 0)), ((s, 0), (c, 0))), device=device)
    if name == "rz":
        return _matrix((((c, -s), (0, 0)), ((0, 0), (c, s))), device=device)
    if name in {"phase", "u1"}:
        theta = _scalar(params["theta"], device=device)
        return _matrix(
            (((1, 0), (0, 0)), ((0, 0), (torch.cos(theta), torch.sin(theta)))),
            device=device,
        )
    if name in {"u2", "u3"}:
        phi = _scalar(params["phi"], device=device)
        lbd = _scalar(params["lbd"], device=device)
        if name == "u2":
            c = s = _scalar(1.0 / math.sqrt(2.0), device=device)
        else:
            theta = _scalar(params["theta"], device=device)
            c, s = _rotation_pair(theta)
        return _matrix(
            (
                ((c, 0), (-s * torch.cos(lbd), -s * torch.sin(lbd))),
                (
                    (s * torch.cos(phi), s * torch.sin(phi)),
                    (c * torch.cos(phi + lbd), c * torch.sin(phi + lbd)),
                ),
            ),
            device=device,
        )
    if name in {"crx", "cry", "crz"}:
        base = Instruction(name=name[1:], wires=(0,), params={"theta": theta})
        block_real, block_imag = _parameter_matrix(base, device=device)
        entries = _zero_entries(4)
        entries[0][0] = entries[1][1] = (1, 0)
        for row in range(2):
            for column in range(2):
                entries[row + 2][column + 2] = (
                    block_real[row, column],
                    block_imag[row, column],
                )
        return _matrix(entries, device=device)
    if name == "cphase":
        theta = _scalar(params["theta"], device=device)
        entries = _zero_entries(4)
        for index in range(3):
            entries[index][index] = (1, 0)
        entries[3][3] = (torch.cos(theta), torch.sin(theta))
        return _matrix(entries, device=device)
    if name in {"rxx", "ryy", "rzz"}:
        entries = _zero_entries(4)
        for index in range(4):
            entries[index][index] = (c, 0)
        if name == "rxx":
            for row, column in ((0, 3), (1, 2), (2, 1), (3, 0)):
                entries[row][column] = (0, -s)
        elif name == "ryy":
            for row, column, sign in (
                (0, 3, 1),
                (1, 2, -1),
                (2, 1, -1),
                (3, 0, 1),
            ):
                entries[row][column] = (0, sign * s)
        else:
            for index, sign in enumerate((-1, 1, 1, -1)):
                entries[index][index] = (c, sign * s)
        return _matrix(entries, device=device)
    raise KeyError(name)


def _instruction_matrix_pair(
    instruction: Instruction, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if instruction.matrix is not None:
        raise NotImplementedError(
            "split real/imag P0 rejects custom matrices until their basis and "
            "device-resident conversion contract is certified"
        )
    if instruction.name not in SPLIT_REAL_IMAG_SUPPORTED_GATES:
        raise NotImplementedError(
            f"split real/imag P0 does not support gate {instruction.name!r}"
        )
    try:
        return _fixed_matrix(instruction.name, device=device)
    except KeyError:
        return _parameter_matrix(instruction, device=device)


def _apply_gate_pair(
    real: torch.Tensor,
    imag: torch.Tensor,
    matrix_real: torch.Tensor,
    matrix_imag: torch.Tensor,
    wires: Sequence[int],
    *,
    n_wires: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    wires = tuple(int(wire) for wire in wires)
    remaining = tuple(wire for wire in range(n_wires) if wire not in wires)
    permutation = remaining + wires
    inverse = tuple(permutation.index(wire) for wire in range(n_wires))
    gate_dimension = 2 ** len(wires)
    logical_shape = (2,) * n_wires
    values_real = (
        real.reshape(logical_shape).permute(permutation).reshape(-1, gate_dimension)
    )
    values_imag = (
        imag.reshape(logical_shape).permute(permutation).reshape(-1, gate_dimension)
    )
    updated_real = values_real @ matrix_real.transpose(
        0, 1
    ) - values_imag @ matrix_imag.transpose(0, 1)
    updated_imag = values_real @ matrix_imag.transpose(
        0, 1
    ) + values_imag @ matrix_real.transpose(0, 1)
    return (
        updated_real.reshape(logical_shape).permute(inverse).reshape(-1),
        updated_imag.reshape(logical_shape).permute(inverse).reshape(-1),
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
    resolved_device = resolve_platform_device(device)
    platform = get_platform_runtime(resolved_device.type)
    identity = platform.identity()
    if preflight:
        from ...operator_probes import preflight_split_real_imag_statevector_p0

        operator_report = preflight_split_real_imag_statevector_p0(
            device=resolved_device,
            provider=identity.provider,
        )
        operator_report.require_supported()
    else:
        from ...capabilities import load_operator_profile

        profile = load_operator_profile("split_real_imag_statevector_p0")
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


__all__ = (
    "SPLIT_REAL_IMAG_SCHEMA",
    "SPLIT_REAL_IMAG_SUPPORTED_GATES",
    "SplitRealImagConformanceCase",
    "SplitRealImagConformanceReport",
    "SplitRealImagStatevectorResult",
    "execute_split_real_imag_statevector",
    "run_split_real_imag_conformance",
)
