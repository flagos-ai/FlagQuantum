"""Managed device-simulator workflow for service-side target routing.

This module deliberately does not change the client-side remote execution
contract.  A provider service can call this workflow after it has resolved a
``quafu:<device>-sim`` task and fetched the calibration that belongs to the
named physical device.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..noise import NoiseModel
    from ..runtime.result import ExecutionResult


_QUAFU_MANAGED_SIMULATORS = {
    "quafu:Baihua-sim": "Baihua",
    "quafu:Shenglian-sim": "Shenglian",
    "quafu:Dongling-sim": "Dongling",
}


@dataclass(frozen=True, slots=True)
class ManagedSimulatorReceipt:
    """Auditable binding between one simulator run and device calibration."""

    target: str
    calibration_device: str
    calibration_source: str
    calibration_captured_at: str
    noise_model_identity: str
    execution_representation: str
    approximate: bool
    schema: str = "flagquantum.managed_simulator_receipt.v1"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ManagedSimulatorExecution:
    """Execution result plus the calibration and routing evidence used."""

    result: ExecutionResult
    receipt: ManagedSimulatorReceipt


def managed_quafu_simulator_device(target: str) -> str:
    """Return the physical-device name represented by a managed target."""

    try:
        return _QUAFU_MANAGED_SIMULATORS[target]
    except KeyError as error:
        supported = ", ".join(_QUAFU_MANAGED_SIMULATORS)
        raise ValueError(
            f"unsupported managed Quafu simulator target {target!r}; "
            f"expected one of: {supported}"
        ) from error


def run_managed_quafu_simulator(
    program: Any,
    *,
    target: str,
    calibration_device: str,
    noise_model: NoiseModel,
    shots: int,
    memory_limit_bytes: int,
    seed: int | None = None,
) -> ManagedSimulatorExecution:
    """Run a Quafu managed simulator using a device-bound noise snapshot.

    Small workloads use exact density-matrix evolution.  When that state does
    not fit ``memory_limit_bytes``, the stable planner may select CPU noisy MPS;
    that approximation is explicit in the returned receipt.
    """

    from .._api import run
    from ..core.ir import ensure_circuit_ir
    from ..noise import NoiseModel as NoiseModelType
    from ..observables import counts
    from ..runtime.execution_plan import ExecutionPlan
    from ..runtime.options import ExecutionOptions

    expected_device = managed_quafu_simulator_device(target)
    if calibration_device != expected_device:
        raise ValueError(
            f"target {target!r} requires {expected_device!r} calibration, "
            f"not {calibration_device!r}"
        )
    if not isinstance(noise_model, NoiseModelType):
        raise TypeError("noise_model must be a NoiseModel")
    profile = noise_model.device_profile
    if profile is None:
        raise ValueError("managed simulator requires a device noise profile")

    ir = ensure_circuit_ir(program)
    calibrated_wires = {item.wire for item in profile.qubits}
    missing_wires = sorted(set(range(ir.n_wires)) - calibrated_wires)
    if missing_wires:
        raise ValueError(
            "device noise profile does not cover logical wire(s): "
            + ", ".join(map(str, missing_wires))
        )

    result = run(
        program,
        options=ExecutionOptions(
            mode="auto",
            device="cpu",
            shots=shots,
            seed=seed,
            memory_limit_bytes=memory_limit_bytes,
            allow_approximate=True,
        ),
        outputs=counts(),
        noise_model=noise_model,
    )
    plan = result.plan
    if not isinstance(plan, ExecutionPlan):
        raise RuntimeError(
            "managed simulator execution did not return an execution plan"
        )
    noisy_plan = plan.noisy_execution_plan
    representation = (
        noisy_plan.representation if noisy_plan is not None else plan.state_mode
    )
    approximate = representation == "mps"
    return ManagedSimulatorExecution(
        result=result,
        receipt=ManagedSimulatorReceipt(
            target=target,
            calibration_device=calibration_device,
            calibration_source=profile.source,
            calibration_captured_at=profile.captured_at,
            noise_model_identity=noise_model.identity,
            execution_representation=representation,
            approximate=approximate,
        ),
    )


__all__ = (
    "ManagedSimulatorExecution",
    "ManagedSimulatorReceipt",
    "managed_quafu_simulator_device",
    "run_managed_quafu_simulator",
)
