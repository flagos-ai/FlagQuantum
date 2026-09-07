"""Canonical reusable compiled MPS training executors."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from ..simulation.mps.entrypoints import run_mps

CircuitBuilder = Callable[[torch.Tensor], Any]
LossBuilder = Callable[[Any], torch.Tensor]


@dataclass
class OwnerShardedParameterSynchronizer:
    """Synchronize owner-updated scalar parameters with one packed collective.

    Parameter tensor identities are preserved so autograd references and
    owner-local optimizer state remain valid. Each rank contributes only the
    values it owns; an all-reduce reconstructs the complete parameter vector.
    """

    parameters: tuple[torch.Tensor, ...]
    owners: tuple[int, ...]
    rank: int
    world_size: int
    process_group: Any = None

    def __post_init__(self) -> None:
        if not self.parameters:
            raise ValueError("at least one parameter is required")
        if len(self.parameters) != len(self.owners):
            raise ValueError("parameters and owners must have equal length")
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be inside a positive world size")
        if any(not 0 <= owner < self.world_size for owner in self.owners):
            raise ValueError("every parameter owner must be a valid rank")
        reference = self.parameters[0]
        if any(
            parameter.numel() != 1
            or parameter.device != reference.device
            or parameter.dtype != reference.dtype
            for parameter in self.parameters
        ):
            raise ValueError(
                "packed owner synchronization requires scalar parameters with "
                "one device and dtype"
            )
        self._owner_mask = torch.tensor(
            [owner == self.rank for owner in self.owners],
            dtype=reference.dtype,
            device=reference.device,
        )

    def synchronize(self) -> dict[str, int | str]:
        if self.world_size == 1:
            return {
                "strategy": "single_rank_noop",
                "collective_count": 0,
                "logical_payload_bytes": 0,
            }
        packed = torch.stack(
            tuple(parameter.detach().reshape(()) for parameter in self.parameters)
        )
        packed.mul_(self._owner_mask)
        dist.all_reduce(packed, group=self.process_group)
        with torch.no_grad():
            torch._foreach_copy_(list(self.parameters), list(packed.unbind()))
        return {
            "strategy": "owner_masked_packed_all_reduce",
            "collective_count": 1,
            "logical_payload_bytes": packed.numel() * packed.element_size(),
        }


@dataclass
class MPSTrainingStep:
    """Callable value-and-gradient executor for fixed-topology MPS training.

    The circuit topology is supplied by ``circuit_builder`` while trainable
    values are supplied as the tensor argument to ``__call__``. ``torch.compile``
    therefore sees a stable graph when parameter shape and circuit structure
    remain fixed, while optimizer-updated parameter values continue to flow as
    normal tensor inputs.
    """

    circuit_builder: CircuitBuilder
    mode: str = "mps"
    observable: str | LossBuilder = "z_sum"
    observable_wires: tuple[int, ...] | None = None
    mps_options: dict[str, Any] = field(default_factory=dict)
    compile: bool = False
    compile_backend: str | None = None
    compile_mode: str | None = "reduce-overhead"
    fullgraph: bool = False
    fallback: bool = True
    status: str = "eager"
    compile_error: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"mps", "local_mps"}:
            raise ValueError(
                "MPSTrainingStep currently supports mode='mps' or 'local_mps'."
            )
        self._compiled_value_and_grad = None
        if self.compile:
            self._try_compile()

    def __call__(self, parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.value_and_grad(parameters)

    def value_and_grad(
        self, parameters: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._compiled_value_and_grad is not None:
            try:
                loss, grad = self._compiled_value_and_grad(parameters)
                self.status = "compiled"
                return loss, grad
            except (
                Exception
            ) as exc:  # noqa: BLE001 - compiled execution must be recoverable.
                self.compile_error = f"{type(exc).__name__}: {exc}"
                if not self.fallback:
                    raise
                self._compiled_value_and_grad = None
                self.status = "compile_failed_fallback"
        loss, grad = self._eager_value_and_grad(parameters)
        if self.status == "compiled":
            self.status = "eager"
        return loss, grad

    def loss(self, parameters: torch.Tensor) -> torch.Tensor:
        circuit = self.circuit_builder(parameters)
        result = run_mps(circuit, **self.mps_options)
        if callable(self.observable):
            return self.observable(result)
        if self.observable == "z_sum":
            return result.expectation_z_sum(self.observable_wires).sum()
        if self.observable == "z":
            return result.expectation_z(self.observable_wires).sum()
        raise ValueError("observable must be 'z_sum', 'z', or a callable.")

    def summary(self) -> dict[str, Any]:
        return {
            "executor": "mps_training_step",
            "mode": self.mode,
            "status": self.status,
            "compile": self.compile,
            "compile_backend": self.compile_backend,
            "compile_mode": self.compile_mode,
            "fullgraph": self.fullgraph,
            "fallback": self.fallback,
            "compile_error": self.compile_error,
            "mps_options": dict(self.mps_options),
        }

    def _eager_value_and_grad(
        self, parameters: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            work = parameters.detach().clone().requires_grad_(True)
            loss = self.loss(work)
            grad = torch.autograd.grad(loss, work)[0]
        return loss.detach(), grad.detach()

    def _try_compile(self) -> None:
        if not hasattr(torch, "compile"):
            self.status = (
                "compile_unavailable_fallback"
                if self.fallback
                else "compile_unavailable"
            )
            self.compile_error = "torch.compile is not available in this PyTorch build."
            if not self.fallback:
                raise RuntimeError(self.compile_error)
            return
        try:
            kwargs: dict[str, Any] = {"fullgraph": self.fullgraph}
            if self.compile_backend:
                kwargs["backend"] = self.compile_backend
            if self.compile_mode:
                kwargs["mode"] = self.compile_mode
            self._compiled_value_and_grad = torch.compile(
                self._eager_value_and_grad, **kwargs
            )
            self.status = "compile_ready"
        except (
            Exception
        ) as exc:  # noqa: BLE001 - optional compiler should not break eager training.
            self.compile_error = f"{type(exc).__name__}: {exc}"
            self.status = (
                "compile_failed_fallback" if self.fallback else "compile_failed"
            )
            if not self.fallback:
                raise


def compile_mps_training_step(
    circuit_builder: CircuitBuilder,
    example_parameters: torch.Tensor | None = None,
    *,
    mode: str = "mps",
    observable: str | LossBuilder = "z_sum",
    observable_wires: Iterable[int] | int | None = None,
    compile: bool = True,
    compile_backend: str | None = None,
    compile_mode: str | None = "reduce-overhead",
    fullgraph: bool = False,
    fallback: bool = True,
    **mps_options: Any,
) -> MPSTrainingStep:
    """Create a reusable MPS value-and-gradient training step.

    ``example_parameters`` is accepted to make call sites self-documenting and
    future-proof for ahead-of-time warmup; parameter values are never captured as
    constants and can change every optimizer step.
    """

    del example_parameters
    if isinstance(observable_wires, int):
        wires = (int(observable_wires),)
    elif observable_wires is None:
        wires = None
    else:
        wires = tuple(int(wire) for wire in observable_wires)
    return MPSTrainingStep(
        circuit_builder=circuit_builder,
        mode=mode,
        observable=observable,
        observable_wires=wires,
        mps_options=dict(mps_options),
        compile=compile,
        compile_backend=compile_backend,
        compile_mode=compile_mode,
        fullgraph=fullgraph,
        fallback=fallback,
    )


__all__ = [
    "MPSTrainingStep",
    "OwnerShardedParameterSynchronizer",
    "compile_mps_training_step",
]
