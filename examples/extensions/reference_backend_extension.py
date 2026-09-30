"""A third-party execution backend implemented without importing internals.

This file is the ten-minute golden path for an extension-provided backend. It
imports only the public extension namespace, as a real third-party backend does,
and it owns its own numerics: Runtime routes a program to it and never evaluates
the circuit on its behalf.

The narrow gate set below is deliberate. A real backend declares what it supports
and fails closed on anything else; this reference route does the same so that an
unsupported program produces an explicit error rather than a wrong amplitude.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar

import torch

from flagquantum.ecosystem.extensions import (
    CapabilityRequest,
    CapabilityResponse,
    ExtensionConfig,
    ExtensionManifest,
)

_SUPPORTED_GATES = frozenset({"h", "x", "cx", "rz", "rx", "ry"})


def _one_qubit_matrix(name: str, params: Mapping[str, Any]) -> torch.Tensor:
    """Return the 2x2 matrix of one declared gate, in FlagQuantum's convention."""

    if name == "h":
        return torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / math.sqrt(2)
    if name in {"x", "cx"}:
        return torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    angle = float(params.get("theta", params.get("phi", 0.0)))
    half = angle / 2
    cosine, sine = math.cos(half), math.sin(half)
    if name == "rz":
        return torch.tensor(
            [[complex(cosine, -sine), 0j], [0j, complex(cosine, sine)]],
            dtype=torch.complex64,
        )
    if name == "rx":
        return torch.tensor(
            [[cosine, complex(0, -sine)], [complex(0, -sine), cosine]],
            dtype=torch.complex64,
        )
    return torch.tensor([[cosine, -sine], [sine, cosine]], dtype=torch.complex64)


class ReferenceStatevectorBackend:
    """Exact statevector evaluation of a declared, narrow gate set."""

    manifest = ExtensionManifest(
        name="reference_statevector",
        version="1.0.0",
        kind="backend",
        capabilities=frozenset({"complex64", "cpu", "statevector"}),
    )

    # Host-owned keys are absent on purpose: admission owns the name and the
    # route, and the host probes accelerators itself.
    backend_capabilities: ClassVar[dict[str, Any]] = {
        "tensor_backend": "torch",
        "devices": ("cpu",),
        "dtypes": ("complex64",),
        "supports_autograd": False,
        "supports_distributed": False,
        "supports_statevector": True,
        "supports_density_matrix": False,
        "supports_mps": False,
    }

    def __init__(self) -> None:
        self.active = False
        self.executions = 0

    def negotiate(self, request: CapabilityRequest) -> CapabilityResponse:
        blockers = []
        if request.dtype not in (None, "complex64"):
            blockers.append(f"dtype {request.dtype!r} is unsupported")
        if request.device_type not in (None, "cpu"):
            blockers.append(f"device {request.device_type!r} is unsupported")
        return CapabilityResponse(
            not blockers, self.manifest.capabilities, tuple(blockers)
        )

    def start(self, config: ExtensionConfig) -> None:
        self.active = True

    def close(self) -> None:
        self.active = False

    def execute(self, program: Any, *, options: Mapping[str, Any]) -> torch.Tensor:
        """Evaluate ``program`` and return one amplitude row per batch element."""

        if not self.active:
            raise RuntimeError("backend is closed")
        unsupported = sorted(
            {instruction.name for instruction in program} - _SUPPORTED_GATES
        )
        if unsupported:
            raise ValueError(
                "reference_statevector supports a narrow gate set; unsupported "
                "gates: " + ", ".join(unsupported)
            )
        state = torch.zeros(
            (int(options["batch_size"]), 2**program.n_wires), dtype=torch.complex64
        )
        state[:, 0] = 1.0
        for instruction in program:
            state = _apply(state, instruction, program.n_wires)
        self.executions += 1
        return state


def _apply(state: torch.Tensor, instruction: Any, n_wires: int) -> torch.Tensor:
    wires = tuple(instruction.wires)
    matrix = _one_qubit_matrix(instruction.name, instruction.params)
    if len(wires) == 1:
        return _apply_one(state, matrix, wires[0], n_wires)
    control, target = wires
    columns = torch.arange(state.shape[1])
    # FlagQuantum numbers wire 0 as the most significant amplitude bit, so wire
    # ``w`` selects bit ``n_wires - 1 - w`` of the amplitude column index.
    mask = (columns >> (n_wires - 1 - control)) & 1 == 1
    flipped = _apply_one(state, matrix, target, n_wires)
    return torch.where(mask, flipped, state)


def _apply_one(
    state: torch.Tensor, matrix: torch.Tensor, wire: int, n_wires: int
) -> torch.Tensor:
    # The same ordering puts wire ``w`` on axis ``w + 1`` of the
    # ``2 x ... x 2`` amplitude view, whose axis 1 is the most significant bit.
    axis = wire + 1
    reshaped = torch.movedim(state.reshape(state.shape[0], *([2] * n_wires)), axis, -1)
    leading = reshaped.shape[:-1]
    flat = reshaped.reshape(-1, 2)
    updated = (flat.unsqueeze(1) @ matrix.T.unsqueeze(0)).squeeze(1)
    return torch.movedim(updated.reshape(*leading, 2), -1, axis).reshape(state.shape)


__all__ = ("ReferenceStatevectorBackend",)
