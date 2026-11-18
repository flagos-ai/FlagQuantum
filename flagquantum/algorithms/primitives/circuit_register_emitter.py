"""Bind owned directional circuits to existing named oracle callbacks."""

from collections.abc import Mapping
from dataclasses import dataclass

from ...compiler.registers.layout.partition import (
    RegisterLayout,
    partition_width,
    unpartition_register,
)
from ...core.ir import CircuitIR, Instruction
from ._circuit_snapshot import _emit, _storage


@dataclass(frozen=True)
class CircuitRegisterEmitter:
    """Implement PREPARE and SELECT callbacks without retaining an external SDK.

    Each snapshot packs the named logical ports, followed by clean workspace.
    Array ports are flattened in row-major order, with big-endian word bits.
    Both directions are supplied explicitly: measured circuits are never reversed.
    Binding swaps are physical instructions already included in the snapshots;
    their counts are evidence, not an independent source resource model.
    """

    layout: RegisterLayout
    snapshots: tuple[CircuitIR, CircuitIR]
    binding_swaps: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        # Reuse the layout owner's exact name, width and uniqueness checks.
        width = partition_width(self.layout)
        if type(width) is not int:
            raise ValueError("Register emitters require concrete nonnegative widths")
        if len(self.snapshots) != 2 or len(self.binding_swaps) != 2:
            raise ValueError("Register emitters require both directional snapshots")
        if any(type(n) is not int or n < 0 for n in self.binding_swaps):
            raise ValueError(
                "Register binding swap counts must be nonnegative integers"
            )
        _storage(self.snapshots, width)

    def requirements(self) -> tuple[int, int]:
        """Return reusable clean work and classical result storage for either direction."""
        return _storage(self.snapshots, sum(width for _, width in self.layout))

    def __call__(
        self,
        ports: Mapping[str, tuple[int, ...]],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        """Bind the existing named PREPARE/SELECT emitter interface."""
        if type(adjoint) is not bool:
            raise ValueError("Register emitter direction must be Boolean")
        return _emit(
            self.snapshots[int(adjoint)],
            unpartition_register(ports, self.layout),
            workspace,
            classical_bits,
            self.requirements(),
            adjoint,
        )
