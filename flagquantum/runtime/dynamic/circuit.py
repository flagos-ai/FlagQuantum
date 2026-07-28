"""Dynamic circuit construction implementation."""

from typing import Any, Iterable, Mapping

import torch

from ...circuit import Circuit
from ...core.ir import Instruction


class DynamicCircuit(Circuit):
    """Circuit builder for experimental mid-circuit measurement workflows."""

    def _append_dynamic(self, instruction: Instruction) -> "DynamicCircuit":
        if any(wire >= self.n_wires for wire in instruction.wires):
            raise ValueError("dynamic instruction wire is outside the circuit")
        self._instructions.append(instruction)
        self._state_cache = None
        self._ir_cache = None
        return self

    def measure(
        self, wire: int, *, classical_bit: int | None = None
    ) -> "DynamicCircuit":
        bit = int(wire if classical_bit is None else classical_bit)
        if bit < 0:
            raise ValueError("classical_bit must be non-negative")
        return self._append_dynamic(
            Instruction(
                "measure",
                (int(wire),),
                metadata={"is_dynamic": True, "classical_bit": bit},
            )
        )

    def reset(self, wire: int) -> "DynamicCircuit":
        return self._append_dynamic(
            Instruction("reset", (int(wire),), metadata={"is_dynamic": True})
        )

    def conditional(
        self,
        name: str,
        wires: Iterable[int] | int,
        *,
        classical_bit: int | None = None,
        equals: int = 1,
        conditions: Mapping[int, int] | None = None,
        params: Mapping[str, Any] | None = None,
        matrix: Any | None = None,
    ) -> "DynamicCircuit":
        if conditions is not None and classical_bit is not None:
            raise ValueError("use either classical_bit or conditions, not both")
        raw_conditions = (
            {int(classical_bit): int(equals)}
            if classical_bit is not None
            else {int(bit): int(value) for bit, value in (conditions or {}).items()}
        )
        if not raw_conditions:
            raise ValueError(
                "conditional gate requires at least one classical condition"
            )
        if any(bit < 0 or value not in {0, 1} for bit, value in raw_conditions.items()):
            raise ValueError("conditions require non-negative bits and values 0 or 1")
        wire_tuple = (int(wires),) if isinstance(wires, int) else tuple(wires)
        return self._append_dynamic(
            Instruction(
                name,
                wire_tuple,
                params=dict(params or {}),
                matrix=matrix,
                metadata={"conditions": tuple(sorted(raw_conditions.items()))},
            )
        )

    def state(
        self, parameter_bindings: Mapping[Any, Any] | None = None
    ) -> torch.Tensor:
        if any(
            instruction.metadata.get("is_dynamic")
            or instruction.metadata.get("condition")
            for instruction in self._instructions
        ):
            raise RuntimeError(
                "dynamic circuits require fq.experimental.run_dynamic(..., shots=...)"
            )
        return super().state(parameter_bindings)


__all__ = ("DynamicCircuit",)
