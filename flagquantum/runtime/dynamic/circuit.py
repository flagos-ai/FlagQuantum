"""Dynamic circuit construction implementation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch

from ...circuit import Circuit
from ...core.ir import Instruction
from ...errors import CapabilityError, ValidationError


class DynamicCircuit(Circuit):
    """Circuit builder for mid-circuit measurement and classical feedback."""

    def _append_dynamic(self, instruction: Instruction) -> DynamicCircuit:
        if any(qubit < 0 or qubit >= self.n_qubits for qubit in instruction.wires):
            raise ValidationError("dynamic instruction qubit is outside the circuit")
        self._instructions.append(instruction)
        self._invalidate_execution_cache(instructions_changed=True)
        return self

    def measure(
        self, qubit: int, *, classical_bit: int | None = None
    ) -> DynamicCircuit:
        bit = int(qubit if classical_bit is None else classical_bit)
        if bit < 0:
            raise ValidationError("classical_bit must be non-negative")
        return self._append_dynamic(
            Instruction(
                "measure",
                (int(qubit),),
                metadata={"is_dynamic": True, "classical_bit": bit},
            )
        )

    def reset(self, qubit: int) -> DynamicCircuit:
        return self._append_dynamic(
            Instruction("reset", (int(qubit),), metadata={"is_dynamic": True})
        )

    def conditional(
        self,
        name: str,
        qubits: Iterable[int] | int,
        *,
        classical_bit: int | None = None,
        equals: int = 1,
        conditions: Mapping[int, int] | None = None,
        params: Mapping[str, Any] | None = None,
        matrix: Any | None = None,
    ) -> DynamicCircuit:
        if conditions is not None and classical_bit is not None:
            raise ValidationError("use either classical_bit or conditions, not both")
        raw_conditions = (
            {int(classical_bit): int(equals)}
            if classical_bit is not None
            else {int(bit): int(value) for bit, value in (conditions or {}).items()}
        )
        if not raw_conditions:
            raise ValidationError(
                "conditional gate requires at least one classical condition"
            )
        if any(bit < 0 or value not in {0, 1} for bit, value in raw_conditions.items()):
            raise ValidationError(
                "conditions require non-negative bits and values 0 or 1"
            )
        qubit_tuple = (
            (int(qubits),)
            if isinstance(qubits, int)
            else tuple(int(qubit) for qubit in qubits)
        )
        return self._append_dynamic(
            Instruction(
                name,
                qubit_tuple,
                params=dict(params or {}),
                matrix=matrix,
                metadata={"conditions": tuple(sorted(raw_conditions.items()))},
            )
        )

    def state(self, *, refresh: bool = False) -> torch.Tensor:
        if any(
            instruction.metadata.get("is_dynamic")
            or instruction.metadata.get("conditions")
            or instruction.metadata.get("condition_clauses")
            for instruction in self._instructions
        ):
            raise CapabilityError(
                "dynamic circuits require fq.experimental.dynamic.run_dynamic(..., shots=...)"
            )
        return super().state(refresh=refresh)


__all__ = ("DynamicCircuit",)
