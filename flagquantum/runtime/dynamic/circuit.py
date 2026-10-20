"""Dynamic circuit construction implementation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch

from ...circuit import Circuit
from ...core.ir import Instruction
from ...errors import CapabilityError, ValidationError
from ._conditions import (
    validate_condition_clauses as _validate_condition_clauses,
)
from ._conditions import (
    validate_disjunctive_clauses as _validate_disjunctive_clauses,
)


def _disjunctive_clauses(
    any_of: Iterable[Mapping[int, int]],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Turn the clauses a caller wrote into the one spelling the IR stores."""

    clauses: list[tuple[tuple[int, int], ...]] = []
    for clause in any_of:
        if not isinstance(clause, Mapping):
            raise ValidationError(
                "any_of takes one mapping per clause, from a classical bit to the "
                "value that bit must hold"
            )
        if not clause:
            raise ValidationError("an any_of clause must test at least one bit")
        clauses.append(
            tuple(sorted((int(bit), int(value)) for bit, value in clause.items()))
        )
    return _validate_disjunctive_clauses(
        tuple(sorted(clauses, key=lambda item: (len(item), item)))
    )


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
        any_of: Iterable[Mapping[int, int]] | None = None,
        params: Mapping[str, Any] | None = None,
        matrix: Any | None = None,
    ) -> DynamicCircuit:
        """Append a gate that runs on the classical register's contents.

        One conjunction is stated with ``classical_bit`` or ``conditions``, and a
        disjunction of them with ``any_of``, which is the general form: any boolean
        function of the classical bits is a disjunction of conjunctions, so these
        two spellings reach every multi-bit condition rather than only the ones
        that happen to hold simultaneously.

        Args:
            name: Opcode of the gate to append.
            qubits: Qubits the gate acts on.
            classical_bit: One classical bit to test.
            equals: Value ``classical_bit`` must hold for the gate to run.
            conditions: One conjunction, from classical bit to required value.
            any_of: One mapping per clause, each mapping a conjunction the gate
                runs on. At least two clauses are required, because one clause is
                what ``conditions`` states.
            params: Gate parameters.
            matrix: Explicit matrix for an opcode no schema describes.

        Raises:
            ValidationError: If no condition is stated, if a clause tests a bit
                twice or against a value other than zero or one, if the clauses are
                not distinct and minimal, or if more than one spelling is used.

        Examples:
            Flip a data qubit unless both readouts agree:

            >>> from flagquantum.dynamic import DynamicCircuit
            >>> circuit = DynamicCircuit(3)
            >>> circuit = circuit.measure(0, classical_bit=0).measure(1, classical_bit=1)
            >>> _ = circuit.conditional("x", 2, any_of=({1: 0}, {0: 0}))
            >>> circuit.to_ir().instructions[-1].metadata["condition_clauses"]
            (((0, 0),), ((1, 0),))
        """

        if matrix is None and name in {"measure", "reset"}:
            raise ValidationError(
                f"a conditional {name} is not constructible: {name} projects a "
                "qubit and writes or discards a classical outcome rather than "
                "applying a matrix, so append it unconditionally and condition the "
                "gate that reads its bit"
            )
        if any_of is not None:
            if classical_bit is not None or conditions is not None:
                raise ValidationError(
                    "state the condition once: any_of replaces classical_bit and "
                    "conditions rather than combining with them"
                )
            metadata: Mapping[str, Any] = {
                "condition_clauses": _disjunctive_clauses(any_of)
            }
        else:
            if conditions is not None and classical_bit is not None:
                raise ValidationError(
                    "use either classical_bit or conditions, not both"
                )
            raw_conditions = (
                {int(classical_bit): int(equals)}
                if classical_bit is not None
                else {int(bit): int(value) for bit, value in (conditions or {}).items()}
            )
            if not raw_conditions:
                raise ValidationError(
                    "conditional gate requires at least one classical condition"
                )
            clause = tuple(sorted(raw_conditions.items()))
            _validate_condition_clauses((clause,))
            metadata = {"conditions": clause}
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
                metadata=metadata,
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
