"""Fixed four-sigma Ising source construction over the existing ProgramRecord."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import torch

from .ir import IRValidationError
from .ir.program.model import BlockRecord, FunctionRecord, Operation, ProgramRecord
from .ir.program.physical import IsingContext, IsingExchange, check_ising_exchange
from .ir.program.verifier import verify_program_record


def _channel(value: str) -> Literal["1", "psi"]:
    if not isinstance(value, str):
        raise TypeError("fusion channel must be a string")
    if value not in ("1", "psi"):
        raise ValueError("fusion channel must be '1' or 'psi'")
    return "1" if value == "1" else "psi"


class IsingBraidProgram:
    """Ordered adjacent exchanges of four sigma anyons with total vacuum charge.

    The fixed left-associated fusion basis is not a tensor product of qubits.
    Exchanges record discrete source order, not physical time or laser controls.

    Examples:
        >>> from flagquantum.experimental.simulation import IsingBraidProgram
        >>> program = IsingBraidProgram().exchange(1).exchange(0, clockwise=True)
        >>> program.basis_labels
        (('1', 'sigma', '1'), ('psi', 'sigma', '1'))
    """

    def __init__(self, *, initial_channel: Literal["1", "psi"] = "1") -> None:
        context = IsingContext(
            ("a", "b", "c", "d"),
            "1",
            "left_associated",
            "nayak_ising",
            _channel(initial_channel),
        )
        self._record = ProgramRecord(
            (FunctionRecord("main", (BlockRecord("entry"),)),),
            physical_context=context,
        )

    @property
    def basis_labels(self) -> tuple[tuple[str, ...], ...]:
        return (("1", "sigma", "1"), ("psi", "sigma", "1"))

    def exchange(self, left: int, *, clockwise: bool = False) -> IsingBraidProgram:
        """Exchange positions left and left+1; counterclockwise is the default.

        Raises:
            TypeError: The index is not an int or direction is not a bool.
            ValueError: The index is outside the three adjacent pairs.
        """
        data = IsingExchange(left, clockwise)
        check_ising_exchange(data)
        return self._append(Operation("ising.exchange", physical_data=data))

    def dephase_pair(self) -> IsingBraidProgram:
        """Append an ideal unread charge observation of the current first pair.

        No outcome is selected or returned. This records a channel in source
        order, without adding physical time, classical bits or feedback.

        Raises:
            ValueError: An unread or sampled observation is already present.

        Examples:
            >>> from flagquantum.experimental.simulation import IsingBraidProgram
            >>> program = IsingBraidProgram().exchange(1).dephase_pair()
            >>> program.to_ir().functions[0].blocks[0].operations[-1].opcode
            'ising.dephase_pair'
        """
        return self._append_pair_observation("ising.dephase_pair")

    def measure_pair(self) -> IsingBraidProgram:
        """Record one sampled QND charge observation of the current first pair.

        This source declaration returns the program, not an outcome or a
        classical bit. Runtime owns explicit shot/seed admission and execution.

        Raises:
            ValueError: An unread or sampled observation is already present.

        Examples:
            >>> from flagquantum.experimental.simulation import IsingBraidProgram
            >>> program = IsingBraidProgram().exchange(1).measure_pair()
            >>> program.to_ir().functions[0].blocks[0].operations[-1].opcode
            'ising.measure_pair'
        """
        return self._append_pair_observation("ising.measure_pair")

    def _append_pair_observation(
        self, opcode: Literal["ising.dephase_pair", "ising.measure_pair"]
    ) -> IsingBraidProgram:
        if any(
            operation.opcode in ("ising.dephase_pair", "ising.measure_pair")
            for operation in self._record.functions[0].blocks[0].operations
        ):
            kind = "unread" if opcode == "ising.dephase_pair" else "sampled"
            raise ValueError(f"only one {kind} pair observation is admitted")
        return self._append(Operation(opcode))

    def _append(self, operation: Operation) -> IsingBraidProgram:
        function = self._record.functions[0]
        block = function.blocks[0]
        self._record = replace(
            self._record,
            functions=(
                replace(
                    function,
                    blocks=(
                        replace(
                            block,
                            operations=(
                                *block.operations,
                                operation,
                            ),
                        ),
                    ),
                ),
            ),
        )
        return self

    def projector(self, channel: Literal["1", "psi"]) -> torch.Tensor:
        """Return a fresh CPU complex128 charge projector for the current first pair.

        This names positions in the declared fusion tree, not initial anyon IDs.
        """
        selected = _channel(channel)
        return torch.diag(
            torch.tensor(
                [selected == "1", selected == "psi"],
                dtype=torch.complex128,
                device="cpu",
            )
        )

    def to_ir(self) -> ProgramRecord:
        """Return the immutable, verified source snapshot without circuit lowering."""
        diagnostics = verify_program_record(self._record)
        if diagnostics:
            raise IRValidationError(
                "; ".join(f"{item.code}: {item.message}" for item in diagnostics)
            )
        return self._record
