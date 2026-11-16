# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 BloqAsCirqGate.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/cirq_interop/_bloq_to_cirq.py

"""Expose a bound native register body through Cirq's optional gate protocols."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from ...compiler.registers.primitives.power import register_power_ir
from ...compiler.registers.register_binding import BoundRegisterProgram
from ...core.ir import CircuitIR, IRSerializationError
from ...core.ir.program.registers import Register, RegisterSide
from ...core.ir.program.signatures import Signature
from ._gate_metadata import cirq_wire_label
from ._register_export import _register_shape
from ._register_workspace import bind_workspace_operations
from .conversion import _cirq, _export_cirq
from .models import CirqConversionError

if TYPE_CHECKING:
    from cirq import CircuitDiagramInfo, DecompositionContext, Gate, Qid

    from ...drawer.register_score_data import SymbolClass

    _WireSymbol = Callable[
        [Register, tuple[int, ...]], tuple[SymbolClass, dict[str, str | int | bool]]
    ]
else:
    Gate = _cirq().Gate


class RegisterCirqGate(Gate):
    """Adapt an owned body over ordered register bits and optional private leases."""

    def __init__(
        self,
        body: CircuitIR | BoundRegisterProgram,
        signature: Signature,
        *,
        wire_symbol: _WireSymbol | None = None,
        feedback_key_prefix: str | None = None,
        guarded_measurement_work: int | None = None,
    ) -> None:
        labels: list[str] = []
        input_indices: list[int] = []
        right_indices: list[int] = []
        output_groups: list[tuple[int, ...]] = []
        names: set[str] = set()
        for register in signature:
            if register.name in names:
                raise ValueError("Cirq gate register names must be distinct")
            names.add(register.name)
            shape = _register_shape(register)
            width = shape[-1]
            first_bit = len(labels)
            for element_index in register.all_idxs():
                bits = range(len(labels), len(labels) + width)
                if register.side & RegisterSide.LEFT:
                    input_indices.extend(bits)
                if register.side & RegisterSide.RIGHT:
                    right_indices.extend(bits)
                if wire_symbol is None:
                    label = register.name
                    if element_index:
                        label += f"[{', '.join(map(str, element_index))}]"
                else:
                    label = cirq_wire_label(*wire_symbol(register, element_index))
                labels.extend([label] * width)
            if register.side == RegisterSide.RIGHT:
                output_groups.append(tuple(range(first_bit, len(labels))))
        bound = body if isinstance(body, BoundRegisterProgram) else None
        ir = body if isinstance(body, CircuitIR) else body.circuit
        if bound is None and len(labels) != ir.n_wires:
            raise ValueError(
                "Cirq gate requires one declared operand per physical wire"
            )
        output_wires: dict[int, int] = {}
        consumed_wires: frozenset[int] = frozenset()
        if bound is not None:
            outputs = tuple(wire for word in bound.outputs for wire in word)
            if (
                bound.classical_outputs
                or len(outputs) != len(right_indices)
                or len(input_indices) > ir.n_wires
                or len(set(outputs)) != len(outputs)
                or any(not 0 <= wire < ir.n_wires for wire in outputs)
            ):
                raise ValueError(
                    "Cirq workspace gate ports do not match the bound body"
                )
            input_wires = {index: wire for wire, index in enumerate(input_indices)}
            consumed_wires = frozenset(
                input_wires[index] for index in input_wires.keys() - set(right_indices)
            )
            for wire, index in zip(outputs, right_indices, strict=True):
                if index in input_wires:
                    if input_wires[index] != wire:
                        raise ValueError(
                            "Cirq workspace gates require unchanged through wires"
                        )
                elif wire < len(input_indices) and wire not in consumed_wires:
                    raise ValueError(
                        "Cirq right-only output cannot reuse a through wire"
                    )
                else:
                    output_wires[index] = wire
        if guarded_measurement_work is not None:
            if (
                bound is None
                or type(guarded_measurement_work) is not int
                or guarded_measurement_work < len(input_indices)
                or guarded_measurement_work in output_wires.values()
            ):
                raise ValueError("Cirq guarded measurement requires private bound work")
            leases = [
                event
                for event in bound.events
                if event.kind == "allocate" and guarded_measurement_work in event.wires
            ]
            if not leases or any(event.dirty for event in leases):
                raise ValueError("Cirq guarded measurement requires a clean work lease")
        body_key: str | None
        try:
            frozen_body = CircuitIR.from_dict(ir.to_dict())
            body_key = frozen_body.content_hash
        except IRSerializationError:
            frozen_body = ir
            body_key = None
        self._body = frozen_body
        self._bound = bound
        self._feedback_key_prefix = feedback_key_prefix
        self._guarded_measurement_work = guarded_measurement_work
        self._signature = signature
        self._wire_symbol = wire_symbol
        self._labels = tuple(labels)
        self._input_indices = tuple(input_indices)
        self._consumed_wires = consumed_wires
        self._output_groups = (
            tuple(
                tuple((output_wires[index], index) for index in group)
                for group in output_groups
            )
            if bound is not None
            else ()
        )
        self._value_key: tuple[object, ...] | None = None
        if body_key is not None:
            workspace = (
                None
                if bound is None
                else (
                    bound.outputs,
                    tuple(
                        (event.kind, event.wires, event.dirty, event.instruction_offset)
                        for event in bound.events
                    ),
                )
            )
            self._value_key = (
                body_key,
                tuple(signature),
                self._labels,
                workspace,
                feedback_key_prefix,
                guarded_measurement_work,
            )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        return (
            isinstance(other, RegisterCirqGate)
            and self._value_key is not None
            and self._value_key == other._value_key
        )

    def __hash__(self) -> int:
        if self._value_key is None:
            return object.__hash__(self)
        return hash(self._value_key)

    def _num_qubits_(self) -> int:
        return len(self._labels)

    def _decompose_(self, qubits: Sequence[Qid]) -> object:
        return self._decompose_with_context_(qubits)

    def _decompose_with_context_(
        self, qubits: Sequence[Qid], context: DecompositionContext | None = None
    ) -> object:
        if (
            self._bound is not None
            and self._feedback_key_prefix is None
            and any(gate.name == "measure" for gate in self._body.instructions)
        ):
            return NotImplemented
        offsets: list[int] | None = (
            []
            if self._bound is not None and self._feedback_key_prefix is not None
            else None
        )
        try:
            result = _export_cirq(
                self._body,
                allow_lossy=False,
                include_inverse_phases=True,
                extent_declared_by_gate=True,
                feedback_key_prefix=self._feedback_key_prefix,
                guarded_measurement_work=self._guarded_measurement_work,
                instruction_offsets=offsets,
            )
        except CirqConversionError:
            return NotImplemented
        if self._bound is not None:
            operations = tuple(result.circuit.all_operations())
            if offsets is None and len(operations) != len(self._body.instructions):
                return NotImplemented
            return bind_workspace_operations(
                operations,
                self._bound.events,
                tuple(qubits[index] for index in self._input_indices),
                self._consumed_wires,
                tuple(
                    {wire: qubits[index] for wire, index in group}
                    for group in self._output_groups
                ),
                context.qubit_manager if context else None,
                instruction_offsets=offsets,
            )
        remapped = result.circuit.transform_qubits(lambda q: qubits[q.x])
        return tuple(remapped.all_operations())

    def _unitary_(self) -> object:
        if not self._signature.thru_registers_only:
            return NotImplemented
        import torch

        from ...simulation.unitary import _unitary_with_workspace, get_unitary

        try:
            if self._bound is None:
                matrix = get_unitary(self._body, device="cpu", dtype=torch.complex128)
            else:
                borrowed = frozenset(
                    wire
                    for event in self._bound.events
                    if event.kind == "allocate" and event.dirty
                    for wire in event.wires
                )
                matrix = _unitary_with_workspace(
                    self._body, len(self._labels), borrowed
                )
        except (TypeError, ValueError, NotImplementedError):
            if self._bound is not None:
                return None
            return NotImplemented
        return np.asarray(matrix.detach().numpy(), dtype=np.complex128)

    def _circuit_diagram_info_(self, args: object) -> CircuitDiagramInfo:
        info: CircuitDiagramInfo = _cirq().CircuitDiagramInfo(self._labels)
        return info

    def __pow__(self, power: object) -> RegisterCirqGate:
        if type(power) is not int or power == 0:
            raise ValueError("Cirq gate power requires a positive integer magnitude")
        if power == 1:
            return self
        if self._feedback_key_prefix is not None:
            raise ValueError("Cirq feedback gate power requires fresh measurement keys")
        if self._bound is not None:
            raise ValueError(
                "Cirq workspace gate power requires rebinding its lifetimes"
            )
        if power != -1 and not self._signature.thru_registers_only:
            raise ValueError("Repeated Cirq gate power requires through registers")
        instructions = register_power_ir(self._body, abs(power), adjoint=power < 0)
        signature = self._signature if power > 0 else self._signature.adjoint()
        return RegisterCirqGate(
            replace(self._body, instructions=instructions),
            signature,
            wire_symbol=self._wire_symbol,
        )
