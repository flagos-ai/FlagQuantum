# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 BloqAsCirqGate.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/cirq_interop/_bloq_to_cirq.py

"""Expose a bound native register body through Cirq's optional gate protocols."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from ...compiler.registers.primitives.power import register_power_ir
from ...compiler.registers.register_binding import BoundRegisterProgram
from ...core.ir import CircuitIR
from ...core.ir.program.signatures import Signature
from ._register_export import _register_shape
from ._register_workspace import bind_workspace_operations
from .conversion import _cirq, _export_cirq
from .models import CirqConversionError

if TYPE_CHECKING:
    from cirq import CircuitDiagramInfo, DecompositionContext, Gate, Qid
else:
    Gate = _cirq().Gate


class RegisterCirqGate(Gate):
    """Adapt an owned body over ordered register bits and optional private leases."""

    def __init__(
        self, body: CircuitIR | BoundRegisterProgram, signature: Signature
    ) -> None:
        labels: list[str] = []
        names: set[str] = set()
        for register in signature:
            if register.name in names:
                raise ValueError("Cirq gate register names must be distinct")
            names.add(register.name)
            shape = _register_shape(register)
            width = shape[-1]
            for index in register.all_idxs():
                label = register.name
                if index:
                    label += f"[{', '.join(map(str, index))}]"
                labels.extend([label] * width)
        bound = body if isinstance(body, BoundRegisterProgram) else None
        ir = body if isinstance(body, CircuitIR) else body.circuit
        if bound is None and len(labels) != ir.n_wires:
            raise ValueError(
                "Cirq gate requires one declared operand per physical wire"
            )
        if bound is not None:
            outputs = tuple(wire for word in bound.outputs for wire in word)
            if (
                not signature.thru_registers_only
                or bound.classical_outputs
                or outputs != tuple(range(len(labels)))
                or len(labels) > ir.n_wires
            ):
                raise ValueError("Cirq workspace gates require unchanged through wires")
        self._body = ir
        self._bound = bound
        self._signature = signature
        self._labels = tuple(labels)

    def _num_qubits_(self) -> int:
        return len(self._labels)

    def _decompose_(self, qubits: Sequence[Qid]) -> object:
        return self._decompose_with_context_(qubits)

    def _decompose_with_context_(
        self, qubits: Sequence[Qid], context: DecompositionContext | None = None
    ) -> object:
        if self._bound is not None and any(
            gate.name == "measure" for gate in self._body.instructions
        ):
            return NotImplemented
        try:
            result = _export_cirq(
                self._body,
                allow_lossy=False,
                include_inverse_phases=True,
                extent_declared_by_gate=True,
            )
        except CirqConversionError:
            return NotImplemented
        if self._bound is not None:
            operations = tuple(result.circuit.all_operations())
            if len(operations) != len(self._body.instructions):
                return NotImplemented
            return bind_workspace_operations(
                operations,
                self._bound.events,
                qubits,
                context.qubit_manager if context else None,
            )
        remapped = result.circuit.transform_qubits(lambda q: qubits[q.x])
        return tuple(remapped.all_operations())

    def _unitary_(self) -> object:
        if self._bound is not None or not self._signature.thru_registers_only:
            return NotImplemented
        import torch

        from ...simulation.unitary import get_unitary

        try:
            matrix = get_unitary(self._body, device="cpu", dtype=torch.complex128)
        except (TypeError, ValueError, NotImplementedError):
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
        if self._bound is not None:
            raise ValueError(
                "Cirq workspace gate power requires rebinding its lifetimes"
            )
        if power != -1 and not self._signature.thru_registers_only:
            raise ValueError("Repeated Cirq gate power requires through registers")
        instructions = register_power_ir(self._body, abs(power), adjoint=power < 0)
        signature = self._signature if power > 0 else self._signature.adjoint()
        return RegisterCirqGate(
            replace(self._body, instructions=instructions), signature
        )
