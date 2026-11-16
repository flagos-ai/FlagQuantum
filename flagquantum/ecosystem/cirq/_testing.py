# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 cirq_interop/testing.py.

"""Helpers for testing explicitly supplied Cirq operations."""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from ...core.ir.program.signatures import Signature
from ._decompose import decompose_cirq_once
from ._legacy_t_complexity import LegacyTComplexity, cirq_t_complexity
from ._qubit_manager import InteropQubitManager
from ._register_export import _named_qubits, _Qureg, _register_shape
from .conversion import _cirq


@dataclass(frozen=True)
class CirqGateFixture:
    """Bind a Cirq gate to a concrete native signature for decomposition tests."""

    gate: Any
    signature: Signature
    qubit_manager: Any = None

    @cached_property
    def quregs(self) -> dict[str, _Qureg]:
        cirq = _cirq()
        ports = {
            register.name: _named_qubits(cirq, register, _register_shape(register))
            for register in self.signature
        }
        if len(ports) != len(self.signature):
            raise ValueError("Cirq test registers must have distinct names")
        return ports

    @cached_property
    def _port_qubits(self) -> tuple[Any, ...]:
        qubits = tuple(qubit for port in self.quregs.values() for qubit in port.flat)
        if len(qubits) != len(set(qubits)):
            raise ValueError("Cirq test register names produce duplicate qubits")
        return qubits

    @cached_property
    def operation(self) -> Any:
        return self.gate.on(*self._port_qubits)

    @cached_property
    def circuit(self) -> Any:
        return _cirq().Circuit(self.operation)

    @cached_property
    def decomposed_circuit(self) -> Any:
        cirq = _cirq()
        manager = InteropQubitManager(self.qubit_manager)
        manager.manage_qubits(self._port_qubits)
        context = cirq.DecompositionContext(qubit_manager=manager)
        return cirq.Circuit(cirq.decompose(self.operation, context=context))

    @cached_property
    def all_qubits(self) -> list[Any]:
        work = self.decomposed_circuit.all_qubits() - frozenset(self._port_qubits)
        return [*self._port_qubits, *sorted(work)]


def get_circuit_input_output(
    circuit: Any,
    qubit_order: Sequence[Any],
    inputs: Sequence[int],
    outputs: Sequence[int],
    decimals: int = 2,
) -> tuple[str, str]:
    """Return the simulated Dirac state and expected bitstring in caller order."""
    result = (
        _cirq()
        .Simulator(dtype=np.complex128)
        .simulate(circuit, initial_state=inputs, qubit_order=qubit_order)
    )
    actual = result.dirac_notation(decimals=decimals)[1:-1]
    return actual, "".join(str(bit) for bit in outputs)


def assert_circuit_input_output(
    circuit: Any,
    qubit_order: Sequence[Any],
    inputs: Sequence[int],
    outputs: Sequence[int],
    decimals: int = 2,
) -> None:
    """Check a reversible basis transition at the requested display precision."""
    actual, expected = get_circuit_input_output(
        circuit, qubit_order, inputs, outputs, decimals
    )
    assert actual == expected, f"\n{actual=}\n{expected=}"


def assert_decomposition_cost(value: Any) -> None:
    """Compare the SDK object's explicit cost with its one-step decomposition."""
    annotation = getattr(value, "_t_complexity_", None)
    expected = NotImplemented if annotation is None else annotation()
    if expected is None or expected is NotImplemented:
        raise AssertionError("No consistent legacy T complexity: no _t_complexity_")
    children = decompose_cirq_once(value)
    if children is None:
        raise AssertionError("No consistent legacy T complexity: no decomposition")
    declared = LegacyTComplexity(expected.t, expected.clifford, expected.rotations)
    observed = cirq_t_complexity(children)
    assert declared == observed, f"{declared} != {observed}"
