# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 cirq_gate_to_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/cirq_interop/_cirq_to_bloq.py

"""Map supported Cirq gates to FlagQuantum's declared resource identities."""

from collections.abc import Hashable
from typing import Literal, cast

from ...compiler._component_trace.resource_generalization import (
    CliffordCall,
    ControlledRotationCall,
    ControlledSwapCall,
    ControlledZPowerCall,
    ControlledZRotationCall,
    FixedTCall,
    PhaseCall,
    PredicateControlledCall,
    ResourceCall,
    RotationCall,
    ZeroCostCall,
)
from ...compiler._component_trace.resource_model import ComponentCost, GateCost
from ...compiler._component_trace.symbolic_math import SymbolicReal
from .conversion import _cirq

_CONTROLLED_H = ComponentCost("CHadamard", 2, GateCost(clifford=4, rotation=2))


def cirq_gate_resource_key(gate: object) -> ResourceCall | None:
    """Return a recognized resource identity or leave the gate at the SDK boundary.

    This follows the source's exact-instance lookup before inspecting power
    parameters. Unknown gates and unsupported control wrappers return None.
    It does not import or lower a circuit.
    """
    cirq = _cirq()
    if not isinstance(gate, cirq.Gate) or any(d != 2 for d in cirq.qid_shape(gate)):
        return None

    fixed: dict[object, ResourceCall] = {
        cirq.T: PhaseCall("t"),
        cirq.T**-1: PhaseCall("t", adjoint=True),
        cirq.S: PhaseCall("s"),
        cirq.S**-1: PhaseCall("s", adjoint=True),
        cirq.H: CliffordCall("h", 1),
        cirq.ControlledGate(cirq.H): _CONTROLLED_H,
        cirq.CNOT: CliffordCall("cx", 2),
        cirq.TOFFOLI: FixedTCall("toffoli"),
        cirq.X: CliffordCall("x", 1),
        cirq.Y: CliffordCall("y", 1),
        cirq.ControlledGate(cirq.Y): CliffordCall("cy", 2),
        cirq.Z: CliffordCall("z", 1),
        cirq.CZ: CliffordCall("cz", 2),
        cirq.SWAP: CliffordCall("swap_bit", 2),
        cirq.CSWAP: ControlledSwapCall(1),
        cirq.I: ZeroCostCall("identity", (1,)),
    }
    try:
        key = fixed.get(gate)
    except TypeError:
        return None
    if key is not None:
        return key

    if isinstance(gate, cirq.ControlledGate):
        controls = gate.control_values
        if isinstance(controls, cirq.ProductOfSums) and all(
            len(allowed) == 1 for allowed in controls
        ):
            return _controlled_key(
                cirq_gate_resource_key(gate.sub_gate),
                tuple(allowed[0] for allowed in controls),
            )
        return None

    rotations: dict[type, Literal["rx", "ry", "rz"]] = {
        cirq.Rx: "rx",
        cirq.Ry: "ry",
        cirq.Rz: "rz",
    }
    gate_type = type(gate)
    if gate_type in rotations:
        return RotationCall(rotations[gate_type], cast(SymbolicReal, gate._rads))

    powers: dict[type, Literal["xpow", "ypow", "zpow"]] = {
        cirq.XPowGate: "xpow",
        cirq.YPowGate: "ypow",
        cirq.ZPowGate: "zpow",
    }
    if gate_type in powers and gate.global_shift == 0:
        return RotationCall(powers[gate_type], cast(SymbolicReal, gate.exponent))
    if gate_type is cirq.CZPowGate and gate.global_shift == 0:
        return ControlledZPowerCall(cast(SymbolicReal, gate.exponent))
    if isinstance(gate, cirq.GlobalPhaseGate):
        return ZeroCostCall("global_phase", (cast(Hashable, gate.coefficient),))
    return None


def _controlled_key(
    key: ResourceCall | None, controls: tuple[int, ...]
) -> ResourceCall | None:
    if isinstance(key, ZeroCostCall) and key.opcode == "identity":
        return ZeroCostCall("identity", (1 + len(controls),))
    if isinstance(key, CliffordCall) and key.opcode == "x" and controls == (1, 1):
        return FixedTCall("toffoli")
    positive = _single_positive_key(key)
    if positive is None or controls == (1,):
        return positive
    return PredicateControlledCall(positive, controls)


def _single_positive_key(key: ResourceCall | None) -> ResourceCall | None:
    if isinstance(key, RotationCall) and key.opcode in {"zpow", "rz"}:
        assert key.parameter is not None
        if key.opcode == "rz":
            return ControlledZRotationCall(key.parameter, key.precision)
        return ControlledZPowerCall(key.parameter, key.precision)
    if isinstance(key, (RotationCall, PhaseCall)):
        return ControlledRotationCall(key, (1,))
    if not isinstance(key, CliffordCall):
        return None
    if key.opcode == "x":
        return CliffordCall("cx", 2)
    if key.opcode == "y":
        return CliffordCall("cy", 2)
    if key.opcode == "z":
        return CliffordCall("cz", 2)
    if key.opcode == "h":
        return _CONTROLLED_H
    if key.opcode == "swap_bit":
        return FixedTCall("fredkin")
    return None
