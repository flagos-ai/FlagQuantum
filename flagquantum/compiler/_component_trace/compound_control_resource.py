# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Qualtran v0.7.0 compound control systems.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/specialized_ctrl.py

"""Select resource call identities when a compound gate gains controls."""

from .resource_generalization import (
    CliffordCall,
    ControlledRegisterSwapCall,
    ControlledRotationCall,
    ControlledSwapCall,
    FixedTCall,
    MergedFredkinCall,
    PhaseCall,
    PredicateControlledCall,
    ResourceCall,
    RotationCall,
    ZeroCostCall,
)
from .resource_model import ComponentCost, GateCost
from .symbolic_math import SymbolicInt, SymbolicReal

CONTROLLED_HADAMARD_COST = ComponentCost(
    "CHadamard", 2, GateCost(clifford=4, rotation=2)
)


_CONTROLLED_PAULI = {
    "x": CliffordCall("cx", 2),
    "y": CliffordCall("cy", 2),
    "z": CliffordCall("cz", 2),
}


def global_phase_cost_key(
    exponent: SymbolicReal,
    controls: tuple[int, ...] = (),
    precision: SymbolicReal = 1e-11,
) -> ResourceCall:
    """Use the source zero-cost leaf or the controlled Z-power call chain."""
    if any(type(bit) is not int or bit not in (0, 1) for bit in controls):
        raise ValueError("Global phase controls require binary activation values")
    if not controls:
        return ZeroCostCall("global_phase", (exponent, precision))
    positive = RotationCall("zpow", exponent, precision)
    return positive if controls == (1,) else PredicateControlledCall(positive, controls)


def controlled_swap_cost_key(
    width: SymbolicInt, controls: tuple[int, ...]
) -> ResourceCall:
    """Model the source register swap under its first group of controls."""
    if not controls or any(
        type(bit) is not int or bit not in (0, 1) for bit in controls
    ):
        raise ValueError(
            "Controlled swap requires a nonempty binary activation pattern"
        )
    child = ControlledSwapCall(width)
    return child if controls == (1,) else PredicateControlledCall(child, controls)


def compound_control_key(
    gate: ResourceCall, controls: tuple[int, ...]
) -> ResourceCall | None:
    """Select a locked compound-control path for a concrete activation pattern."""
    if not controls or any(
        type(bit) is not int or bit not in (0, 1) for bit in controls
    ):
        raise ValueError(
            "Compound control requires a nonempty binary activation pattern"
        )
    if isinstance(gate, PhaseCall):
        phase = ControlledRotationCall(gate, (1,))
        return phase if controls == (1,) else PredicateControlledCall(phase, controls)
    if isinstance(gate, CliffordCall) and gate.opcode in _CONTROLLED_PAULI:
        if gate.opcode == "x" and controls == (1, 1):
            return FixedTCall("toffoli")
        pauli = _CONTROLLED_PAULI[gate.opcode]
        return pauli if controls == (1,) else PredicateControlledCall(pauli, controls)
    if isinstance(gate, CliffordCall) and gate.opcode == "swap_bit":
        swap = FixedTCall("fredkin")
        return swap if controls == (1,) else PredicateControlledCall(swap, controls)
    if isinstance(gate, CliffordCall) and gate.opcode == "h":
        return (
            CONTROLLED_HADAMARD_COST
            if controls == (1,)
            else PredicateControlledCall(CONTROLLED_HADAMARD_COST, controls)
        )
    if isinstance(gate, CliffordCall) and gate.opcode in {"cy", "cz"}:
        if len(controls) == 1:
            return PredicateControlledCall(gate, (*controls, 1))
        return PredicateControlledCall(PredicateControlledCall(gate, (1, 1)), controls)
    if isinstance(gate, CliffordCall) and gate.opcode == "cx":
        toffoli = FixedTCall("toffoli")
        return (
            toffoli if controls == (1,) else PredicateControlledCall(toffoli, controls)
        )
    if isinstance(gate, FixedTCall) and gate.opcode == "toffoli":
        return PredicateControlledCall(CliffordCall("cx", 2), (*controls, 1, 1))
    if isinstance(gate, FixedTCall) or isinstance(gate, MergedFredkinCall):
        existing = (1,) if isinstance(gate, FixedTCall) else gate.controls
        if len(controls) == 1:
            return MergedFredkinCall((*controls, *existing))
        return PredicateControlledCall(MergedFredkinCall((1, *existing)), controls)
    if isinstance(gate, ControlledSwapCall) and not gate.approximate:
        child = ControlledRegisterSwapCall(gate.width)
        return child if controls == (1,) else PredicateControlledCall(child, controls)
    if isinstance(gate, PredicateControlledCall) and isinstance(gate.controls, tuple):
        return PredicateControlledCall(gate.positive, (*controls, *gate.controls))
    return None
