# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Qualtran v0.7.0 compound control systems.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/specialized_ctrl.py

"""Select resource call identities when a compound gate gains controls."""

from .resource_generalization import (
    CliffordCall,
    ControlledRegisterSwapCall,
    ControlledSwapCall,
    FixedTCall,
    MergedFredkinCall,
    PredicateControlledCall,
    ResourceCall,
)


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
