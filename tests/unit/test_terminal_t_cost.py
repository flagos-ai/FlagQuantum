"""Verify precision-sensitive terminal T budgets against the locked source."""

from math import pi

import pytest

from flagquantum.compiler._component_trace.resource_generalization import (
    ConjunctionCall,
    ControlledRotationCall,
    FixedTCall,
    PhaseCall,
    ResourceCall,
    RotationCall,
)
from flagquantum.compiler._component_trace.resource_model import GateCost
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler._component_trace.terminal_t_cost import (
    estimate_terminal_t_cost,
    rotation_t_cost,
)
from flagquantum.compiler.registers.controlled_swap import controlled_swap_bit_ir

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
def test_symbolic_precisions_and_multiplicities_match_source() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import (
        Rx,
        Ry,
        Rz,
        TGate,
        Toffoli,
        XPowGate,
        YPowGate,
        ZPowGate,
    )
    from qualtran.resource_counting.t_counts_from_sigma import t_counts_from_sigma
    from sympy import symbols

    z1, z2, x, y = symbols("z1 z2 x y", positive=True)
    repeats = symbols("repeats", nonnegative=True, integer=True)
    cases = (
        (RotationCall("zpow", 1, z1), ZPowGate(eps=z1), 1),
        (RotationCall("zpow", 1, z2), ZPowGate(eps=z2), 2),
        (RotationCall("zpow", 0.01, z1), ZPowGate(0.01, eps=z1), 1),
        (RotationCall("zpow", 0.01, z2), ZPowGate(0.01, eps=z2), 2),
        (RotationCall("rz", 0.01, z2), Rz(0.01, eps=z2), 3),
        (RotationCall("rx", 0.01, x), Rx(0.01, eps=x), 4),
        (RotationCall("xpow", 1, x), XPowGate(eps=x), 5),
        (RotationCall("xpow", 0.01, x), XPowGate(0.01, eps=x), 5),
        (RotationCall("ry", 0.01, y), Ry(0.01, eps=y), 6),
        (RotationCall("ypow", 1, y), YPowGate(eps=y), 7),
        (RotationCall("ypow", 0.01, y), YPowGate(0.01, eps=y), 7),
        (PhaseCall("t"), TGate(), 100),
        (FixedTCall("toffoli"), Toffoli(), repeats),
    )
    native: dict[ResourceCall, SymbolicInt] = {call: count for call, _, count in cases}
    with pytest.warns(DeprecationWarning):
        expected = t_counts_from_sigma({source: count for _, source, count in cases})
    actual = estimate_terminal_t_cost(native)
    assert actual == expected
    binding = {z1: 0.1, z2: 0.01, x: 0.001, y: 0.0001, repeats: 200}
    assert actual.subs(binding).doit() == expected.subs(binding).doit()


@pytest.mark.ftqc_reference
def test_concrete_controlled_stabilizer_guard_and_plain_t_like_angles() -> None:
    pytest.importorskip("qualtran")
    from qualtran import Controlled, CtrlSpec
    from qualtran.bloqs.basic_gates import Rx, YPowGate
    from qualtran.resource_counting.t_counts_from_sigma import t_counts_from_sigma

    for value in (0.0, 0.25, 0.5, 1.0, 3.0, -4.0, 1.0 + 1e-14):
        for inner, source in (
            (RotationCall("rx", value * pi, 1e-4), Rx(value * pi, eps=1e-4)),
            (RotationCall("ypow", value, 1e-4), YPowGate(value, eps=1e-4)),
        ):
            cases = [
                (inner, source),
                (
                    ControlledRotationCall(inner, (0,)),
                    Controlled(source, CtrlSpec(cvs=0)),
                ),
                (ControlledRotationCall(inner, (1,)), Controlled(source, CtrlSpec())),
                (
                    ControlledRotationCall(inner, (1, 1)),
                    Controlled(source, CtrlSpec(cvs=(1, 1))),
                ),
            ]
            for call, source_call in cases:
                with pytest.warns(DeprecationWarning):
                    assert estimate_terminal_t_cost({call: 3}) == t_counts_from_sigma(
                        {source_call: 3}
                    )
    assert rotation_t_cost(1e-4) == 25


@pytest.mark.ftqc_reference
def test_fixed_gate_and_measured_and_costs_match_source() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import SGate, TGate, TwoBitCSwap
    from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.resource_counting.t_counts_from_sigma import t_counts_from_sigma

    native: dict[ResourceCall, SymbolicInt] = {
        FixedTCall("fredkin"): 1,
        PhaseCall("s", True): 9,
        PhaseCall("t", True): 2,
        ConjunctionCall((0, 1)): 3,
        ConjunctionCall((0, 1), adjoint=True): 8,
        ConjunctionCall((1, 0, 1)): 10,
    }
    source = {
        TwoBitCSwap(): 1,
        SGate().adjoint(): 9,
        TGate().adjoint(): 2,
        And(0, 1): 3,
        And(0, 1, uncompute=True): 8,
        MultiAnd((1, 0, 1)): 10,
    }
    with pytest.warns(DeprecationWarning):
        assert estimate_terminal_t_cost(native) == t_counts_from_sigma(source) == 21
    emitted = sum(op.name in {"t", "tdg"} for op in controlled_swap_bit_ir(0, 1, 2))
    assert estimate_terminal_t_cost({FixedTCall("fredkin"): 1}) == emitted == 7
    assert (
        get_cost_value(TwoBitCSwap(), QECGatesCost()).total_t_count()
        == (GateCost(cswap=1).total_t_count())
        == 4
    )


@pytest.mark.ftqc_reference
def test_controlled_source_limitations_are_reported() -> None:
    pytest.importorskip("qualtran")
    from qualtran import Controlled, CtrlSpec
    from qualtran.bloqs.basic_gates import Rx, SGate, TGate
    from qualtran.resource_counting.t_counts_from_sigma import t_counts_from_sigma
    from sympy import Symbol

    angle = Symbol("a")
    with pytest.warns(DeprecationWarning), pytest.raises(NotImplementedError):
        t_counts_from_sigma({Controlled(Rx(angle), CtrlSpec()): 1})
    with pytest.raises(NotImplementedError, match="stabilizer verdict"):
        estimate_terminal_t_cost(
            {ControlledRotationCall(RotationCall("rx", angle), (1,)): 1}
        )
    for opcode, source in (("s", SGate()), ("t", TGate())):
        with pytest.warns(DeprecationWarning), pytest.raises(AssertionError):
            t_counts_from_sigma({Controlled(source, CtrlSpec()): 1})
        with pytest.raises(ValueError, match="no precision"):
            estimate_terminal_t_cost(
                {ControlledRotationCall(PhaseCall(opcode), (1,)): 1}
            )
