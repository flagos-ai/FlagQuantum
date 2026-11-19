"""Grouped predicate presentation follows the unchanged source callback."""

from functools import partial

import pytest

from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.drawer.control_symbols import control_predicate_wire_symbol
from flagquantum.drawer.register_score import RegisterStep
from tests.unit.test_register_control_predicate import (
    CASES,
    ControlCase,
    control_fixture,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("erase", [False, True])
def test_grouped_predicate_symbols_match_every_source_port(
    case: ControlCase, erase: bool
) -> None:
    from qualtran.bloqs.mcmt.ctrl_spec_and import CtrlSpecAnd

    spec, patterns, _ = control_fixture(case)
    source = CtrlSpecAnd(spec)
    if erase:
        source = source.adjoint()
    callback = partial(
        control_predicate_wire_symbol,
        values=tuple(p.values for p in patterns),
        uncompute=erase,
    )
    step = RegisterStep("predicate")
    for old in (None, *source.signature):
        indices = (None,) if old is None else tuple(old.all_idxs())
        register = (
            None
            if old is None
            else Register(
                old.name,
                RegisterType("opaque", old.bitsize),
                old.shape,
                RegisterSide(old.side.value),
            )
        )
        for index in indices:
            expected = source.wire_symbol(old, index or ()).json_dict()
            assert callback(step, register, index) == (
                expected["symb_cls"],
                expected["symb_attributes"],
            )
