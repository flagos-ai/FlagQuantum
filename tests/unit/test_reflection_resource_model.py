"""Retain complete source reflection graph phase branches and duplicate keys."""

from cmath import exp
from typing import Any

import pytest

from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler.registers.preparation.prepare_reflection import (
    source_reflection_declared_cost,
)
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", range(6))
@pytest.mark.parametrize("control_value", (None, 0, 1))
@pytest.mark.parametrize("coefficient", (1, -1, 1j, -1j, exp(0.37j)))
def test_complete_source_declared_phase_and_polarity_cost(
    width: int, control_value: int | None, coefficient: complex
) -> None:
    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )

    source = ReflectionUsingPrepare.reflection_around_zero(
        () if not width else (width,),
        control_val=control_value,
        global_phase=coefficient,
        eps=1e-7,
    )
    identity = ComponentCost("IdentityPrepare", width)
    model = source_reflection_declared_cost(
        width,
        0,
        identity,
        identity,
        control_value=control_value,
        global_phase=coefficient,
        precision=1e-7,
        prepare_is_self_adjoint=True,
    )
    assert_source_model(model, source)
    calls = source.build_call_graph(None)
    assert calls[source.prepare_gate] == 1
    assert len(model.calls) == 2
    assert model.gates.clifford == (
        2 if control_value is None else 2 * int(control_value == 0 and coefficient != 1)
    ) + int(control_value is not None and coefficient in (-1, 1j, -1j))


def test_declared_cost_rejects_invalid_control_and_equality_metadata() -> None:
    identity = ComponentCost("IdentityPrepare", 2)
    with pytest.raises(ValueError, match="zero or one"):
        source_reflection_declared_cost(2, 0, identity, identity, control_value=2)
    raw_equality: Any = 1
    with pytest.raises(ValueError, match="must be boolean"):
        source_reflection_declared_cost(
            2, 0, identity, identity, prepare_is_self_adjoint=raw_equality
        )
