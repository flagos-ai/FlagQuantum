"""The original finite call graph feeds shared resource aggregation."""

from dataclasses import replace

import pytest

from flagquantum.algorithms.primitives.generalized_qsp_model import (
    source_generalized_qsp_cost,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    GateCost,
    estimate_component_resources,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

ROTATION = ComponentCost("rotation", 1, GateCost(rotation=3))
ZERO_U = ComponentCost("zero-U", 5, GateCost(toffoli=2))
ADJOINT = ComponentCost("U-adjoint", 4, GateCost(t=7))
CONTROLLED_ADJOINT = ComponentCost("controlled-adjoint", 6, GateCost(and_bloq=5))


@pytest.mark.parametrize("count", [2, 3, 6])
@pytest.mark.parametrize("negative_power", [0, 1, 2, 4, 8])
def test_source_keys_multiplicities_order_model_gates_and_width(
    count: int, negative_power: int
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.basic_gates.su2_rotation import SU2RotationGate
    from qualtran.bloqs.qsp.generalized_qsp import GeneralizedQSP as Source
    from qualtran.resource_counting import SympySymbolAllocator

    source_u = XGate()
    source = Source(
        source_u, (0.1,) * count, (0.2,) * count, negative_power=negative_power
    )
    expected = []
    for key, repetitions in source.build_call_graph(SympySymbolAllocator()).items():
        if isinstance(key, SU2RotationGate):
            name = "rotation"
        elif key == source_u.controlled(ctrl_spec=CtrlSpec(cvs=0)):
            name = "zero-U"
        elif key == source_u.adjoint():
            name = "U-adjoint"
        else:
            assert key == source_u.adjoint().controlled()
            name = "controlled-adjoint"
        expected.append((name, int(repetitions)))
    model = source_generalized_qsp_cost(
        count,
        3,
        negative_power=negative_power,
        signal_rotation=ROTATION,
        zero_control_unitary=ZERO_U,
        unitary_adjoint=ADJOINT,
        controlled_adjoint=CONTROLLED_ADJOINT,
    )
    assert [(call.component.name, call.repetitions) for call in model.calls] == expected
    counts = dict(expected)
    result = estimate_component_resources(model)
    assert result.gates == GateCost(
        rotation=3 * count,
        toffoli=2 * counts.get("zero-U", 0),
        t=7 * counts.get("U-adjoint", 0),
        and_bloq=5 * counts.get("controlled-adjoint", 0),
    )
    assert result.peak_logical_qubits == max(
        [4, *[call.component.qubits + call.bystander_qubits for call in model.calls]]
    )
    assert result.success_probability == 1
    assert model.gates == GateCost()


@pytest.mark.parametrize(
    "field,value",
    [
        ("coefficient_count", 1),
        ("coefficient_count", True),
        ("system_width", -1),
        ("system_width", 1.5),
        ("negative_power", -1),
        ("negative_power", True),
    ],
)
def test_invalid_finite_model_inputs(field: str, value: int) -> None:
    kwargs = dict(coefficient_count=3, system_width=2, negative_power=0)
    kwargs[field] = value
    with pytest.raises(ValueError):
        source_generalized_qsp_cost(
            **kwargs,
            signal_rotation=ROTATION,
            zero_control_unitary=ZERO_U,
            unitary_adjoint=ADJOINT,
            controlled_adjoint=CONTROLLED_ADJOINT,
        )


def test_inactive_children_do_not_inflate_width_or_require_numerical_phases() -> None:
    absent = replace(CONTROLLED_ADJOINT, qubits=1000)
    model = source_generalized_qsp_cost(
        3,
        3,
        negative_power=0,
        signal_rotation=ROTATION,
        zero_control_unitary=ZERO_U,
        unitary_adjoint=absent,
        controlled_adjoint=absent,
    )
    assert estimate_component_resources(model).peak_logical_qubits == 5
    assert len(model.calls) == 2
    # Replace child pricing without touching the consuming GQSP formula.
    replacement = source_generalized_qsp_cost(
        3,
        3,
        negative_power=0,
        signal_rotation=replace(ROTATION, gates=GateCost(t=11)),
        zero_control_unitary=ZERO_U,
        unitary_adjoint=absent,
        controlled_adjoint=absent,
    )
    assert estimate_component_resources(replacement).gates == GateCost(t=33, toffoli=4)
