"""Pinned source signatures, declared calls, and append for the gradient tutorial."""

from collections import Counter
from dataclasses import fields
from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
from sympy import Symbol, simplify

from examples.algorithms.phase_gradient_catalog import (
    GradientAdditionExample,
    GradientStateExample,
    GradientUnitaryExample,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    GateCost,
    estimate_component_resources,
)
from flagquantum.compiler.registers.phase.phase_gradient import (
    source_phase_gradient_unitary_cost,
    source_symbolic_phase_gradient_cost,
)
from flagquantum.dynamic import DynamicCircuit

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source_module() -> Any:
    import kickmix
    from qualtran.bloqs.rotations import phase_gradient

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(phase_gradient.__file__).parent
    for name, digest in (
        ("__init__.py", "e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8"),
        ("phase_gradient.ipynb", "ffe4116032691edf96d3ede42c32b03e2fbfa3d6"),
        ("phase_gradient.py", "97885bc9d4c054df375854ea6d9d4d3afecb63e4"),
        ("phase_gradient_test.py", "af6677a79bdc1bff2a02dd44dd22b618ea417263"),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return phase_gradient


def source_call_label(child: Any) -> str:
    name = type(child).__name__
    if name in {"ZPowGate", "CZPowGate"}:
        return f"{name}({child.exponent}, {child.eps})"
    if name == "Allocate":
        return f"Allocate({child.dtype.num_qubits})"
    if name == "OnEach":
        return f"OnEach({child.n}, {type(child.gate).__name__})"
    if name == "PhaseGradientUnitary":
        return f"{name}({child.bitsize}, {child.exponent})"
    if name == "AddIntoPhaseGrad":
        assert (child.right_shift, child.sign, child.controlled_by) == (0, 1, None)
        return f"{name}({child.x_bitsize}, {child.phase_bitsize})"
    assert name == "Toffoli"
    return name


@pytest.mark.parametrize(
    "op, factory, literal",
    [
        (GradientUnitaryExample(), "_phase_gradient_unitary", (4, 0, 0, 0, 0)),
        (GradientUnitaryExample(Symbol("n")), "_phase_gradient_unitary_symbolic", None),
        (GradientStateExample(), "_phase_gradient_state", (4, 4, 0, 0, 0)),
        (GradientAdditionExample(), "_add_into_phase_grad", (0, 9, 12, 33, 3)),
        (
            GradientAdditionExample(scaled=True),
            "_add_scaled_val_into_phase_reg",
            (0, 3, 4, 8, 1),
        ),
    ],
)
def test_locked_tutorial_factory(
    source_module: Any,
    op: GradientUnitaryExample | GradientStateExample | GradientAdditionExample,
    factory: str,
    literal: tuple[int, int, int, int, int] | None,
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.resource_counting.generalizers import ignore_split_join

    source = getattr(source_module, factory).make()
    assert [
        (r.name, r.dtype.num_qubits, r.side.name, r.shape) for r in op.signature
    ] == [(r.name, r.bitsize, r.side.name, r.shape) for r in source.signature]
    assert [dict(r.dtype.parameters) for r in op.signature] == [
        {"fractional_width": r.dtype.num_frac, "signed": r.dtype.signed}
        for r in source.signature
    ]
    _, leaves = source.call_graph(max_depth=1, generalizer=ignore_split_join)
    assert op.call_graph().leaf_counts == {
        source_call_label(child): n for child, n in leaves.items()
    }
    model = op.resource_model()
    ref_gates = get_cost_value(source, QECGatesCost())
    ref_width = get_cost_value(source, QubitCount())
    if literal is None:
        assert not isinstance(model, ComponentCost)
        assert model.logical_qubits == ref_width
        assert model.rotation == ref_gates.rotation
        with pytest.raises(ValueError, match="Bind a concrete"):
            DynamicCircuit(4).append(op)
        return
    assert isinstance(model, ComponentCost)
    result = estimate_component_resources(model)
    assert result.peak_logical_qubits == ref_width
    assert result.gates == GateCost(
        **{f.name: getattr(ref_gates, f.name) for f in fields(GateCost)}
    )
    circuit = DynamicCircuit(op.qubit_count)
    circuit.append(op)
    count = Counter(g.name for g in circuit.to_ir().instructions)
    assert (
        count["phase"],
        count["h"],
        count["t"] + count["tdg"],
        count["cx"],
        count["measure"],
    ) == literal
    assert count["reset"] == count["measure"]


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize(
    "width,exponent", [(1, -1), (4, 0), (4, 1), (5, -0.75), (8, 0.625)]
)
def test_declared_gradient_cost(width: int, exponent: float, controlled: bool) -> None:
    from qualtran.bloqs.rotations.phase_gradient import PhaseGradientUnitary
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = PhaseGradientUnitary(width, exponent=exponent, is_controlled=controlled)
    cost = estimate_component_resources(
        source_phase_gradient_unitary_cost(width, exponent, controlled=controlled)
    )
    ref = get_cost_value(source, QECGatesCost())
    assert cost.peak_logical_qubits == get_cost_value(source, QubitCount())
    assert cost.gates == GateCost(
        **{f.name: getattr(ref, f.name) for f in fields(GateCost)}
    )


@pytest.mark.parametrize("controlled", [False, True])
def test_symbolic_gradient_declaration(controlled: bool) -> None:
    from qualtran.bloqs.rotations.phase_gradient import PhaseGradientUnitary
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    width = Symbol("n")
    source = PhaseGradientUnitary(width, is_controlled=controlled)
    cost = source_symbolic_phase_gradient_cost(width, controlled=controlled)
    ref = get_cost_value(source, QECGatesCost())
    assert cost.logical_qubits == get_cost_value(source, QubitCount())
    for field in fields(GateCost):
        assert simplify(getattr(cost, field.name) - getattr(ref, field.name)) == 0
