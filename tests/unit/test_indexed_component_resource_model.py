"""Compose controlled child resources against untouched indexed source graphs."""

from itertools import product
from math import prod

import pytest

from flagquantum.compiler._component_trace.gate_budget import t_and_ccz_budget
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
    estimate_component_resources,
)
from flagquantum.compiler.registers.arithmetic_resource_model import source_add_cost
from flagquantum.compiler.registers.indexed_component_resource_model import (
    indexed_component_cost,
)
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "shape,widths,controls,position",
    [
        (*case, position)
        for case in (
            ((1,), (1,), 1),
            ((1,), (1,), 2),
            ((2,), (1,), 0),
            ((3,), (2,), 0),
            ((3,), (4,), 1),
            ((4,), (2,), 1),
            ((2,), (1,), 2),
            ((2,), (1,), 3),
            ((2, 3), (1, 2), 0),
            ((2, 2, 2), (1, 1, 1), 1),
            ((1, 2), (1, 1), 1),
        )
        for position in (False, True)
        if position or (case[2] <= 1 and prod(case[0]) > 1)
    ],
)
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("family", ["pauli", "phase"])
def test_all_categories_width_and_magic_budgets_match_source(
    shape: tuple[int, ...],
    widths: tuple[int, ...],
    controls: int,
    position: bool,
    adjoint: bool,
    family: str,
) -> None:
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
    from qualtran import BQUInt, QAny, Register
    from qualtran.bloqs.basic_gates import (
        Hadamard,
        Identity,
        SGate,
        TGate,
        XGate,
        YGate,
        ZGate,
    )
    from qualtran.bloqs.multiplexers.apply_gate_to_lth_target import ApplyGateToLthQubit
    from qualtran.bloqs.multiplexers.apply_lth_bloq import ApplyLthBloq
    from qualtran.cirq_interop import cirq_gate_to_bloq
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    count = prod(shape)
    source_types = (
        (Identity, XGate, YGate, ZGate)
        if family == "pauli"
        else (Hadamard, SGate, TGate)
    )
    native_counts = (
        (GateCost(), GateCost(clifford=1), GateCost(clifford=1), GateCost(clifford=1))
        if family == "pauli"
        else (
            GateCost(clifford=4, rotation=2),
            GateCost(rotation=1),
            GateCost(rotation=1),
        )
    )
    cirq_types = (
        (cirq.I, cirq.X, cirq.Y, cirq.Z)
        if family == "pauli"
        else (cirq.H, cirq.S, cirq.T)
    )
    child_widths = (2,) * len(native_counts)
    if position and family == "phase":
        # Actual target-position export recognizes CZPowGate, not Controlled(S/T).
        native_counts = (
            GateCost(clifford=4, rotation=2),
            GateCost(and_bloq=1, clifford=2, measurement=1),
            GateCost(t=1, and_bloq=1, clifford=1, measurement=1),
        )
        child_widths = (2, 3, 3)
    children = tuple(
        ComponentCost(
            f"leaf{index}",
            child_widths[index % len(child_widths)],
            native_counts[index % len(native_counts)],
        )
        for index in range(count)
    )
    source_ops = [source_types[index % len(source_types)]() for index in range(count)]
    if position:
        source_child_ops = [
            cirq_gate_to_bloq(cirq_types[index % len(cirq_types)].controlled())
            for index in range(count)
        ]
        if adjoint:
            source_child_ops = [op.adjoint() for op in source_child_ops]
    elif adjoint:
        source_child_ops = [op.adjoint().controlled() for op in source_ops]
    else:
        source_child_ops = [op.controlled() for op in source_ops]
    for child, original in zip(children, source_child_ops, strict=True):
        assert_source_model(child, original)
    regs = tuple(
        Register(f"s{index}", BQUInt(width, size))
        for index, (width, size) in enumerate(zip(widths, shape, strict=True))
    )
    if position:
        cirq_types = (
            (cirq.I, cirq.X, cirq.Y, cirq.Z)
            if family == "pauli"
            else (cirq.H, cirq.S, cirq.T)
        )
        source = ApplyGateToLthQubit(
            regs,
            lambda *indices: cirq_types[
                int(np.ravel_multi_index(indices, shape)) % len(cirq_types)
            ],
            control_regs=(Register("control", QAny(controls)),) if controls else (),
        )
    else:
        source = ApplyLthBloq(
            np.array(source_ops, dtype=object).reshape(shape),
            selection_regs=regs,
            control_val=1 if controls else None,
        )
    original = source.adjoint() if adjoint else source
    model = indexed_component_cost(
        widths,
        shape,
        children,
        target_width=count if position else 1,
        child_target_width=1,
        control_count=controls,
        adjoint=adjoint,
    )
    assert_source_model(model, original)
    result = estimate_ftqc_resources(model)
    source_gates = get_cost_value(original, QECGatesCost())
    assert result.logical.gates.total_t_count() == source_gates.total_t_count()
    assert (
        t_and_ccz_budget(result.logical.gates) == source_gates.total_t_and_ccz_count()
    )
    assert model.gates.and_bloq == count - 2 + controls
    assert result.emitted is None


def _add_model(width: int, constant: int) -> ComponentCost:
    return ComponentCost(
        "ControlledAddK",
        width + 1,
        GateCost(clifford=2 * constant.bit_count()),
        (CostCall(source_add_cost(width, width), bystander_qubits=1),),
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "width,where,controlled,adjoint",
    tuple(product((1, 2, 3), (0, 2), (False, True), (False, True))),
)
def test_heterogeneous_child_work_is_charged_at_its_actual_prefix(
    width: int, where: int, controlled: bool, adjoint: bool
) -> None:
    pytest.importorskip("qualtran")
    import numpy as np
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import AddK
    from qualtran.bloqs.arithmetic.bitwise import XorK
    from qualtran.bloqs.multiplexers.apply_lth_bloq import ApplyLthBloq

    cheap = ComponentCost("ControlledXorK", width + 1, GateCost(clifford=1))
    expensive = _add_model(width, 1)
    ops = [XorK(QUInt(width), 1)] * 3
    models = [cheap] * 3
    ops[where], models[where] = AddK(QUInt(width), 1), expensive
    for child, op in zip(models, ops, strict=True):
        assert_source_model(child, (op.adjoint() if adjoint else op).controlled())
    source = ApplyLthBloq(
        np.array(ops, dtype=object), control_val=1 if controlled else None
    )
    model = indexed_component_cost(
        (2,),
        (3,),
        tuple(models),
        target_width=width,
        control_count=int(controlled),
        adjoint=adjoint,
    )
    assert_source_model(model, source.adjoint() if adjoint else source)
    expected_depth = (2 if where == 0 else 1) - int(not controlled)
    call = next(call for call in model.calls if call.component is expensive)
    assert call.bystander_qubits == int(controlled) + 2 + expected_depth - 1
    if width == 3 and controlled:
        assert estimate_component_resources(model).peak_logical_qubits == (
            13 if where == 0 else 12
        )


def test_dimension_model_does_not_emit_or_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.compiler.registers.indexed_component as emitters

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("Analytical indexed costs must not construct a gate stream")

    monkeypatch.setattr(emitters, "indexed_component_ir", unexpected)
    monkeypatch.setattr(emitters, "append_nested_bounded_unary_tree", unexpected)
    model = indexed_component_cost(
        (256,),
        (3,),
        (ComponentCost("CX", 2, GateCost(clifford=1)),) * 3,
        target_width=1,
        control_count=1,
    )
    report = estimate_ftqc_resources(model)
    assert report.logical.peak_logical_qubits == 260
    assert report.logical.gates.total_t_count() == 8
    assert report.emitted is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"axis_widths": ()},
        {"axis_widths": (0,)},
        {"extents": (3,)},
        {"extents": (True,)},
        {"target_width": True},
        {"target_width": -1},
        {"control_count": -1},
        {"control_count": True},
        {"child_target_width": 2},
        {"adjoint": 1},
        {"children": (ComponentCost("bad", 1),) * 2},
        {"children": ()},
        {"extents": (1,), "children": (ComponentCost("CX", 2),)},
    ],
)
def test_invalid_model_dimensions_fail_closed(kwargs: dict[str, object]) -> None:
    args = dict(
        axis_widths=(1,),
        extents=(2,),
        children=(ComponentCost("CX", 2),) * 2,
        target_width=1,
    )
    args.update(kwargs)
    with pytest.raises(ValueError):
        indexed_component_cost(**args)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("adjoint", [False, True])
def test_phase_only_children_have_no_target_bank(adjoint: bool) -> None:
    pytest.importorskip("qualtran")
    import numpy as np
    from qualtran.bloqs.basic_gates import GlobalPhase
    from qualtran.bloqs.multiplexers.apply_lth_bloq import ApplyLthBloq

    children = (
        ComponentCost("ControlledMinus", 1, GateCost(clifford=1)),
        ComponentCost("ControlledEighthTurn", 1, GateCost(t=1)),
    )
    source = ApplyLthBloq(
        np.array((GlobalPhase(exponent=1), GlobalPhase(exponent=0.25)), dtype=object)
    )
    model = indexed_component_cost(
        (1,), (2,), children, target_width=0, adjoint=adjoint
    )
    assert_source_model(model, source.adjoint() if adjoint else source)
    assert estimate_component_resources(model).peak_logical_qubits == 1


@pytest.mark.parametrize("adjoint", [False, True])
def test_composed_arithmetic_child_full_complex_branches(adjoint: bool) -> None:
    import torch

    from examples.algorithms.indexed_component_resources import make_gates, make_model
    from tests.unit.measurement_checks import forced_branch

    initial = torch.zeros(1024, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    generator = torch.Generator().manual_seed(59)
    amplitudes = torch.complex(
        torch.randn(32, generator=generator, dtype=torch.float64),
        torch.randn(32, generator=generator, dtype=torch.float64),
    )
    for word in range(32):
        enable, address, value = (word >> 4) & 1, (word >> 2) & 3, word & 3
        if address == 3:
            continue
        updated = (
            ((value + (-1 if adjoint else 1)) % 4 if address == 2 else value ^ 1)
            if enable
            else value
        )
        initial[word << 5] = amplitudes[word]
        expected[((word & ~3) | updated) << 5] = amplitudes[word]
    norm = initial.norm()
    initial /= norm
    expected /= norm
    gates = make_gates(adjoint=adjoint)
    events = sum(gate.name == "measure" for gate in gates)
    assert events == estimate_component_resources(make_model()).gates.measurement == 3
    for outcomes in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, gates, outcomes),
            expected / 2 ** (events / 2),
            atol=2e-13,
            rtol=2e-13,
        )


def test_resource_example() -> None:
    from examples.algorithms.indexed_component_resources import main

    main()


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("adjoint", [False, True])
def test_composed_child_gate_flow_matches_original_source(adjoint: bool) -> None:
    pytest.importorskip("qualtran")
    import numpy as np
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import AddK
    from qualtran.bloqs.arithmetic.bitwise import XorK
    from qualtran.bloqs.multiplexers.apply_lth_bloq import ApplyLthBloq

    from examples.algorithms.indexed_component_resources import make_gates
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_relative_controlled_swap import assert_source_dependencies

    source = ApplyLthBloq(
        np.array(
            (XorK(QUInt(2), 1), XorK(QUInt(2), 1), AddK(QUInt(2), 1)), dtype=object
        ),
        control_val=1,
    )
    original = source.adjoint().decompose_bloq() if adjoint else source
    # Preserve the source export's reserved parent-then-child physical pool.
    # Logical peak counting can free inactive prefixes sooner than this pool.
    allocations = (5, 6, 7, 8, 9)
    expected = reference_arithmetic_gates(
        original,
        {"control": (0,), "selection": (1, 2), "x": (3, 4)},
        allocations,
        {5: 0, 6: 1, 9: 2},
        cirq_style=True,
        preserve_components=True,
        reuse_allocations=not adjoint,
    )
    assert_source_dependencies(make_gates(adjoint=adjoint), expected)
