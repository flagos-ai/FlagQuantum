"""Unknown-data QROM factories retain original symbolic metadata and declarations."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest
import sympy

from flagquantum.compiler.registers.lookup.qrom_table import QromTable
from flagquantum.compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def dimensions(mode):
    n, m, b, c = sympy.symbols("n m b c", integer=True, positive=True)
    if mode == "scalar":
        return n, b, (), (), c
    if mode == "grouped":
        return (m, n), (b, 3), ((2,), (m,)), (), c
    if mode == "explicit":
        return (m, n), (b, 3), ((), (2,)), (m, b), 2
    if mode == "symbolic_literals":
        return sympy.Integer(4), (sympy.Integer(2),), (), (), sympy.Integer(1)
    if mode == "singleton":
        return (1, n), (b,), (), (), 0
    if mode == "zero_target":
        return n, (0, b), ((), (2,)), (), 0
    if mode == "one_row":
        return 1, 2, (), (), 0
    if mode == "rank_zero":
        return (), 2, (), (), 0
    raise AssertionError(mode)


@pytest.mark.parametrize(
    "mode,legacy",
    tuple(
        product(
            (
                "scalar",
                "grouped",
                "explicit",
                "symbolic_literals",
                "singleton",
                "zero_target",
                "one_row",
                "rank_zero",
            ),
            (False, True),
        )
    ),
)
def test_original_shape_factory_and_all_resource_categories(mode, legacy):
    from qualtran.bloqs.data_loading.qrom import QROM
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    shape, widths, targets, selection, controls = dimensions(mode)
    native = QromTable.from_shape(
        shape,
        widths,
        target_shapes=targets,
        selection_bitsizes=selection,
        num_controls=controls,
    )
    source = QROM.build_from_bitsize(
        shape,
        widths,
        target_shapes=targets,
        selection_bitsizes=selection,
        num_controls=controls,
    )
    assert source.is_symbolic() and not source.has_data()
    assert native.datasets is None
    assert native.shape == source.data_shape
    assert native.selection_bitsizes == source.selection_bitsizes
    assert native.target_bitsizes == source.target_bitsizes
    assert native.target_shapes == source.target_shapes
    assert native.num_controls == source.num_controls
    assert native.target_layout == tuple(
        (sympy.prod(target), width)
        for target, width in zip(
            source.target_shapes, source.target_bitsizes, strict=True
        )
    )
    formula = native.resource_formula(legacy_shims=legacy)
    expected = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
    for name in (
        "and_bloq",
        "clifford",
        "measurement",
        "rotation",
        "cswap",
        "toffoli",
        "t",
    ):
        assert sympy.simplify(getattr(formula, name) - getattr(expected, name)) == 0
    assert (
        sympy.simplify(formula.logical_qubits - get_cost_value(source, QubitCount()))
        == 0
    )
    if mode in ("one_row", "rank_zero"):
        # Preserve the original declaration, including its nonphysical negative count.
        assert formula.and_bloq == formula.measurement == -1
    with pytest.raises(ValueError, match="unavailable"):
        native.resource_model()
    with pytest.raises(ValueError, match="unavailable"):
        native.bind((), (), (), (), ())


def test_shared_formula_is_the_only_shape_cost_authority(monkeypatch):
    import flagquantum.compiler.registers.lookup.qrom_table as module

    n, b, c = sympy.symbols("n b c", integer=True, positive=True)
    table = QromTable.from_shape(n, b, num_controls=c)
    declaration = SymbolicResourceFormula(13, 17, 19, 23)
    calls = []

    def replacement(shape, widths, **kwargs):
        calls.append((shape, widths, kwargs))
        return declaration

    monkeypatch.setattr(module, "source_qrom_resource_formula", replacement)
    assert table.resource_formula(legacy_shims=True) is declaration
    assert calls == [
        (
            (n,),
            (b,),
            dict(
                target_shapes=((),),
                control_count=c,
                selection_widths=table.selection_bitsizes,
                legacy_shims=True,
            ),
        )
    ]


def test_bound_dimensions_keep_shape_formula_separate_from_data_pruning():
    from qualtran.bloqs.data_loading.qrom import QROM
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    n, b = sympy.symbols("n b", integer=True, positive=True)
    shape = QromTable.from_shape(n, b)
    with pytest.raises(ValueError, match="preserve every declared shape"):
        shape.with_data([0, 0, 0, 0])
    # Resolving metadata is explicit and does not change symbolic literal dispatch.
    bound = replace(shape, shape=(4,), selection_bitsizes=(2,), target_bitsizes=(2,))
    loaded = bound.with_data([0, 0, 0, 0])
    assert loaded.resource_formula() == bound.resource_formula()
    assert loaded.resource_model().gates.and_bloq == 0
    assert bound.resource_formula().and_bloq == 2
    source = QROM.build_from_data(np.zeros(4, dtype=int), target_bitsizes=(2,))
    assert (
        loaded.resource_model().gates.and_bloq
        == get_cost_value(source, QECGatesCost()).and_bloq
    )
    assert loaded.bind((), ((0, 1),), (((2, 3),),), (), ()) == ()
    # A sympy integer is retained as a symbolic expression, never cast for emission.
    literal = QromTable.from_shape(sympy.Integer(4), sympy.Integer(2))
    assert type(literal.shape[0]) is sympy.Integer
    literal_data = literal.with_data([0, 0, 0, 0])
    original_literal = QROM.build_from_bitsize(
        sympy.Integer(4), sympy.Integer(2)
    ).with_data([0, 0, 0, 0])
    assert literal_data.datasets == tuple(
        tuple(map(int, a.flat)) for a in original_literal.data
    )
    assert type(literal_data.shape[0]) is sympy.Integer
    with pytest.raises(ValueError, match="integer dimensions"):
        literal_data.bind((), ((0, 1),), (((2, 3),),), (), ())


@pytest.mark.parametrize("invalid", [True, -1, 1.5, "n", None])
def test_unknown_data_dimensions_reject_non_symbolic_invalid_inputs(invalid):
    with pytest.raises(ValueError):
        QromTable.from_shape((4,), (2,), num_controls=invalid)


@pytest.mark.parametrize("legacy", [None, 0, 1, "legacy"])
def test_shape_profile_validation_remains_shared(legacy):
    with pytest.raises(ValueError):
        QromTable.from_shape(4, 2).resource_formula(legacy_shims=legacy)


@pytest.mark.parametrize("control_count", [0, 1])
def test_resolved_shape_rejoins_original_gate_and_all_complex_measurement_branches(
    control_count,
):
    import torch
    from qualtran.bloqs.data_loading.qrom import QROM

    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_register_subtraction import assert_source
    from tests.unit.test_source_qrom import source_qrom_gates

    n, b, c = sympy.symbols("n b c", integer=True, positive=True)
    shape = QromTable.from_shape(n, b, num_controls=c)
    resolved = replace(
        shape,
        shape=(4,),
        selection_bitsizes=(2,),
        target_bitsizes=(2,),
        num_controls=control_count,
    )
    values = [1, 3, 0, 2]
    table = resolved.with_data(values)
    source = QROM.build_from_bitsize(4, 2, num_controls=control_count).with_data(values)
    controls = tuple(range(control_count))
    selections = (tuple(range(control_count, control_count + 2)),)
    targets = ((tuple(range(control_count + 2, control_count + 4)),),)
    work = tuple(range(control_count + 4, control_count + 4 + table.workspace_size))
    slots = tuple(range(len(work)))
    gates = table.bind(controls, selections, targets, work, slots)
    ports = {"selection": selections[0], "target0_": targets[0][0]}
    if controls:
        ports["control"] = controls
    assert_source(gates, source_qrom_gates(source, ports, work, slots))
    events = sum(g.name == "measure" for g in gates)
    assert table.resource_model().gates.measurement == events
    assert sum(g.name in {"t", "tdg"} for g in gates) == 4 * events
    qubits = control_count + 4 + len(work)
    state = torch.zeros(1 << qubits, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    # Include all target words and the coherent inactive control subspace.
    size = 1 << (control_count + 4)
    weights = torch.arange(1, size + 1, dtype=torch.float64)
    coefficients = torch.complex(weights, weights % 5 - 2)
    coefficients /= coefficients.norm()
    for value in range(size):
        control, row, target = value >> 4, (value >> 2) & 3, value & 3
        mask = values[row] if control == (1 << control_count) - 1 else 0
        state[value << len(work)] = coefficients[value]
        expected[((value & ~3) | (target ^ mask)) << len(work)] = coefficients[value]
    for outcomes in product((0, 1), repeat=events):
        actual = forced_branch(state, gates, outcomes)
        torch.testing.assert_close(
            actual, expected * 2 ** (-events / 2), atol=2e-12, rtol=0
        )
