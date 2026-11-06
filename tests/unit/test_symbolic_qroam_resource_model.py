"""Compare symbolic blocked lookup formulas with the locked source models."""

import pytest

from flagquantum.compiler.registers.qroam_resource_model import (
    source_qroam_measurement_erase_cost,
)
from flagquantum.compiler.registers.source_qroam_clean import QroamCleanTable
from flagquantum.compiler.registers.symbolic_qroam_resource_model import (
    source_blocked_lookup_resource_formula,
    source_shape_erasure_resource_formula,
)
from flagquantum.compiler.registers.symbolic_resource_model import (
    SymbolicResourceFormula,
)

pytestmark = pytest.mark.unit
GATE_FIELDS = ("and_bloq", "clifford", "measurement", "t", "toffoli")


def assert_source_cost(formula: SymbolicResourceFormula, source: object) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from sympy import simplify

    gates = get_cost_value(source, QECGatesCost())
    for field in GATE_FIELDS:
        assert simplify(getattr(formula, field) - getattr(gates, field)) == 0, field
    assert simplify(formula.logical_qubits - get_cost_value(source, QubitCount())) == 0


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("dirty", [None, False, True])
@pytest.mark.parametrize("rank", [1, 2])
def test_blocked_shape_formula_and_substitution(dirty: bool | None, rank: int) -> None:
    pytest.importorskip("qualtran")
    sympy = pytest.importorskip("sympy")
    from qualtran.bloqs.data_loading.qroam_clean import QROAMClean
    from qualtran.bloqs.data_loading.select_swap_qrom import SelectSwapQROM

    n, m, width, exponent, controls = sympy.symbols(
        "n m b k c", positive=True, integer=True
    )
    if rank == 1:
        shape, widths, logs, selections = (n,), (width,), (exponent,), None
    else:
        shape, widths, logs, selections = (
            (n, m),
            (width, 2),
            (exponent, 1),
            (exponent + 3, 4),
        )
    kwargs = dict(
        data_len_or_shape=shape,
        target_bitsizes=widths,
        log_block_sizes=logs,
        num_controls=controls,
    )
    if selections is not None:
        kwargs["selection_bitsizes"] = selections
    source = (
        QROAMClean.build_from_bitsize(**kwargs)
        if dirty is None
        else SelectSwapQROM.build_from_bitsize(**kwargs, use_dirty_ancilla=dirty)
    )
    actual = source_blocked_lookup_resource_formula(
        shape,
        widths,
        logs,
        control_count=controls,
        selection_widths=selections,
        dirty=dirty,
    )
    assert_source_cost(actual, source)

    values = {n: 9, m: 6, width: 3, exponent: 2, controls: 1}

    def bind(seq: tuple) -> tuple[int, ...]:
        return tuple(int(sympy.sympify(item).subs(values)) for item in seq)

    bound = source_blocked_lookup_resource_formula(
        bind(shape),
        bind(widths),
        bind(logs),
        control_count=1,
        selection_widths=None if selections is None else bind(selections),
        dirty=dirty,
    )
    for field in (*GATE_FIELDS, "logical_qubits"):
        assert int(sympy.sympify(getattr(actual, field)).subs(values)) == getattr(
            bound, field
        )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("rank", [1, 2])
def test_shape_erasure_wrapper_and_leaf(explicit: bool, rank: int) -> None:
    pytest.importorskip("qualtran")
    sympy = pytest.importorskip("sympy")
    from qualtran.bloqs.data_loading.qroam_clean import QROAMClean

    n, m, width, exponent, controls = sympy.symbols(
        "n m b k c", positive=True, integer=True
    )
    shape = (n,) if rank == 1 else (n, m)
    widths = (width,) if rank == 1 else (width, 2)
    logs = (exponent,) if rank == 1 else (exponent, 1)
    erase_logs = (0,) * rank if explicit else None
    source = QROAMClean.build_from_bitsize(
        shape, widths, log_block_sizes=logs, num_controls=controls
    ).adjoint()
    if erase_logs is not None:
        source = source.with_log_block_sizes(erase_logs)
    actual = source_shape_erasure_resource_formula(
        shape, widths, logs, control_count=controls, erase_logs=erase_logs
    )
    assert_source_cost(actual.wrapper, source)
    assert_source_cost(actual.leaf, source.qroam_clean_adjoint_bloq)


def test_shape_erasure_keeps_original_rows_without_padding() -> None:
    declared = source_shape_erasure_resource_formula(
        (5,), (2,), (1,), control_count=1, erase_logs=(0,)
    )
    with_data = source_qroam_measurement_erase_cost(
        QroamCleanTable((5,), (1,), ((1, 2, 3, 0, 1),)),
        1,
        (2,),
        erase_logs=(0,),
    )
    assert declared.wrapper.toffoli == with_data.wrapper.gates.toffoli == 3
    assert with_data.padded_leaf.gates.toffoli == 4
    assert declared.wrapper.logical_qubits == with_data.wrapper.qubits == 8


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("dirty", [None, False, True])
def test_source_symbolic_forward_block_selection(dirty: bool | None) -> None:
    pytest.importorskip("qualtran")
    sympy = pytest.importorskip("sympy")
    from qualtran.bloqs.data_loading.qroam_clean import QROAMClean
    from qualtran.bloqs.data_loading.select_swap_qrom import SelectSwapQROM

    n, b = sympy.symbols("n b", positive=True, integer=True)
    source = (
        QROAMClean.build_from_bitsize((n,), (b, 2), num_controls=1)
        if dirty is None
        else SelectSwapQROM.build_from_bitsize(
            (n,), (b, 2), num_controls=1, use_dirty_ancilla=dirty
        )
    )
    assert_source_cost(
        source_blocked_lookup_resource_formula(
            (n,), (b, 2), control_count=1, dirty=dirty
        ),
        source,
    )
    if dirty is None:
        assert_source_cost(
            source_shape_erasure_resource_formula(
                (n,), (b, 2), control_count=1
            ).wrapper,
            source.adjoint(),
        )
