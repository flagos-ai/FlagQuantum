"""Source QROM signatures map to existing FQ partition layouts and packed ports."""

from itertools import product

import pytest
import sympy

from flagquantum.compiler.registers.lookup.qrom_table import QromTable

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def shape_case(case):
    n, m, b = sympy.symbols("n m b", positive=True, integer=True)
    cases = {
        "scalar": (4, 2, ()),
        "array": ((2, 3), (1, 2), ((), (2, 2))),
        "leading_singleton": ((1, 3), 2, ()),
        "trailing_singleton": ((3, 1), 2, ()),
        "middle_singleton": ((2, 1, 3), 2, ()),
        "only_singletons": ((1, 1), (1, 2), ()),
        "rank_zero": ((), 2, ()),
        "zero_width": (4, (0, 2), ()),
        "symbolic": ((m, n), (b, 2), ((m,), ())),
        "symbolic_one": ((sympy.Integer(1), n), 2, ()),
        "symbolic_zero_width": (4, (sympy.Integer(0), 2), ()),
    }
    return cases[case]


@pytest.mark.parametrize(
    "case,controls",
    tuple(
        product(
            (
                "scalar",
                "array",
                "leading_singleton",
                "trailing_singleton",
                "middle_singleton",
                "only_singletons",
                "rank_zero",
                "zero_width",
                "symbolic",
                "symbolic_one",
                "symbolic_zero_width",
            ),
            (0, 2),
        )
    ),
)
def test_original_control_selection_target_names_and_flat_widths(case, controls):
    from qualtran.bloqs.data_loading.qrom import QROM

    shape, widths, targets = shape_case(case)
    native = QromTable.from_shape(
        shape, widths, target_shapes=targets, num_controls=controls
    )
    source = QROM.build_from_bitsize(
        shape, widths, target_shapes=targets, num_controls=controls
    )
    assert native.register_layout == tuple(
        (reg.name, reg.total_bits()) for reg in source.signature
    )


def test_control_name_follows_original_symbolic_truth_dispatch():
    from qualtran.bloqs.data_loading.qrom import QROM

    c = sympy.Symbol("c", nonnegative=True, integer=True)
    for controls in (c, sympy.Integer(0), sympy.Integer(2)):
        native = QromTable.from_shape(4, 2, num_controls=controls)
        source = QROM.build_from_bitsize(4, 2, num_controls=controls)
        assert native.register_layout == tuple(
            (r.name, r.total_bits()) for r in source.signature
        )


def test_packed_binding_reuses_partition_validation_and_caller_work_ownership():
    table = QromTable.from_data([1, 0, 2, 3])
    with pytest.raises(ValueError, match="sum of partition widths"):
        table.bind_packed((0, 1, 2), (4,), (0,))
    with pytest.raises(ValueError):
        table.bind_packed((0, 1, 2, 2), (4,), (0,))
    with pytest.raises(ValueError):
        table.bind_packed((0, 1, 2, 3), (3,), (0,))
    with pytest.raises(ValueError, match="classical slot"):
        table.bind_packed((0, 1, 2, 3), (4,), ())
    symbolic = QromTable.from_shape(sympy.Symbol("n", positive=True, integer=True), 2)
    with pytest.raises(ValueError, match="unavailable"):
        symbolic.bind_packed((0, 1, 2, 3), (), ())


def test_layout_rejects_original_bounded_selection_type_failure():
    from qualtran.bloqs.data_loading.qrom import QROM

    native = QromTable.from_shape(5, 2, selection_bitsizes=(2,))
    source = QROM.build_from_bitsize(5, 2, selection_bitsizes=(2,))
    with pytest.raises(ValueError):
        _ = source.signature
    with pytest.raises(ValueError, match="fit the source axes"):
        _ = native.register_layout
