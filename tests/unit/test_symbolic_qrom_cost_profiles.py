"""Shape-only QROM declares positive And pairs in both original QEC profiles."""

from itertools import product

import pytest

from flagquantum.compiler.registers.lookup.qrom_table import QromTable
from flagquantum.compiler.registers.resources.symbolic_resource_model import (
    source_qrom_resource_formula,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "mode,legacy", tuple(product(("rank_one", "rank_two", "explicit"), (False, True)))
)
def test_original_unknown_data_call_graph_and_bound_shape_profiles(mode, legacy):
    import sympy
    from qualtran.bloqs.data_loading.qrom import QROM
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.symbolics import Shaped

    m, n, bits, controls = sympy.symbols("m n b c", positive=True, integer=True)
    if mode == "rank_one":
        shape, widths, targets, selection = (n,), (bits,), ((),), None
    else:
        shape, widths, targets = (m, n), (bits, 3), ((), (2,))
        selection = (m, bits) if mode == "explicit" else None

    def original(shape, widths, selection, controls):
        if selection is None:
            return QROM.build_from_bitsize(
                shape, widths, target_shapes=targets, num_controls=controls
            )
        return QROM(
            tuple(Shaped((*shape, *target)) for target in targets),
            selection,
            widths,
            targets,
            num_controls=controls,
        )

    def verify(shape, widths, selection, controls):
        formula = source_qrom_resource_formula(
            shape,
            widths,
            target_shapes=targets,
            selection_widths=selection,
            control_count=controls,
            legacy_shims=legacy,
        )
        table = QromTable.from_shape(
            shape,
            widths,
            target_shapes=targets,
            selection_bitsizes=() if selection is None else selection,
            num_controls=controls,
        )
        assert table.resource_formula(legacy_shims=legacy) == formula
        source = original(shape, widths, selection, controls)
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
            sympy.simplify(
                formula.logical_qubits - get_cost_value(source, QubitCount())
            )
            == 0
        )
        return formula

    formula = verify(shape, widths, selection, controls)
    substitutions = {m: 4, n: 5, bits: 4, controls: 2}

    def bind(values):
        return tuple(int(sympy.sympify(value).subs(substitutions)) for value in values)

    bound = verify(
        bind(shape), bind(widths), None if selection is None else bind(selection), 2
    )
    for name in ("and_bloq", "clifford", "measurement", "logical_qubits"):
        assert (
            sympy.simplify(
                getattr(formula, name).subs(substitutions) - getattr(bound, name)
            )
            == 0
        )


@pytest.mark.parametrize("value", [None, 0, 1, "legacy"])
def test_symbolic_qrom_profile_flag_requires_literal_boolean(value):
    with pytest.raises(ValueError):
        source_qrom_resource_formula((5,), (3,), legacy_shims=value)
