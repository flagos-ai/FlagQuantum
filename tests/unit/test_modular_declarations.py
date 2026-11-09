"""Original placeholder signatures, direct calls and model-only resource evidence."""

from itertools import product

import pytest

from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
from flagquantum.compiler.registers.modular.modular_declarations import (
    modular_arithmetic_calls,
    modular_arithmetic_registers,
    source_modular_arithmetic_resource_formula,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def original(kind, width, signed=False):
    from qualtran.bloqs.arithmetic._shims import CHalf, CSub, Lt, MultiCToffoli

    if kind == "multi_controlled_x":
        return MultiCToffoli(width)
    if kind == "controlled_subtraction":
        return CSub(width)
    if kind == "less_than":
        return Lt(width, signed=signed)
    return CHalf(width)


@pytest.mark.parametrize(
    "kind,width,legacy",
    tuple(
        product(
            (
                "multi_controlled_x",
                "controlled_subtraction",
                "less_than",
                "controlled_half",
            ),
            (0, 1, 2, 3, 8),
            (False, True),
        )
    ),
)
def test_original_registers_direct_calls_categories_and_fallback_peak(
    kind, width, legacy
):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = original(kind, width)
    assert modular_arithmetic_registers(kind, width) == tuple(
        (r.name, r.total_bits()) for r in source.signature
    )
    calls = modular_arithmetic_calls(kind, width, legacy_shims=legacy)
    expected_calls = source.build_call_graph(None)
    if kind == "controlled_subtraction":
        from qualtran import QMontgomeryUInt
        from qualtran.bloqs.arithmetic.bitwise import BitwiseNot
        from qualtran.bloqs.arithmetic.controlled_addition import CAdd

        add, invert = calls
        assert add == ArithmeticCall(
            "controlled_add", (width, width), ("montgomery", "montgomery")
        )
        assert invert == ArithmeticCall("bitwise_not", (width,), ("montgomery",))
        assert list(calls.values()) == [1, 3]
        assert list(expected_calls.items()) == [
            (
                CAdd(
                    QMontgomeryUInt(add.widths[0]),
                    QMontgomeryUInt(add.widths[1]),
                    cv=add.control_value,
                ),
                1,
            ),
            (BitwiseNot(QMontgomeryUInt(invert.widths[0])), 3),
        ]
    else:
        from qualtran.bloqs.basic_gates import Toffoli
        from qualtran.bloqs.basic_gates.swap import TwoBitCSwap

        actual = {
            Toffoli() if child.opcode == "toffoli" else TwoBitCSwap(): count
            for child, count in calls.items()
        }
        assert list(actual.items()) == list(expected_calls.items())
    if kind == "controlled_subtraction" and width == 0:
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            get_cost_value(source, QECGatesCost(legacy_shims=legacy))
        with pytest.raises((TypeError, RuntimeError)):
            get_cost_value(source, QubitCount())
        with pytest.raises(ValueError, match="zero-width CSub"):
            source_modular_arithmetic_resource_formula(kind, width, legacy_shims=legacy)
        return
    formula = source_modular_arithmetic_resource_formula(
        kind, width, legacy_shims=legacy
    )
    for operation in (source, source.adjoint()):
        cost = get_cost_value(operation, QECGatesCost(legacy_shims=legacy))
        for name in (
            "t",
            "toffoli",
            "cswap",
            "and_bloq",
            "clifford",
            "rotation",
            "measurement",
        ):
            assert getattr(formula, name) == getattr(cost, name)
        assert formula.logical_qubits == get_cost_value(operation, QubitCount())
    if kind == "multi_controlled_x" and width < 3:
        assert formula.toffoli == width - 2
        assert formula.logical_qubits == 3
    if kind in ("less_than", "controlled_half") and width == 0:
        assert formula.logical_qubits == 3


@pytest.mark.parametrize("width,legacy", tuple(product((0, 1, 2, 5), (False, True))))
def test_source_signed_flag_does_not_change_unsigned_signature_or_counts(width, legacy):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    signed = original("less_than", width, signed=True)
    formula = source_modular_arithmetic_resource_formula(
        "less_than", width, signed=True, legacy_shims=legacy
    )
    assert formula == source_modular_arithmetic_resource_formula(
        "less_than", width, legacy_shims=legacy
    )
    assert (
        formula.toffoli
        == get_cost_value(signed, QECGatesCost(legacy_shims=legacy)).toffoli
    )
    assert formula.logical_qubits == get_cost_value(signed, QubitCount())


@pytest.mark.parametrize(
    "kind",
    ["multi_controlled_x", "controlled_subtraction", "less_than", "controlled_half"],
)
def test_original_placeholders_do_not_have_gate_or_classical_decompositions(kind):
    from qualtran import DecomposeNotImplementedError

    source = original(kind, 3)
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    assert source.on_classical_vals() is NotImplemented
    # Native output is an explicit formula, never an appendable operation.
    formula = source_modular_arithmetic_resource_formula(kind, 3)
    assert not hasattr(formula, "append_to")


@pytest.mark.parametrize("legacy", [False, True])
def test_csub_consumes_shared_controlled_add_model(monkeypatch, legacy):
    import flagquantum.compiler.registers.modular.modular_declarations as module
    from flagquantum.compiler._component_trace.resource_model import (
        ComponentCost,
        GateCost,
    )

    replacement = ComponentCost("CAdd", 29, GateCost(t=13))
    calls = []

    def shared(m, n, *, legacy_shims=False):
        calls.append((m, n, legacy_shims))
        return replacement

    monkeypatch.setattr(module, "source_controlled_add_cost", shared)
    formula = module.source_modular_arithmetic_resource_formula(
        "controlled_subtraction", 3, legacy_shims=legacy
    )
    assert calls == [(3, 3, legacy)]
    assert formula.t == 13 and formula.clifford == 9 and formula.logical_qubits == 29


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": -1},
        {"width": True},
        {"kind": "unknown"},
        {"signed": 1},
        {"kind": "controlled_half", "signed": True},
        {"legacy_shims": 0},
    ],
)
def test_native_declaration_options_are_explicit(kwargs):
    options = {"kind": "less_than", "width": 3} | kwargs
    with pytest.raises(ValueError):
        source_modular_arithmetic_resource_formula(**options)
