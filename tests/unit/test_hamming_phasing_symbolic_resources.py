"""Source-positive symbolic Hamming cost queries retain exact classification."""

from dataclasses import fields
from typing import Any

import pytest
import sympy

from flagquantum.compiler._component_trace.qec_gate_cost import QECGateCost
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_hamming_phase_factories import (
    hamming_phase_factories,
)
from flagquantum.compiler.registers.phase.hamming_weight_phasing import (
    source_hamming_weight_gradient_phasing_cost,
    source_hamming_weight_phasing_cost,
    source_symbolic_hamming_weight_gradient_phasing_cost,
)
from flagquantum.compiler.registers.phase.phase_gradient import (
    source_phase_gradient_state_cost,
    source_phase_gradient_unitary_cost,
    source_symbolic_phase_gradient_cost,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.errors import SerializationError
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_register_component_calls import SourceCalls
from tests.unit.test_subtraction_foreign_file import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_THETA = sympy.Symbol("theta", real=True)
_EPS = sympy.Symbol("eps", positive=True)


def _assert_gates(actual: Any, expected: Any) -> None:
    for field in fields(QECGateCost):
        assert (
            sympy.simplify(getattr(actual, field.name) - getattr(expected, field.name))
            == 0
        )


@pytest.mark.parametrize("legacy_shims", (False, True))
@pytest.mark.parametrize("width", (1, 2, 4, 8))
@pytest.mark.parametrize(
    "exponent",
    (
        _THETA,
        2 * _THETA + 1,
        sympy.Integer(0),
        sympy.Rational(1, 4),
        0,
        0.25,
        -0.5,
        0.375,
    ),
)
def test_direct_whole_source_categories_with_literal_and_symbolic_exponents(
    width: int, exponent: Any, legacy_shims: bool
) -> None:
    from qualtran.bloqs.rotations.hamming_weight_phasing import HammingWeightPhasing
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = HammingWeightPhasing(width, exponent)
    native = estimate_component_resources(
        source_hamming_weight_phasing_cost(width, exponent, legacy_shims=legacy_shims)
    )
    _assert_gates(
        native.gates, get_cost_value(source, QECGatesCost(legacy_shims=legacy_shims))
    )
    if width == 1:
        with pytest.raises(RuntimeError, match="QAny\\(bitsize=0\\)"):
            get_cost_value(source, QubitCount())
        assert native.peak_logical_qubits == 2
    else:
        assert native.peak_logical_qubits == get_cost_value(source, QubitCount())


@pytest.mark.parametrize("legacy_shims", (False, True))
@pytest.mark.parametrize("width", (1, 2, 4))
@pytest.mark.parametrize("epsilon", (0.2, 1e-10, _EPS))
@pytest.mark.parametrize("exponent", (_THETA, sympy.Integer(0), sympy.Rational(1, 4)))
def test_gradient_symbolic_qvr_composition_and_exact_source_width_disposition(
    width: int, epsilon: Any, legacy_shims: bool, exponent: Any
) -> None:
    from qualtran.bloqs.rotations.hamming_weight_phasing import (
        HammingWeightPhasingViaPhaseGradient,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = HammingWeightPhasingViaPhaseGradient(width, exponent, epsilon)
    native = source_symbolic_hamming_weight_gradient_phasing_cost(
        width, exponent, epsilon, legacy_shims=legacy_shims
    )
    _assert_gates(
        native, get_cost_value(source, QECGatesCost(legacy_shims=legacy_shims))
    )
    if width == 1:
        with pytest.raises(RuntimeError, match="QAny\\(bitsize=0\\)"):
            get_cost_value(source, QubitCount())
    elif isinstance(epsilon, sympy.Expr):
        assert (
            sympy.simplify(native.logical_qubits - get_cost_value(source, QubitCount()))
            == 0
        )
    else:
        with pytest.raises(
            RuntimeError, match=f"Symbolic gamma {exponent / 2} not allowed"
        ):
            get_cost_value(source, QubitCount())
        assert (
            native.logical_qubits
            == width + (width - width.bit_count()) + width.bit_length() + source.b_grad
        )


@pytest.mark.parametrize("controlled", (False, True))
@pytest.mark.parametrize(
    "exponent", (0, sympy.Integer(0), sympy.Rational(1, 4), _THETA)
)
def test_original_symbolic_gradient_classification_is_not_algebraically_optimized(
    controlled: bool, exponent: Any
) -> None:
    from qualtran.bloqs.rotations.phase_gradient import PhaseGradientUnitary
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    width = sympy.Symbol("n")
    source = PhaseGradientUnitary(width, exponent=exponent, is_controlled=controlled)
    expected = get_cost_value(source, QECGatesCost())
    # Even mathematical zero is a SymPy expression after division by 2**n.
    # The unchanged source classifies that parameterized leaf as a rotation.
    assert expected.rotation == width
    _assert_gates(
        source_symbolic_phase_gradient_cost(width, controlled=controlled), expected
    )


@pytest.mark.parametrize("width", (2, 4))
@pytest.mark.parametrize("legacy_shims", (False, True))
def test_numeric_gradient_default_costs_retain_source_profile(
    width: int, legacy_shims: bool
) -> None:
    from qualtran.bloqs.rotations.hamming_weight_phasing import (
        HammingWeightPhasingViaPhaseGradient,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = HammingWeightPhasingViaPhaseGradient(width, -1.1934341, 0.2)
    native = estimate_component_resources(
        source_hamming_weight_gradient_phasing_cost(
            width, -1.1934341, 0.2, legacy_shims=legacy_shims
        )
    )
    _assert_gates(
        native.gates, get_cost_value(source, QECGatesCost(legacy_shims=legacy_shims))
    )


@pytest.mark.parametrize("kind", ("state", "unitary", "controlled"))
@pytest.mark.parametrize("width", (1, 4))
@pytest.mark.parametrize(
    "exponent", (_THETA, 2 * _THETA + 1, sympy.Integer(0), sympy.Rational(1, 4))
)
def test_finite_gradient_and_state_models_accept_original_symbolic_exponents(
    kind: str, width: int, exponent: Any
) -> None:
    from qualtran.bloqs.rotations.phase_gradient import (
        PhaseGradientState,
        PhaseGradientUnitary,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    if kind == "state":
        source = PhaseGradientState(width, exponent=exponent)
        model = source_phase_gradient_state_cost(width, exponent)
    else:
        source = PhaseGradientUnitary(
            width, exponent=exponent, is_controlled=kind == "controlled"
        )
        model = source_phase_gradient_unitary_cost(
            width, exponent, controlled=kind == "controlled"
        )
    actual = estimate_component_resources(model)
    _assert_gates(actual.gates, get_cost_value(source, QECGatesCost()))
    assert actual.peak_logical_qubits == get_cost_value(source, QubitCount())


def test_sdk_free_queries_and_nonfinite_numeric_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    expected = source_symbolic_hamming_weight_gradient_phasing_cost(4, _THETA, _EPS)
    original = builtins.__import__

    def no_reference(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq"}:
            raise ImportError(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_reference)
    assert (
        source_symbolic_hamming_weight_gradient_phasing_cost(4, _THETA, _EPS)
        == expected
    )
    assert (
        estimate_component_resources(
            source_hamming_weight_phasing_cost(4, _THETA)
        ).gates.rotation
        == 3
    )
    for exponent in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="finite exponent"):
            source_hamming_weight_phasing_cost(4, exponent)


@pytest.mark.parametrize("width", (1, 2, 4))
@pytest.mark.parametrize("epsilon", (0.2, _EPS, 2 * _EPS))
def test_symbolic_gradient_hamming_preserves_root_and_qvr_source_call_layers(
    width: int, epsilon: Any
) -> None:
    from qualtran.bloqs.rotations.hamming_weight_phasing import (
        HammingWeightPhasingViaPhaseGradient,
    )

    source = HammingWeightPhasingViaPhaseGradient(width, _THETA, epsilon)
    library = _library(source, 0)
    program = resolve_component_program(library, 0, hamming_phase_factories())
    assert verify_program_record(program) == ()
    assert _native_trace(program) == _source_trace(SourceCalls, source.decompose_bloq())
    root = program.functions[0]
    assert len(root.blocks[0].operations) == 3
    qvr_name = root.blocks[0].operations[1].callee
    assert qvr_name is not None
    qvr = next(function for function in program.functions if function.name == qvr_name)
    assert qvr.registers == tuple(native_signature(source.phase_oracle.signature))
    assert len(qvr.blocks[0].operations) == 1
    assert _native_trace(program, qvr_name) == _source_trace(
        SourceCalls, source.phase_oracle.decompose_bloq()
    )
    scaled_name = qvr.blocks[0].operations[0].callee
    scaled = next(
        function for function in program.functions if function.name == scaled_name
    )
    assert not scaled.blocks
    source_scaled = next(iter(source.phase_oracle.decompose_bloq().bloq_instances)).bloq
    assert scaled.registers == tuple(native_signature(source_scaled.signature))
    with pytest.raises(ValueError):
        bind_register_program(program, ())


def test_numeric_coefficient_symbolic_precision_retains_original_refusal() -> None:
    from qualtran.bloqs.rotations.hamming_weight_phasing import (
        HammingWeightPhasingViaPhaseGradient,
    )

    source = HammingWeightPhasingViaPhaseGradient(4, 1.2, _EPS)
    with pytest.raises(ValueError, match="Fxp template for symbolic bitsizes"):
        source.decompose_bloq()
    factory = hamming_phase_factories()[
        type(source).__module__ + "." + type(source).__name__
    ]
    with pytest.raises(SerializationError, match="concrete width and precision"):
        factory("rejected", None, {"bitsize": 4, "exponent": 1.2, "eps": _EPS})
