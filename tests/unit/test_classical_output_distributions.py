"""Scalar output distributions resolve before original integer binding checks."""

import networkx as nx
import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_distributions import (
    ClassicalDistribution,
    draw_classical_value,
)
from flagquantum.simulation.classical_registers import bind_integer_register_values
from tests.unit.test_classical_comparison import same_value

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def original_assign(values, width, *, signed=False, shape=(), rng=None, fixed=None):
    from qualtran import BloqInstance, QInt, QUInt, Register, Side, Signature
    from qualtran.bloqs.basic_gates import Identity
    from qualtran.simulation.classical_sim import (
        ClassicalSimState,
        _BannedClassicalValHandler,
        _FixedClassicalValHandler,
        _RandomClassicalValHandler,
    )

    handler = (
        _RandomClassicalValHandler(rng)
        if rng is not None
        else (
            _FixedClassicalValHandler(fixed)
            if fixed is not None
            else _BannedClassicalValHandler()
        )
    )
    state = ClassicalSimState(Signature([]), nx.DiGraph(), {}, random_handler=handler)
    port = Register(
        "x", (QInt if signed else QUInt)(width), shape=shape, side=Side.RIGHT
    )
    instance = BloqInstance(Identity(1), 7)
    state._update_assign_from_vals((port,), instance, values)
    return {soquet.idx: value for soquet, value in state.soq_assign.items()}


@pytest.mark.parametrize(
    "fixed",
    [
        0,
        1,
        -1,
        2,
        np.int8(1),
        True,
        1.0,
        np.array(1),
        None,
        sympy.Integer(1),
        sympy.Symbol("q"),
        ClassicalDistribution(2),
    ],
)
@pytest.mark.parametrize("signed", [False, True])
def test_fixed_results_are_sampled_once_then_validated_without_symbolic_bypass(
    fixed, signed
):
    from qualtran.simulation.classical_sim import ClassicalValDistribution

    source = ClassicalValDistribution(-1, [4, -3])
    native = ClassicalDistribution(-1, [4, -3])
    seen = []

    def resolve(distribution):
        seen.append(distribution)
        return draw_classical_value(
            distribution, instance_id=7, fixed_values={7: fixed}
        )

    try:
        expected = original_assign({"x": source}, 2, signed=signed, fixed={7: fixed})
    except Exception as error:
        with pytest.raises(type(error)):
            bind_integer_register_values(
                "x", {"x": native}, 2, signed=signed, resolve_distribution=resolve
            )
    else:
        actual = bind_integer_register_values(
            "x", {"x": native}, 2, signed=signed, resolve_distribution=resolve
        )
        assert actual.keys() == expected.keys()
        assert actual[()] is expected[()] is fixed
        same_value(actual[()], expected[()])
    assert seen == [native]


@pytest.mark.parametrize("seed", [0, 131])
@pytest.mark.parametrize("signed", [False, True])
def test_seeded_output_binding_matches_source_values_types_and_rng_state(seed, signed):
    from qualtran.simulation.classical_sim import ClassicalValDistribution

    choices = np.array([-2, -1, 0, 1] if signed else [0, 1, 2, 3])
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])
    source, native = ClassicalValDistribution(
        choices, probabilities
    ), ClassicalDistribution(choices, probabilities)
    source_rng, native_rng = np.random.default_rng(seed), np.random.default_rng(seed)
    for _ in range(16):
        expected = original_assign({"x": source}, 2, signed=signed, rng=source_rng)
        actual = bind_integer_register_values(
            "x",
            {"x": native},
            2,
            signed=signed,
            resolve_distribution=lambda d: draw_classical_value(
                d, instance_id=7, rng=native_rng
            ),
        )
        same_value(actual[()], expected[()])
    assert source_rng.bit_generator.state == native_rng.bit_generator.state


@pytest.mark.parametrize("shape", [(1,), (2,), (0,), (1, 1)])
@pytest.mark.parametrize("array_value", [False, True])
def test_shaped_output_never_samples_a_distribution(shape, array_value):
    from qualtran.simulation.classical_sim import ClassicalValDistribution

    source, native = ClassicalValDistribution(2), ClassicalDistribution(2)
    if array_value:
        source, native = np.array([source], dtype=object), np.array(
            [native], dtype=object
        )
    source_rng = np.random.default_rng(0)
    before = source_rng.bit_generator.state
    with pytest.raises((ValueError, TypeError)) as expected:
        original_assign({"x": source}, 1, shape=shape, rng=source_rng)
    with pytest.raises(type(expected.value)):
        bind_integer_register_values(
            "x",
            {"x": native},
            1,
            shape=shape,
            resolve_distribution=lambda d: pytest.fail(
                "Arrays must not sample distributions"
            ),
        )
    assert source_rng.bit_generator.state == before


def test_ordinary_symbolic_value_bypasses_sampling_but_sampled_symbolic_value_does_not():
    value = sympy.Integer(99)
    source = original_assign({"x": value}, 1)
    native = bind_integer_register_values(
        "x",
        {"x": value},
        1,
        resolve_distribution=lambda d: pytest.fail(
            "Existing symbolic value must bypass resolver"
        ),
    )
    assert source[()] is native[()] is value


def test_banned_sampling_and_missing_fixed_instance_match_original():
    from qualtran.simulation.classical_sim import ClassicalValDistribution

    source, native = ClassicalValDistribution(2), ClassicalDistribution(2)
    with pytest.raises(ValueError, match="non-deterministic"):
        original_assign({"x": source}, 1)
    with pytest.raises(ValueError, match="non-deterministic"):
        bind_integer_register_values("x", {"x": native}, 1)
    with pytest.raises(KeyError):
        original_assign({"x": source}, 1, fixed={})
    with pytest.raises(KeyError):
        bind_integer_register_values(
            "x",
            {"x": native},
            1,
            resolve_distribution=lambda d: draw_classical_value(
                d, instance_id=7, fixed_values={}
            ),
        )


def test_binding_calls_resolver_then_shared_validator_once_without_rechecking_symbolic(
    monkeypatch,
):
    import flagquantum.simulation.classical_registers as native

    distribution, value = ClassicalDistribution(2), sympy.Integer(1)
    seen = []

    def resolve(actual):
        assert actual is distribution
        seen.append("resolve")
        return value

    def validate(actual, width, *, signed):
        assert actual is value and width == 3 and signed is True
        seen.append("validate")

    monkeypatch.setattr(native, "validate_integer_word_value", validate)
    output = native.bind_integer_register_values(
        "x", {"x": distribution}, 3, signed=True, resolve_distribution=resolve
    )
    assert output[()] is value and seen == ["resolve", "validate"]


@pytest.mark.parametrize("value", [0, 1])
@pytest.mark.parametrize("seed", [0, 137])
def test_dispatch_sample_bind_collect_and_phase_match_original_measurement_leaf_graph(
    value, seed
):
    from qualtran.bloqs.basic_gates import MeasureX
    from qualtran.simulation.classical_sim import do_phased_classical_simulation

    from flagquantum.simulation.classical_action import evaluate_classical_action
    from flagquantum.simulation.classical_measurement import (
        apply_classical_phase,
        measure_x_basis_phase,
        measure_x_classical_values,
    )
    from flagquantum.simulation.classical_registers import (
        collect_integer_register_value,
    )

    source_rng, native_rng = np.random.default_rng(seed), np.random.default_rng(seed)
    for _ in range(16):
        expected, phase = do_phased_classical_simulation(
            MeasureX(), {"q": value}, rng=source_rng
        )
        outputs, contribution = evaluate_classical_action(
            {"q": value},
            lambda **v: measure_x_classical_values(v["q"]),
            lambda **v: measure_x_basis_phase(v["q"]),
            lambda v: pytest.fail("MeasureX has direct numerical action"),
            operation_name="MeasureX",
        )
        bound = bind_integer_register_values(
            "c",
            outputs,
            1,
            resolve_distribution=lambda d: draw_classical_value(
                d, instance_id=0, rng=native_rng
            ),
        )
        outcome = collect_integer_register_value(bound, 1)
        actual_phase = apply_classical_phase(1.0, contribution, {("c", ()): outcome})
        same_value(outcome, expected["c"])
        same_value(actual_phase, phase)
    assert source_rng.bit_generator.state == native_rng.bit_generator.state
