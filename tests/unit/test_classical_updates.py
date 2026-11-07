"""Output writes remain visible before original phase guards or phase failures."""

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_distributions import (
    ClassicalDistribution,
    draw_classical_value,
)
from flagquantum.simulation.classical_measurement import MeasurementOutcomePhase
from flagquantum.simulation.classical_registers import bind_integer_register_values
from flagquantum.simulation.classical_updates import (
    commit_classical_outputs,
    commit_phased_classical_outputs,
)
from tests.unit.test_classical_comparison import same_value

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_update(
    outputs, contribution, *, phased, current=1.0, shape=(), rng=None, fixed=None
):
    from qualtran import Bloq, QUInt, Register, Signature
    from qualtran.simulation.classical_sim import (
        ClassicalSimState,
        ClassicalValDistribution,
        MeasurementPhase,
        PhasedClassicalSimState,
        _BannedClassicalValHandler,
        _FixedClassicalValHandler,
        _RandomClassicalValHandler,
    )

    class OutputLeaf(Bloq):
        @property
        def signature(self):
            return Signature(
                [Register("x", QUInt(2), shape=shape), Register("y", QUInt(2))]
            )

    leaf = OutputLeaf()
    composite = leaf.as_composite_bloq()
    inputs = {"x": np.zeros(shape, dtype=int) if shape else 0, "y": 0}
    handler = (
        _RandomClassicalValHandler(rng)
        if rng is not None
        else (
            _FixedClassicalValHandler(fixed)
            if fixed is not None
            else _BannedClassicalValHandler()
        )
    )
    if phased:
        state = PhasedClassicalSimState(
            composite.signature, composite._binst_graph, inputs, random_handler=handler
        )
        # The original update does not revalidate this mutable accumulator.
        state.phase = current
    else:
        state = ClassicalSimState(
            composite.signature, composite._binst_graph, inputs, random_handler=handler
        )
    instance = next(iter(composite.bloq_instances))
    original_outputs = {
        name: (
            ClassicalValDistribution(value.choices, value.probabilities)
            if isinstance(value, ClassicalDistribution)
            else value
        )
        for name, value in outputs.items()
    }
    phase = (
        MeasurementPhase(contribution.register, contribution.index)
        if isinstance(contribution, MeasurementOutcomePhase)
        else contribution
    )
    error = None
    try:
        state._update(instance, original_outputs, phase)
    except Exception as caught:
        error = caught
    assigned = {
        (soquet.reg.name, soquet.idx): value
        for soquet, value in state.soq_assign.items()
        if soquet.binst == instance
    }
    return error, assigned, state.phase if phased else None


def native_update(
    outputs, contribution, *, phased, current=1.0, shape=(), rng=None, fixed=None
):
    assigned = {}

    def assign(values):
        assert values is outputs
        for name in ("x", "y"):
            bound = bind_integer_register_values(
                name,
                values,
                2,
                shape=shape if name == "x" else (),
                resolve_distribution=lambda d: draw_classical_value(
                    d, instance_id=0, rng=rng, fixed_values=fixed
                ),
            )
            assigned.update({(name, index): value for index, value in bound.items()})

    error, phase = None, current if phased else None
    try:
        if phased:
            phase = commit_phased_classical_outputs(
                outputs, contribution, assign, phase=phase, outcomes=assigned
            )
        else:
            commit_classical_outputs(
                outputs, contribution, assign, operation_name="OutputLeaf"
            )
    except Exception as caught:
        error = caught
    return error, assigned, phase


def compare_update(outputs, contribution, **options):
    expected_error, expected_values, expected_phase = source_update(
        outputs, contribution, **options
    )
    actual_error, actual_values, actual_phase = native_update(
        outputs, contribution, **options
    )
    assert type(actual_error) is type(expected_error)
    if expected_error is not None:
        assert type(actual_error.__cause__) is type(expected_error.__cause__)
    assert tuple(actual_values) == tuple(expected_values)
    for key in expected_values:
        same_value(actual_values[key], expected_values[key])
    same_value(actual_phase, expected_phase)
    return actual_error, actual_values, actual_phase


@pytest.mark.parametrize("phased", [False, True])
@pytest.mark.parametrize("contribution", [None, 0, 1, 1j, 2])
@pytest.mark.parametrize(
    "outputs",
    [
        {"x": 1, "y": 2},
        {"x": 4, "y": 2},
        {"x": 1, "y": 4},
        {"x": 1},
        {"y": 2},
        {"x": sympy.Integer(99), "y": 2},
    ],
)
def test_original_updates_match_write_order_errors_and_partial_state(
    outputs, contribution, phased
):
    compare_update(outputs, contribution, phased=phased)


@pytest.mark.parametrize("outcome", [0, 1, 2])
@pytest.mark.parametrize("current", [1.0, 1j, 2.0])
def test_scalar_measurement_phase_reads_new_output_without_revalidating_accumulator(
    outcome, current
):
    compare_update(
        {"x": outcome, "y": 0},
        MeasurementOutcomePhase("x"),
        phased=True,
        current=current,
    )


@pytest.mark.parametrize("index", [(0,), (1,)])
@pytest.mark.parametrize(
    "outputs",
    [
        {"x": np.array([0, 1]), "y": 2},
        {"x": np.array([1, 4]), "y": 2},
        {"x": np.array([1]), "y": 2},
        {"x": np.array([0, 1])},
    ],
)
def test_array_output_validation_precedes_indexed_measurement_phase(outputs, index):
    compare_update(
        outputs, MeasurementOutcomePhase("x", index), phased=True, shape=(2,)
    )


@pytest.mark.parametrize("contribution", [None, 1, 2, float("nan")])
@pytest.mark.parametrize("phased", [False, True])
def test_scalar_distributions_are_assigned_before_phase_policy_with_exact_rng_state(
    contribution, phased
):
    source_rng, native_rng = np.random.default_rng(149), np.random.default_rng(149)
    outputs = {"x": ClassicalDistribution(4), "y": ClassicalDistribution(4)}
    expected_error, expected, expected_phase = source_update(
        outputs, contribution, phased=phased, rng=source_rng
    )
    actual_error, actual, actual_phase = native_update(
        outputs, contribution, phased=phased, rng=native_rng
    )
    assert type(actual_error) is type(expected_error)
    assert tuple(actual) == tuple(expected)
    for key in expected:
        same_value(actual[key], expected[key])
    same_value(actual_phase, expected_phase)
    assert source_rng.bit_generator.state == native_rng.bit_generator.state


@pytest.mark.parametrize("fixed", [{0: 1}, {0: 4}, {}, {0: sympy.Integer(1)}])
@pytest.mark.parametrize("phased", [False, True])
def test_fixed_distribution_failures_precede_phase_rejection_or_measurement_feedback(
    fixed, phased
):
    compare_update(
        {"x": 1, "y": ClassicalDistribution(2)},
        MeasurementOutcomePhase("y"),
        phased=phased,
        fixed=fixed,
    )


def test_binding_failure_short_circuits_phase_helper_and_preserves_partial_writes(
    monkeypatch,
):
    import flagquantum.simulation.classical_updates as native

    assigned = {}
    marker = KeyError("second output")

    def assign(values):
        assigned[("x", ())] = values["x"]
        raise marker

    def forbidden(*args, **kwargs):
        pytest.fail("Phase application must not run after binding fails")

    monkeypatch.setattr(native, "apply_classical_phase", forbidden)
    with pytest.raises(KeyError) as actual:
        native.commit_phased_classical_outputs(
            {"x": 1}, 1j, assign, phase=1.0, outcomes=assigned
        )
    assert actual.value is marker and assigned == {("x", ()): 1}


def test_phase_helper_replacement_observes_fully_assigned_outputs(monkeypatch):
    import flagquantum.simulation.classical_updates as native

    outputs, assigned, seen = {"x": 1}, {}, []

    def assign(values):
        assert values is outputs
        assigned[("x", ())] = values["x"]
        seen.append("assigned")

    def apply(phase, contribution, outcomes):
        assert phase == 1j and contribution == -1
        assert outcomes is assigned and outcomes[("x", ())] == 1
        seen.append("phase")
        return -1j

    monkeypatch.setattr(native, "apply_classical_phase", apply)
    result = native.commit_phased_classical_outputs(
        outputs, -1, assign, phase=1j, outcomes=assigned
    )
    assert result == -1j and seen == ["assigned", "phase"]


@pytest.mark.parametrize("value", [0, 1])
def test_measurement_example_uses_ordered_commit_and_matches_complete_source_graph(
    value, monkeypatch
):
    from qualtran.bloqs.basic_gates import MeasureX
    from qualtran.simulation.classical_sim import do_phased_classical_simulation

    import examples.algorithms.classical_measurement_values as example

    original, visited = example.commit_phased_classical_outputs, []

    def commit(*args, **kwargs):
        visited.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(example, "commit_phased_classical_outputs", commit)
    source_rng, native_rng = np.random.default_rng(151), np.random.default_rng(151)
    for _ in range(16):
        expected, phase = do_phased_classical_simulation(
            MeasureX(), {"q": value}, rng=source_rng
        )
        actual, actual_phase = example.evaluate_measurement(value, native_rng)
        same_value(actual, expected["c"])
        same_value(actual_phase, phase)
    assert len(visited) == 16
    assert source_rng.bit_generator.state == native_rng.bit_generator.state
