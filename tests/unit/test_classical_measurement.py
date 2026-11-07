"""Measured numerical phases follow the unchanged original execution path."""

from dataclasses import asdict

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_distributions import draw_classical_value
from flagquantum.simulation.classical_measurement import (
    MeasurementOutcomePhase,
    apply_classical_phase,
    assert_unit_classical_phase,
    measure_x_basis_phase,
    measure_x_classical_values,
)
from tests.unit.test_classical_comparison import compare_calls, same_value

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "value",
    [
        0,
        1,
        -1,
        2,
        False,
        True,
        0.0,
        1.0,
        1j,
        np.int8(1),
        np.array(0),
        np.array([1]),
        np.array([0, 1]),
        sympy.Integer(1),
        sympy.Symbol("q"),
        None,
        "1",
    ],
)
def test_measure_x_raw_callbacks_retain_source_guards_and_phase_descriptors(value):
    from qualtran.bloqs.basic_gates import MeasureX
    from qualtran.simulation.classical_sim import MeasurementPhase

    source = MeasureX()
    try:
        original = source.on_classical_vals(q=value)
    except Exception as error:
        with pytest.raises(type(error)):
            measure_x_classical_values(value)
    else:
        native = measure_x_classical_values(value)
        assert list(native) == list(original) == ["c"]
        assert native["c"].choices == original["c"].a == 2
        assert native["c"].probabilities is original["c"].p is None
    try:
        original_phase = source.basis_state_phase(q=value)
    except Exception as error:
        with pytest.raises(type(error)):
            measure_x_basis_phase(value)
    else:
        native_phase = measure_x_basis_phase(value)
        if isinstance(original_phase, MeasurementPhase):
            assert isinstance(native_phase, MeasurementOutcomePhase)
            assert native_phase.register == original_phase.reg_name
            assert native_phase.index == original_phase.idx
        else:
            same_value(native_phase, original_phase)


@pytest.mark.parametrize(
    "phase",
    [
        1,
        -1,
        1j,
        np.exp(0.2j),
        0,
        2,
        1 + 1e-9,
        1 + 1e-7,
        float("nan"),
        float("inf"),
        np.array(1),
        np.array([1, 1]),
        None,
        "1",
    ],
)
@pytest.mark.parametrize("atol", [0, 1e-8])
def test_unit_phase_validation_matches_original_including_nan_and_array_errors(
    phase, atol
):
    from qualtran.simulation.classical_sim import _assert_valid_phase

    compare_calls(
        lambda p: (_assert_valid_phase(p, atol=atol),),
        lambda p: (assert_unit_classical_phase(p, atol=atol),),
        (phase,),
    )


@pytest.mark.parametrize("current", [1.0, 1j, 2.0])
@pytest.mark.parametrize(
    "contribution", [None, 1, -1, 1j, np.exp(0.3j), 0, 2, "measured"]
)
@pytest.mark.parametrize("outcome", [0, 1])
def test_bound_phase_update_matches_original_state_without_revalidating_accumulator(
    current, contribution, outcome
):
    from qualtran.bloqs.basic_gates import MeasureX
    from qualtran.simulation.classical_sim import (
        MeasurementPhase,
        PhasedClassicalSimState,
    )

    composite = MeasureX().as_composite_bloq()
    source = PhasedClassicalSimState.from_cbloq(
        composite, {"q": 0}, fixed_random_vals={}
    )
    instance = next(iter(composite.bloq_instances))
    source.phase = current
    original_phase = (
        MeasurementPhase("c") if contribution == "measured" else contribution
    )
    native_phase = (
        MeasurementOutcomePhase("c") if contribution == "measured" else contribution
    )
    try:
        source._update(instance, {"c": outcome}, original_phase)
    except Exception as error:
        with pytest.raises(type(error)):
            apply_classical_phase(current, native_phase, {("c", ()): outcome})
    else:
        actual = apply_classical_phase(current, native_phase, {("c", ()): outcome})
        same_value(actual, source.phase)


@pytest.mark.parametrize("value", [0, 1])
@pytest.mark.parametrize("seed", [0, 91])
def test_rng_and_fixed_measurement_sequences_match_original_complete_leaf_graph(
    value, seed
):
    from qualtran.bloqs.basic_gates import MeasureX
    from qualtran.simulation.classical_sim import do_phased_classical_simulation

    original_rng, native_rng = np.random.default_rng(seed), np.random.default_rng(seed)
    for _ in range(32):
        expected, phase = do_phased_classical_simulation(
            MeasureX(), {"q": value}, rng=original_rng
        )
        distribution = measure_x_classical_values(value)["c"]
        outcome = draw_classical_value(distribution, instance_id=0, rng=native_rng)
        actual_phase = apply_classical_phase(
            1.0, measure_x_basis_phase(value), {("c", ()): outcome}
        )
        same_value(outcome, expected["c"])
        same_value(actual_phase, phase)
    for fixed in (0, 1):
        expected, phase = do_phased_classical_simulation(
            MeasureX(), {"q": value}, fixed_random_vals={0: fixed}
        )
        outcome = draw_classical_value(
            measure_x_classical_values(value)["c"],
            instance_id=0,
            fixed_values={0: fixed},
        )
        assert outcome == expected["c"] == fixed
        same_value(
            apply_classical_phase(
                1.0, measure_x_basis_phase(value), {("c", ()): outcome}
            ),
            phase,
        )


def test_phase_resolution_uses_exact_index_and_preserves_short_circuit(monkeypatch):
    import flagquantum.simulation.classical_measurement as native

    index = (1, 2)
    descriptor = MeasurementOutcomePhase("out", index)
    assert descriptor.index is index
    assert apply_classical_phase(1j, descriptor, {("out", index): 1}) == -1j
    with pytest.raises(KeyError):
        apply_classical_phase(1j, descriptor, {})

    class Forbidden(dict):
        def __getitem__(self, key):
            pytest.fail("Numeric phases must not read measurement results")

    assert apply_classical_phase(1j, None, Forbidden()) == 1j
    assert apply_classical_phase(1j, -1, Forbidden()) == -1j
    called = []

    def validate(value, *, atol):
        called.append((value, atol))
        raise ValueError("replacement validator")

    monkeypatch.setattr(native, "assert_unit_classical_phase", validate)
    with pytest.raises(ValueError, match="replacement validator"):
        native.apply_classical_phase(1j, 1, {}, atol=1e-5)
    assert called == [(1, 1e-5)]
    assert native.apply_classical_phase(1j, descriptor, {("out", index): 1}) == -1j


@pytest.mark.parametrize("control,value", [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("include_output", [False, True])
def test_controlled_measurement_retains_original_right_port_callback_failures(
    control, value, include_output
):
    from qualtran import CtrlSpec
    from qualtran._infra.controlled import Controlled
    from qualtran.bloqs.basic_gates import MeasureX

    from flagquantum.simulation.classical_control_patterns import (
        normalize_control_patterns,
    )
    from flagquantum.simulation.classical_controlled_values import (
        controlled_basis_phase,
    )

    source = Controlled(MeasureX(), CtrlSpec())
    named = {"ctrl": control, "q": value}
    if include_output:
        named["c"] = 0

    def child(q):
        return measure_x_basis_phase(q)

    def native():
        return controlled_basis_phase(
            named, ("ctrl",), ("q", "c"), normalize_control_patterns(1), child
        )

    try:
        expected = source.basis_state_phase(**named)
    except Exception as error:
        with pytest.raises(type(error)):
            native()
    else:
        same_value(native(), expected)


def test_complete_complex_measurement_channels_and_copy_erasure_all_branches():
    import torch
    from qualtran.bloqs.basic_gates import MeasureX
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from examples.algorithms.classical_measurement_values import ClearCopiedWork
    from flagquantum.compiler._component_trace.gate_budget import t_and_ccz_budget
    from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
    from flagquantum.compiler._component_trace.resource_generalization import (
        MeasurementCall,
    )
    from flagquantum.compiler.registers.measurement import measure_x_clear_ir
    from flagquantum.dynamic import DynamicCircuit
    from tests.unit.measurement_checks import forced_branch

    initial = torch.tensor([1 + 2j, 3 - 4j, -2 + 1j, 4 + 3j], dtype=torch.complex128)
    initial /= initial.norm()
    for measured in (0, 1):
        expected = torch.zeros_like(initial)
        for value in (0, 1):
            phase = apply_classical_phase(
                1.0, measure_x_basis_phase(value), {("c", ()): measured}
            )
            expected[:2] += initial[2 * value : 2 * value + 2] * phase / 2**0.5
        actual = forced_branch(initial, measure_x_clear_ir(0, 0), (measured,))
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-12)
        torch.testing.assert_close(
            actual[2:], torch.zeros_like(actual[2:]), rtol=0, atol=0
        )
    circuit = DynamicCircuit(3)
    circuit.append(ClearCopiedWork())
    before = torch.zeros(8, dtype=torch.complex128)
    before[[0, 1, 4, 5]] = initial
    for measured in (0, 1):
        branch = forced_branch(before, tuple(circuit.to_ir().instructions), (measured,))
        torch.testing.assert_close(branch, before / 2**0.5, rtol=0, atol=2e-12)
        torch.testing.assert_close(
            branch.reshape(2, 2, 2)[:, 1, :],
            torch.zeros((2, 2), dtype=torch.complex128),
            rtol=0,
            atol=0,
        )
    for legacy in (False, True):
        model = query_qec_gate_cost(MeasurementCall("x"), legacy_shims=legacy)
        source = get_cost_value(MeasureX(), QECGatesCost(legacy_shims=legacy))
        assert {
            name: value for name, value in asdict(model).items() if value
        } == source.asdict()
        assert t_and_ccz_budget(model) == source.total_t_and_ccz_count()
        assert model.measurement == 1 and source.total_t_count() == 0
