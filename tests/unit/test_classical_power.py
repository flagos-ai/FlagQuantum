"""Numerical repetitions retain original constructor and literal-build dispatch."""

from dataclasses import asdict
from itertools import product

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_power import power_classical_values
from tests.unit.test_classical_comparison import same_value

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


class SymbolicInteger(int):
    def is_symbolic(self):
        return True


@pytest.mark.parametrize(
    "count",
    [
        1,
        2,
        3,
        True,
        False,
        0,
        -1,
        1.5,
        np.int64(2),
        np.bool_(True),
        sympy.Integer(2),
        sympy.Symbol("n"),
        SymbolicInteger(2),
        SymbolicInteger(0),
        SymbolicInteger(-1),
        None,
    ],
)
@pytest.mark.parametrize("value", [0, 1])
def test_constructor_and_full_graph_numeric_count_dispatch_match_original(count, value):
    from qualtran.bloqs.basic_gates import Power, XGate
    from qualtran.simulation.classical_sim import do_phased_classical_simulation

    from flagquantum.simulation.classical_pauli_values import x_classical_value

    seen = []

    def child(inputs):
        seen.append(inputs["q"])
        return {"q": x_classical_value(inputs["q"])}, 1.0

    try:
        source = Power(XGate(), count)
        expected, phase = do_phased_classical_simulation(source, {"q": value})
    except Exception as error:
        with pytest.raises(type(error)):
            power_classical_values({"q": value}, count, child, lambda outputs: None)
        assert seen == []
    else:
        actual, actual_phase = power_classical_values(
            {"q": value}, count, child, lambda outputs: None
        )
        same_value(actual["q"], expected["q"])
        same_value(actual_phase, phase)
        assert len(seen) == max(0, int(count))


def test_non_thru_guard_precedes_symbolic_protocol_and_any_child_call():
    from qualtran.bloqs.basic_gates import Power, ZeroState

    class ExplodingCount:
        def is_symbolic(self):
            pytest.fail("Non-THRU guard must run before the count classifier")

    with pytest.raises(ValueError, match="THRU"):
        Power(ZeroState(), ExplodingCount())
    with pytest.raises(ValueError, match="THRU"):
        power_classical_values(
            {},
            ExplodingCount(),
            lambda values: pytest.fail("No child call"),
            lambda outputs: pytest.fail("No assignment"),
            thru_only=False,
        )


def test_ordered_child_assign_phase_and_top_level_keyword_copy(monkeypatch):
    import flagquantum.simulation.classical_power as native

    values, events, identities = {"q": 0}, [], []

    def child(inputs):
        assert inputs is not values
        identities.append(inputs)
        events.append(("child", inputs["q"]))
        inputs["q"] += 1
        return inputs, 1j

    def assign(outputs):
        events.append(("assign", outputs["q"]))

    original = native.commit_phased_classical_outputs

    def commit(outputs, contribution, assign_outputs, **kwargs):
        result = original(outputs, contribution, assign_outputs, **kwargs)
        events.append(("phase", result))
        return result

    monkeypatch.setattr(native, "commit_phased_classical_outputs", commit)
    output, phase = native.power_classical_values(values, 2, child, assign)
    assert values == {"q": 0} and output == {"q": 2} and phase == -1
    assert identities[0] is not identities[1] and output is identities[1]
    assert events == [
        ("child", 0),
        ("assign", 1),
        ("phase", 1j),
        ("child", 1),
        ("assign", 2),
        ("phase", -1),
    ]


@pytest.mark.parametrize("failure", ["child", "assign"])
def test_failure_stops_remaining_iterations_and_preserves_exception(failure):
    seen, marker = [], RuntimeError("owned child failure")

    def child(values):
        seen.append("child")
        if failure == "child":
            raise marker
        return values, 1.0

    def assign(outputs):
        seen.append("assign")
        raise marker

    with pytest.raises(RuntimeError) as error:
        power_classical_values({"q": 0}, 3, child, assign)
    assert error.value is marker
    assert seen == (["child"] if failure == "child" else ["child", "assign"])


@pytest.mark.parametrize(
    "count,word,target,inverse",
    tuple(product((1, 2, 3), range(4), (0, 1), (False, True))),
)
def test_complete_controlled_recipe_repetition_values_and_phase_match_source(
    count, word, target, inverse
):
    from qualtran.bloqs.basic_gates import Power
    from qualtran.simulation.classical_sim import do_phased_classical_simulation

    from examples.algorithms.classical_power_values import evaluate_powered_word
    from tests.unit.test_classical_action import original_word_recipe

    source = Power(original_word_recipe(inverse=inverse), count)
    expected, phase = do_phased_classical_simulation(
        source, {"ctrl": word, "q": target}
    )
    actual, actual_phase = evaluate_powered_word(
        {"ctrl": word, "q": target}, count, inverse=inverse
    )
    for name in expected:
        same_value(actual[name], expected[name])
    same_value(actual_phase, phase)


@pytest.mark.parametrize("count", [1, 2])
def test_power_original_stream_tensor_all_measured_histories_work_and_models(count):
    import torch
    from qualtran.bloqs.basic_gates import Power
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from examples.algorithms.classical_power_values import (
        PoweredWordPaulis,
        evaluate_powered_word,
    )
    from flagquantum.compiler._component_trace.compound_control_resource import (
        compound_control_key,
    )
    from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
    from flagquantum.compiler._component_trace.resource_generalization import (
        CliffordCall,
        PowerCall,
    )
    from flagquantum.dynamic import DynamicCircuit
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_classical_action import original_word_recipe
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    source = Power(original_word_recipe(), count)
    circuits = [DynamicCircuit(5), DynamicCircuit(5)]
    circuits[0].append(PoweredWordPaulis(count))
    circuits[1].append(PoweredWordPaulis(count, inverse=True))
    gates, inverse = (tuple(c.to_ir().instructions) for c in circuits)
    for stream, program in ((gates, source), (inverse, source.adjoint())):
        assert_source(
            stream,
            reference_arithmetic_gates(
                program, {"ctrl": (0, 1), "q": (2,)}, (3,), {3: 0}
            ),
        )
    grid = torch.arange(1, 17, dtype=torch.float64)
    weights = torch.complex(grid, 3 * grid - 5)
    weights /= weights.norm()
    initial, expected = (torch.zeros(32, dtype=torch.complex128) for _ in range(2))
    for word, target, spectator in product(range(4), (0, 1), (0, 1)):
        output, phase = evaluate_powered_word({"ctrl": word, "q": target}, count)
        before = (word << 3) | (target << 2) | spectator
        after = (word << 3) | (output["q"] << 2) | spectator
        initial[before] = weights[(word << 2) | (target << 1) | spectator]
        expected[after] = phase * initial[before]
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        8, 8
    )
    torch.testing.assert_close(
        matrix @ initial.reshape(8, 2, 2)[:, 0, :],
        expected.reshape(8, 2, 2)[:, 0, :],
        rtol=0,
        atol=2e-12,
    )
    for history in product((0, 1), repeat=2 * count):
        actual = forced_branch(initial, gates, history)
        torch.testing.assert_close(actual, expected / 2**count, rtol=0, atol=2e-12)
        residual = actual.reshape(8, 2, 2)[:, 1, :]
        torch.testing.assert_close(
            residual, torch.zeros_like(residual), rtol=0, atol=2e-12
        )
    for history in product((0, 1), repeat=4 * count):
        torch.testing.assert_close(
            forced_branch(initial, gates + inverse, history),
            initial / 2 ** (2 * count),
            rtol=0,
            atol=2e-12,
        )
    for legacy in (False, True):
        categories = {}
        for kind in ("x", "z"):
            child = compound_control_key(CliffordCall(kind, 1), (1, 0))
            assert child is not None
            for name, value in asdict(
                query_qec_gate_cost(PowerCall(child, count), legacy_shims=legacy)
            ).items():
                categories[name] = categories.get(name, 0) + value
        model = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
        assert {
            name: value for name, value in categories.items() if value
        } == model.asdict()
        assert (
            model.total_t_count()
            == sum(g.name in ("t", "tdg") for g in gates)
            == 8 * count
        )
        assert model.measurement == sum(g.name == "measure" for g in gates) == 2 * count
        assert get_cost_value(source, QubitCount()) == 4
