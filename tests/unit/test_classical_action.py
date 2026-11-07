"""Bound callback dispatch follows the original graph step, not a rewritten oracle."""

import re
from dataclasses import asdict
from itertools import product

import numpy as np
import pytest

from flagquantum.simulation.classical_action import evaluate_classical_action
from tests.unit.test_classical_comparison import same_value

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def original_probe(output, phase, trace, *, value_error=None, phase_error=None):
    from qualtran import Bloq, Signature
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.simulation.classical_sim import PhasedClassicalSimState

    class Probe(Bloq):
        @property
        def signature(self):
            return Signature.build(q=1)

        def on_classical_vals(self, **values):
            trace.append(("values", values))
            if value_error is not None:
                raise value_error("value failure")
            return output

        def basis_state_phase(self, **values):
            trace.append(("phase", values))
            if phase_error is not None:
                raise phase_error("phase failure")
            return phase

        def build_composite_bloq(self, bb, q):
            trace.append(("decompose", {"q": 0}))
            return {"q": bb.add(XGate(), q=q)}

    class Capture(PhasedClassicalSimState):
        def _update(self, instance, outputs, contribution):
            # Observe the source dispatch boundary, before output binding/phase application.
            self.captured = (outputs, contribution)

    leaf = Probe()
    state = Capture.from_cbloq(leaf.as_composite_bloq(), {"q": 0})
    # Dangling nodes are processed by the real source step before the leaf.
    for _ in range(3):
        state.step()
        if getattr(state.last_binst, "bloq", None) is leaf:
            return state.captured
    pytest.fail("Source leaf was not visited")


def native_probe(output, phase, trace, *, value_error=None, phase_error=None):
    def values(**inputs):
        trace.append(("values", inputs))
        if value_error is not None:
            raise value_error("value failure")
        return output

    def basis_phase(**inputs):
        trace.append(("phase", inputs))
        if phase_error is not None:
            raise phase_error("phase failure")
        return phase

    def recurse(inputs):
        trace.append(("decompose", inputs))
        return {"q": 1}, 1.0

    return evaluate_classical_action(
        {"q": 0}, values, basis_phase, recurse, operation_name="Probe"
    )


class OutputDictionary(dict):
    pass


@pytest.mark.parametrize(
    "output", [NotImplemented, {"q": 0}, OutputDictionary(q=1), None, [], object()]
)
@pytest.mark.parametrize("phase", [None, 0, 1j, object()])
def test_value_phase_type_and_recursion_matrix_uses_unmodified_source_step(
    output, phase
):
    source_trace, native_trace = [], []
    try:
        expected = original_probe(output, phase, source_trace)
    except Exception as error:
        with pytest.raises(type(error), match=re.escape(str(error))):
            native_probe(output, phase, native_trace)
    else:
        actual = native_probe(output, phase, native_trace)
        assert type(actual) is tuple and len(actual) == 2
        if output is NotImplemented:
            assert actual == expected == ({"q": 1}, 1.0)
        else:
            assert actual[0] is expected[0] is output
            assert actual[1] is expected[1] is phase
    assert source_trace == native_trace


@pytest.mark.parametrize(
    "value_error,phase_error", [(RuntimeError, ValueError), (None, ValueError)]
)
@pytest.mark.parametrize("output", [None, NotImplemented, {"q": 1}])
def test_callback_error_order_precedes_output_type_and_recursion(
    output, value_error, phase_error
):
    source_trace, native_trace = [], []
    with pytest.raises(value_error or phase_error) as expected:
        original_probe(
            output, None, source_trace, value_error=value_error, phase_error=phase_error
        )
    with pytest.raises(type(expected.value), match=str(expected.value)):
        native_probe(
            output, None, native_trace, value_error=value_error, phase_error=phase_error
        )
    assert source_trace == native_trace
    assert [name for name, _ in source_trace] == (
        ["values"] if value_error else ["values", "phase"]
    )


def test_kwargs_copies_do_not_leak_top_level_mutation_but_shared_values_remain_shared():
    values = {"q": np.array([0])}
    seen = []

    def value_action(**inputs):
        seen.append(inputs)
        inputs["q"][0] = 1
        inputs["added"] = 5
        return {"q": inputs["q"]}

    def phase_action(**inputs):
        seen.append(inputs)
        assert "added" not in inputs
        assert inputs["q"] is values["q"]
        return

    output, phase = evaluate_classical_action(
        values,
        value_action,
        phase_action,
        lambda inputs: pytest.fail("no recursion"),
        operation_name="MutatingCallbacks",
    )
    assert seen[0] is not seen[1] and all(item is not values for item in seen)
    assert "added" not in values and values["q"][0] == 1
    assert output["q"] is values["q"] and phase is None


def test_recursion_receives_original_mapping_and_opaque_return_without_normalization():
    values, output, marker = {"q": 0}, OutputDictionary(q=1), object()

    def recurse(inputs):
        assert inputs is values
        inputs["retained"] = marker
        return output, marker

    actual = evaluate_classical_action(
        values,
        lambda **inputs: NotImplemented,
        lambda **inputs: None,
        recurse,
        operation_name="Composite",
    )
    assert actual[0] is output and actual[1] is marker
    assert values["retained"] is marker


@pytest.mark.parametrize("error", [NotImplementedError, KeyError, ValueError])
def test_decomposition_evaluator_exceptions_are_left_to_its_owner(error):
    marker = error("owned recursion failure")

    def recurse(inputs):
        raise marker

    with pytest.raises(error) as actual:
        evaluate_classical_action(
            {},
            lambda **values: NotImplemented,
            lambda **values: None,
            recurse,
            operation_name="Composite",
        )
    assert actual.value is marker


def original_word_recipe(inverse=False):
    from qualtran import Bloq, CtrlSpec, QBit, QUInt, Signature
    from qualtran.bloqs.basic_gates import XGate, ZGate

    class WordRecipe(Bloq):
        @property
        def signature(self):
            return Signature.build_from_dtypes(ctrl=QUInt(2), q=QBit())

        def build_composite_bloq(self, bb, ctrl, q):
            for child in ((ZGate(), XGate()) if inverse else (XGate(), ZGate())):
                controlled = child.controlled(CtrlSpec(qdtypes=QUInt(2), cvs=2))
                outputs = bb.add_d(
                    controlled, **{controlled.ctrl_reg_names[0]: ctrl, "q": q}
                )
                ctrl, q = outputs[controlled.ctrl_reg_names[0]], outputs["q"]
            return {"ctrl": ctrl, "q": q}

    return WordRecipe()


@pytest.mark.parametrize(
    "word,target,inverse", tuple(product(range(4), (0, 1), (False, True)))
)
def test_recursive_recipe_matches_original_complete_graph_values_and_phases(
    word, target, inverse
):
    from qualtran.simulation.classical_sim import do_phased_classical_simulation

    from examples.algorithms.classical_action_dispatch import evaluate_word_paulis

    expected, phase = do_phased_classical_simulation(
        original_word_recipe(inverse), {"ctrl": word, "q": target}
    )
    actual, actual_phase = evaluate_word_paulis(
        {"ctrl": word, "q": target}, inverse=inverse
    )
    for name in expected:
        same_value(actual[name], expected[name])
    same_value(actual_phase, phase)


def test_recursive_dispatch_source_stream_tensor_all_histories_clean_work_and_resources():
    import torch
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from examples.algorithms.classical_action_dispatch import evaluate_word_paulis
    from examples.algorithms.classical_controlled_values import WordControlledPaulis
    from flagquantum.compiler._component_trace.compound_control_resource import (
        compound_control_key,
    )
    from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
    from flagquantum.compiler._component_trace.resource_generalization import (
        CliffordCall,
    )
    from flagquantum.dynamic import DynamicCircuit
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    source = original_word_recipe()
    source_inverse = original_word_recipe(inverse=True)
    circuits = [DynamicCircuit(5), DynamicCircuit(5)]
    circuits[0].append(WordControlledPaulis())
    circuits[1].append(WordControlledPaulis(inverse=True))
    gates, inverse = (tuple(c.to_ir().instructions) for c in circuits)
    for sequence, program in ((gates, source), (inverse, source_inverse)):
        assert_source(
            sequence,
            reference_arithmetic_gates(
                program, {"ctrl": (0, 1), "q": (2,)}, (3,), {3: 0}
            ),
        )
    grid = torch.arange(1, 17, dtype=torch.float64)
    amplitudes = torch.complex(grid, 3 * grid - 5)
    amplitudes /= amplitudes.norm()
    initial, expected = (torch.zeros(32, dtype=torch.complex128) for _ in range(2))
    for word, target, spectator in product(range(4), (0, 1), (0, 1)):
        output, phase = evaluate_word_paulis({"ctrl": word, "q": target})
        before, after = (word << 3) | (target << 2) | spectator, (word << 3) | (
            output["q"] << 2
        ) | spectator
        initial[before] = amplitudes[(word << 2) | (target << 1) | spectator]
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
    for history in product((0, 1), repeat=2):
        branch = forced_branch(initial, gates, history)
        torch.testing.assert_close(branch, expected / 2, rtol=0, atol=2e-12)
        residual = branch.reshape(8, 2, 2)[:, 1, :]
        torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=0)
    for history in product((0, 1), repeat=4):
        torch.testing.assert_close(
            forced_branch(initial, gates + inverse, history),
            initial / 4,
            rtol=0,
            atol=2e-12,
        )
    for legacy in (False, True):
        native_categories = {}
        for kind in ("x", "z"):
            key = compound_control_key(CliffordCall(kind, 1), (1, 0))
            assert key is not None
            for name, count in asdict(
                query_qec_gate_cost(key, legacy_shims=legacy)
            ).items():
                native_categories[name] = native_categories.get(name, 0) + count
        model = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
        assert {
            name: value for name, value in native_categories.items() if value
        } == model.asdict()
        assert model.total_t_count() == sum(g.name in ("t", "tdg") for g in gates) == 8
        assert model.measurement == sum(g.name == "measure" for g in gates) == 2
        assert get_cost_value(source, QubitCount()) == 4


def test_example_reaches_shared_dispatcher_for_root_and_each_leaf(monkeypatch):
    import examples.algorithms.classical_action_dispatch as example

    original, visited = example.evaluate_classical_action, []

    def dispatch(*args, **kwargs):
        visited.append(kwargs["operation_name"])
        return original(*args, **kwargs)

    monkeypatch.setattr(example, "evaluate_classical_action", dispatch)
    output, phase = example.evaluate_word_paulis({"ctrl": 2, "q": 0})
    assert output == {"ctrl": 2, "q": 1} and phase == -1
    assert visited == ["WordControlledPaulis", "ControlledX", "ControlledZ"]
