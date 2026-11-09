"""Controlled uniform source costs, unchanged primitive path and exact branches."""

from dataclasses import asdict
from itertools import product

import pytest

from flagquantum.compiler.registers.preparation.uniform_preparation import (
    source_uniform_preparation_cost,
    uniform_preparation_requirements,
)
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.test_preparation_cost_profiles import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
PATTERNS = ((), (0,), (1,), (0, 0), (0, 1), (1, 1), (0, 1, 0), (1, 0, 1))


@pytest.mark.parametrize(
    "count,values,legacy",
    tuple(product((2, 3, 4, 5, 6, 7, 12, 16), PATTERNS, (False, True))),
)
def test_all_source_categories_and_lifetimes_in_both_directions(count, values, legacy):
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )

    source = PrepareUniformSuperposition(count, values)
    model = source_uniform_preparation_cost(
        count, control_values=values, legacy_shims=legacy
    )
    assert_source(model, source, legacy)
    assert_source(model, source.adjoint(), legacy)
    assert estimate_ftqc_resources(model).emitted is None


@pytest.mark.parametrize("values", PATTERNS[1:])
def test_each_initial_h_retains_a_separate_predicate_model(values):
    from flagquantum.compiler._component_trace.compound_control_resource import (
        CONTROLLED_HADAMARD_COST,
    )
    from flagquantum.compiler.registers.control.control_predicate import ControlRegister
    from flagquantum.compiler.registers.control.controlled_component_resource_model import (
        controlled_component_cost,
    )

    child = estimate_ftqc_resources(
        controlled_component_cost(
            CONTROLLED_HADAMARD_COST, (ControlRegister(1, values),), target_width=1
        )
    ).logical.gates
    one = estimate_ftqc_resources(
        source_uniform_preparation_cost(2, control_values=values)
    ).logical
    three = estimate_ftqc_resources(
        source_uniform_preparation_cost(8, control_values=values)
    ).logical
    assert asdict(one.gates) == asdict(child)
    assert asdict(three.gates) == {
        key: 3 * value for key, value in asdict(child).items()
    }
    width, work, _ = uniform_preparation_requirements(8, len(values))
    assert three.peak_logical_qubits == width + len(values) + work


@pytest.mark.parametrize("values", [(False,), (2,), (-1,), (0, True)])
def test_control_patterns_reject_nonbinary_or_boolean_values(values):
    with pytest.raises(ValueError):
        source_uniform_preparation_cost(3, control_values=values)


def test_source_controlled_singleton_cost_failure_is_not_claimed_as_valid():
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    for values in ((0,), (1,), (0, 1)):
        with pytest.raises(KeyError, match="target"):
            get_cost_value(PrepareUniformSuperposition(1, values), QECGatesCost())
        with pytest.raises(ValueError, match="singleton"):
            source_uniform_preparation_cost(1, control_values=values)
    # Preserve the existing uncontrolled zero-width metadata, without an
    # unsupported original decomposition/cost equivalence claim.
    assert source_uniform_preparation_cost(1).qubits == 0


def test_resource_only_control_query_never_emits_quantum_instructions(monkeypatch):
    import flagquantum.compiler.registers.preparation.uniform_preparation as module

    def forbidden(*args, **kwargs):
        raise AssertionError("Resource query invoked quantum emitter")

    monkeypatch.setattr(module, "uniform_preparation_ir", forbidden)
    monkeypatch.setattr(module, "_initial_hadamards", forbidden)
    report = estimate_ftqc_resources(
        module.source_uniform_preparation_cost(6, control_values=(1, 0))
    )
    assert report.emitted is None
    assert report.logical.gates.and_bloq == report.logical.gates.measurement == 10
    assert report.logical.gates.rotation == 8
    assert report.logical.peak_logical_qubits == 9


def test_runnable_controlled_uniform_model_and_lowered_ir_example(capsys):
    from examples.algorithms.register_uniform_preparation import main

    main()
    output = capsys.readouterr().out
    assert "Emitted T/Tdag 46" in output
    assert "Remaining unsynthesized Rz 2" in output
    assert "inverse verified" in output
