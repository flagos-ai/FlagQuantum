"""Public keyword migration preserves numerical and serialized behavior."""

import dataclasses
import warnings

import pytest
import torch

import flagquantum as fq

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("factory", [fq.probabilities, fq.samples, fq.counts])
def test_selection_compatibility(factory):
    new = factory(qubits=(i for i in [0]))
    with pytest.warns(DeprecationWarning, match="0.4.0"):
        old = factory(wires=[0])
    assert new == old == factory([0])
    with pytest.raises(TypeError, match="not both"):
        factory(None, wires=None)
    q = fq.Circuit(2).h(0).cx(0, 1)
    a = fq.run(
        q,
        outputs=new,
        shots=None if factory is fq.probabilities else 64,
        options=fq.ExecutionOptions(seed=12),
    )
    b = fq.run(
        q,
        outputs=old,
        shots=None if factory is fq.probabilities else 64,
        options=fq.ExecutionOptions(seed=12),
    )
    if factory is fq.probabilities:
        assert torch.allclose(a.probabilities, b.probabilities)
    elif factory is fq.counts:
        assert a.counts == b.counts
    else:
        assert torch.equal(a.require_samples(), b.require_samples())


@pytest.mark.parametrize("factory", [fq.samples, fq.counts])
def test_observable_overload(factory):
    with pytest.warns(DeprecationWarning):
        old = factory(wires=fq.X(0))
    assert old == factory(qubits=fq.X(0))


def test_policy_roundtrip_and_gradients():
    new = fq.RuntimePolicy(observable="z_sum", observable_qubits=(0, 1))
    with pytest.warns(DeprecationWarning):
        old = fq.RuntimePolicy(observable="z_sum", observable_wires=(0, 1))
    assert old == new
    assert dataclasses.replace(new, observable="z").observable_qubits == (0, 1)
    payload = new.to_dict()
    assert payload["version"] == "2.0"
    assert fq.RuntimePolicy.from_dict(payload) == new
    historical = {
        k: v
        for k, v in payload.items()
        if k not in ("schema", "version", "observable_qubits")
    }
    historical["observable_wires"] = [0, 1]
    assert fq.RuntimePolicy.from_dict(historical) == new
    for bad in [dict(payload, version="9.0"), dict(payload, observable_wires=[0])]:
        with pytest.raises(ValueError):
            fq.RuntimePolicy.from_dict(bad)
    with pytest.raises(TypeError):
        fq.RuntimePolicy(observable_qubits=(0,), observable_wires=(0,))

    def circuit(p):
        return fq.Circuit(2).ry(0, p[0]).cx(0, 1)

    grads = []
    for policy in [old, new]:
        m = fq.Module(circuit, n_parameters=1, init=torch.tensor([0.25]), policy=policy)
        m().sum().backward()
        grads.append(next(m.parameters()).grad)
    assert torch.allclose(grads[0], grads[1])
    assert torch.allclose(grads[1], -2 * torch.sin(torch.tensor([0.25])), atol=1e-6)


def test_count_alias_warns():
    with pytest.warns(DeprecationWarning, match="n_qubits"):
        assert fq.Circuit(n_wires=2).n_qubits == 2


def test_historical_module_state_loads():
    def circuit(p):
        return fq.Circuit(2).ry(0, p[0]).cx(0, 1)

    model = fq.Module(
        circuit,
        n_parameters=1,
        init=torch.tensor([0.4]),
        policy=fq.RuntimePolicy(observable_qubits=(1,)),
    )
    state = model.state_dict()
    payload = state["_extra_state"]["policy"]
    payload.pop("schema")
    payload.pop("version")
    payload["observable_wires"] = payload.pop("observable_qubits")
    restored = fq.Module(circuit, n_parameters=1)
    restored.load_state_dict(state)
    assert restored.policy.observable_qubits == (1,)
    assert torch.allclose(model(), restored())
    assert "observable_qubits" in restored.state_dict()["_extra_state"]["policy"]


@pytest.mark.parametrize("factory", [fq.probabilities, fq.samples, fq.counts])
def test_invalid_qubit_indices(factory):
    with pytest.raises((ValueError, TypeError)):
        factory(qubits=[-1])


@pytest.mark.parametrize("factory", [fq.X, fq.Y, fq.Z])
def test_pauli_constructor_qubit_keyword(factory):
    new = factory(qubit=2)
    assert new == factory(2)
    with pytest.warns(DeprecationWarning, match="0.4.0"):
        old = factory(wire=2)
    assert old == new
    with pytest.raises(TypeError, match="not both"):
        factory(qubit=2, wire=2)
    with pytest.raises(TypeError, match="requires a qubit"):
        factory()


def test_identity_observable_qubit_keyword():
    assert fq.I() == fq.I(qubit=None)
    assert fq.I(qubit=0) == fq.I()
    with pytest.warns(DeprecationWarning, match="0.4.0"):
        assert fq.I(wire=0) == fq.I()
    with pytest.raises(TypeError, match="not both"):
        fq.I(qubit=0, wire=0)


def test_pauli_constructor_refusals_use_qubit_wording():
    """A refusal a user reads must not name the vocabulary the release is retiring."""

    with pytest.raises(TypeError, match="observable qubit must be an integer"):
        fq.X(0.5)
    with pytest.raises(ValueError, match="observable qubit must be a non-negative"):
        fq.Z(-1)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert fq.X(qubit=0) == fq.X(0)
        assert fq.I(qubit=0) == fq.I()

    with pytest.raises(ValueError, match="disjoint qubits"):
        _ = fq.Z(0) @ fq.X(0)


@pytest.mark.parametrize(
    ("model_name", "expected_qubits"),
    [
        ("HybridQuantumClassifier", (1,)),
        ("VariationalEnergyModel", (0, 1)),
    ],
)
def test_public_model_defaults_avoid_deprecated_keywords(model_name, expected_qubits):
    """A shipped default must not teach a keyword the release deprecates."""

    from flagquantum import models

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        model = getattr(models, model_name)()
    assert model.quantum.policy.observable_qubits == expected_qubits


def test_measurement_result_answers_the_wire_alias_with_a_warning():
    """`fq.MeasurementResult` is exported, so its old field name keeps working.

    The field is `qubits`; the old spelling is a forwarding property that warns,
    the same pattern `fq.Circuit.n_wires` and `fq.OutputRequest.wires` use.
    """

    result = fq.run(fq.Circuit(2).h(0), outputs=fq.probabilities())
    measurement = result.measurements[0]
    assert measurement.qubits == (0, 1)
    with pytest.warns(DeprecationWarning, match="qubits"):
        assert measurement.wires == measurement.qubits
    assert dataclasses.asdict(measurement)["qubits"] == (0, 1)
    assert "wires" not in dataclasses.asdict(measurement)


def test_a_qubit_named_measurement_signature_refuses_the_wire_spelling():
    """The runtime helpers are not exported, so they are renamed, not shimmed."""

    from flagquantum import observables
    from flagquantum.runtime import measurements

    requests = observables.lower_outputs(
        observables.probabilities(), n_qubits=2, shots=None
    )
    measurements.validate_measurements(requests, n_qubits=2)
    with pytest.raises(TypeError, match="n_wires"):
        measurements.validate_measurements(requests, n_wires=2)


def test_a_qubit_named_planner_estimate_refuses_the_wire_spelling():
    from flagquantum.runtime.planner import estimates

    assert estimates.estimate_state_bytes(n_qubits=3) == 64
    with pytest.raises(TypeError, match="n_wires"):
        estimates.estimate_state_bytes(n_wires=3)


def test_dynamic_circuit_uses_qubit_wording():
    """A dynamic program names its operands the way every other entry point does."""

    from flagquantum.runtime.dynamic.circuit import DynamicCircuit

    program = (
        DynamicCircuit(n_qubits=2)
        .h(0)
        .measure(qubit=0)
        .reset(qubit=1)
        .conditional("x", qubits=(1,), conditions={0: 1})
    )
    assert program._instructions[-1].wires == (1,)
    with pytest.raises(TypeError, match="wire"):
        DynamicCircuit(n_qubits=2).measure(wire=0)


def test_the_plan_analysis_view_exposes_a_qubit_named_count():
    """`LayerPlan.wires` and `CircuitAnalysis.n_wires` are frozen plan-JSON keys.

    Both stay, and both gain a qubit-named accessor, because the plan payload is
    the reason they are excluded; the slice renames what it can reach without
    re-keying that payload.
    """

    plan = fq.plan(fq.Circuit(3).h(0).cx(0, 1))
    assert plan.analysis.n_qubits == 3
    assert plan.layers[0].wires is not None
