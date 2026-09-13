"""Public keyword migration preserves numerical and serialized behavior."""

import dataclasses

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
