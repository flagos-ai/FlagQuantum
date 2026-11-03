"""Review-only identity replacement; not a production checkpoint implementation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum.runtime.training_state import TrainingStateError, _workload_signature

pytestmark = pytest.mark.unit


def model(*, cut: bool = True, extra: bool = False, constant: float = 0.1) -> fq.Module:
    fixed = torch.tensor(constant, dtype=torch.float64)

    def build(p: Any, x: Any) -> Any:
        circuit = fq.Circuit(2, bsz=x.shape[0], dtype=torch.complex128)
        circuit.ry(0, p[0].square() + p[1] * x[:, 0].sin() + fixed)
        circuit.rx(1, (p[0].detach() if cut else p[0]) * x[:, 1])
        circuit.cx(0, 1)
        if extra:
            circuit.cx(1, 0)
        return circuit

    return fq.Module(
        build,
        2,
        dtype=torch.float64,
        init=torch.tensor([0.23, -0.31], dtype=torch.float64),
        policy=fq.RuntimePolicy(observable="z_sum", observable_qubits=(0, 1)),
    )


def features(index: int, grad: bool = False) -> torch.Tensor:
    return (
        torch.linspace(-0.3, 0.5, 16, dtype=torch.float64).reshape(8, 2) + index * 0.01
    ).requires_grad_(grad)


def digest(module: fq.Module) -> str:
    capture = module._captured_builder
    template = None if capture is None else (capture.template, capture.binding_names)
    assert template is not None and module._builder_program is not None
    # SAME-ENVIRONMENT FAKE ONLY. Production needs the contract's normalized
    # node encoding and version compatibility, not generated Python code.
    fixed = {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "value": value.tolist(),
        }
        for name, value in module._builder_program.state_dict().items()
    }
    description = {
        "ir": template[0].to_dict(),
        "binding": module._builder_program.code,
        "constants": fixed,
    }
    return hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()


def fake_save(module: fq.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return {
        "version": "flagquantum.training_state.v2",
        "workload_signature": digest(module),
        "module_state": copy.deepcopy(module.state_dict()),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
    }


def fake_restore(
    payload: dict[str, Any], module: fq.Module, optimizer: torch.optim.Optimizer
) -> None:
    if payload["version"] != "flagquantum.training_state.v2":
        raise ValueError("fake only admits its proposed v2 envelope")
    if digest(module) != payload["workload_signature"]:
        raise ValueError("different canonical program or binding")
    # This fake tests identity and deterministic state trajectories only. RNG,
    # precision, topology and optimizer-contract checks remain production work.
    module.load_state_dict(payload["module_state"])
    optimizer.load_state_dict(payload["optimizer_state"])


@pytest.mark.parametrize("kind", ("SGD", "Adam"))
@pytest.mark.parametrize("grad", (False, True))
def test_versioned_identity_fake_replays_six_updates_with_frozen_detached_bindings(
    kind, grad
):
    source = model()
    optimizer = getattr(torch.optim, kind)(source.parameters(), lr=0.03)
    saved = None

    def callback(step, loss, execution):
        nonlocal saved
        if step == 3:
            saved = fake_save(source, optimizer)

    complete = fq.train(
        source,
        optimizer=optimizer,
        objective=lambda value: value.square().mean(),
        inputs=lambda index: features(index, grad),
        steps=6,
        callback=callback,
    )
    target = model()
    target_optimizer = getattr(torch.optim, kind)(target.parameters(), lr=0.03)
    target.execute(features(2, grad))
    assert saved is not None
    fake_restore(saved, target, target_optimizer)
    replay = fq.train(
        target,
        optimizer=target_optimizer,
        objective=lambda value: value.square().mean(),
        inputs=lambda index: features(index + 3, grad),
        steps=3,
    )
    torch.testing.assert_close(
        target.parameters_tensor, source.parameters_tensor, atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        target_optimizer.state_dict(), optimizer.state_dict(), atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        replay.losses, complete.losses[3:], atol=1e-12, rtol=1e-12
    )
    assert (
        replay.last_execution.provenance["ir_hash"]
        == complete.last_execution.provenance["ir_hash"]
    )


@pytest.mark.parametrize("grad", (False, True))
def test_legacy_v1_keeps_its_existing_validation(grad, tmp_path: Path):
    source = model()
    optimizer = torch.optim.Adam(source.parameters(), lr=0.03)
    fq.train(
        source,
        optimizer=optimizer,
        objective=lambda value: value.square().mean(),
        inputs=lambda index: features(index, grad),
        steps=3,
    )
    path = source.save_checkpoint(
        tmp_path / "legacy.pt", optimizer=optimizer, seed=11, step=3
    )
    # Explicit legacy fixture: keep the v1 signature algorithm after the v2
    # writer lands, rather than reinterpreting a v2 digest as a v1 archive.
    payload = torch.load(path, weights_only=True)
    payload["version"] = "flagquantum.training_state.v1"
    payload["workload_signature"] = _workload_signature(source._last_ir)
    torch.save(payload, path)
    target = model()
    target_optimizer = torch.optim.Adam(target.parameters(), lr=0.03)
    target.execute(features(2, grad))
    if grad:
        target.load_checkpoint(path, optimizer=target_optimizer)
    else:
        before = target.parameters_tensor.detach().clone()
        with pytest.raises(TrainingStateError, match="workload"):
            target.load_checkpoint(path, optimizer=target_optimizer)
        torch.testing.assert_close(target.parameters_tensor, before, atol=0, rtol=0)


def test_fake_identity_ignores_dynamic_values_and_grad_masks():
    source, target = model(), model()
    source.execute(features(0))
    with torch.no_grad():
        target.parameters_tensor.add_(0.2)
    target.execute(features(4, True))
    assert digest(source) == digest(target)


@pytest.mark.parametrize("change", ({"cut": False}, {"extra": True}, {"constant": 0.2}))
def test_fake_still_refuses_changed_gradient_binding_topology_and_static_constants(
    change,
):
    source = model()
    source.execute(features(0))
    target = model(**change)
    target.execute(features(0))
    before = target.parameters_tensor.detach().clone()
    optimizer = torch.optim.SGD(source.parameters(), lr=0.03)
    with pytest.raises(ValueError, match="different canonical"):
        fake_restore(
            fake_save(source, optimizer),
            target,
            torch.optim.SGD(target.parameters(), lr=0.03),
        )
    torch.testing.assert_close(target.parameters_tensor, before, atol=0, rtol=0)
