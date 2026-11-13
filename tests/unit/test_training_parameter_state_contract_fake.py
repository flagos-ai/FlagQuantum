"""Review-only v3 parameter-state replacement for production conformance.

This fake checks the proposed parameter metadata, canonical identity and local
state transaction. It does not certify production codec admission, file atomicity,
RNG, precision, topology, or compatibility across devices and Torch versions.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypedDict

import pytest
import torch

import flagquantum as fq
from flagquantum.circuit import Circuit
from flagquantum.runtime.training_state import TrainingStateError

pytestmark = pytest.mark.unit

VERSION = "flagquantum.training_state.v3"
StateReader = Callable[[fq.Module, Mapping[str, Any]], None]


class ReviewCheckpoint(TypedDict):
    version: str
    workload_signature: str
    module_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None
    parameter_training: dict[str, object]


def _model(
    *, symbolic: bool = False, flat: bool = False, scale: float = 1.0
) -> fq.Module:
    if symbolic:
        circuit = (
            fq.Circuit(1, dtype=torch.complex128)
            .ry(0, scale * fq.Parameter("encoder"))
            .rx(0, fq.Parameter("readout"))
        )
        return fq.Module(
            circuit, init={"encoder": 0.4, "readout": 0.7}, dtype=torch.float64
        )
    if flat:
        return fq.Module(
            lambda weights: fq.Circuit(1).ry(0, scale * weights[0]).rx(0, weights[1]),
            2,
            init=torch.tensor([0.4, 0.7]),
            dtype=torch.float64,
        )

    def build(parameters: Mapping[str, torch.Tensor]) -> fq.Circuit:
        return (
            fq.Circuit(1)
            .ry(0, scale * parameters["encoder"])
            .rx(0, parameters["readout"])
        )

    return fq.Module(
        build,
        parameters={"encoder": (), "readout": ()},
        init={"encoder": 0.4, "readout": 0.7},
        dtype=torch.float64,
    )


def _identity(module: fq.Module) -> str:
    if type(module.circuit_builder) is Circuit:
        description = {
            "kind": "symbolic_circuit",
            "encoding": 1,
            "ir": module.circuit_builder.to_ir().to_dict(),
        }
    else:
        digest = module._checkpoint_signature()
        if digest is None:
            raise TrainingStateError("review requires a certified captured builder")
        description = {"kind": "captured_builder", "encoding": 2, "digest": digest}
    return hashlib.sha256(
        json.dumps(
            description, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _optimizer_names(
    module: fq.Module, optimizer: torch.optim.Optimizer | None
) -> list[list[str]] | None:
    names = {id(value): name for name, value in module.named_parameters()}
    if len(list(module.named_parameters(remove_duplicate=False))) != len(names):
        raise TrainingStateError("review refuses registered parameter aliases")
    if optimizer is None:
        return None
    groups = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        current = []
        for parameter in group["params"]:
            if id(parameter) not in names or id(parameter) in seen:
                raise TrainingStateError(
                    "review optimizer parameters must be unique and registered"
                )
            current.append(names[id(parameter)])
            seen.add(id(parameter))
        groups.append(current)
    return groups


def _save(
    module: fq.Module, optimizer: torch.optim.Optimizer | None
) -> ReviewCheckpoint:
    return {
        "version": VERSION,
        "workload_signature": _identity(module),
        "module_state": copy.deepcopy(module.state_dict()),
        "optimizer_state": (
            None if optimizer is None else copy.deepcopy(optimizer.state_dict())
        ),
        "parameter_training": {
            "requires_grad": {
                name: value.requires_grad for name, value in module.named_parameters()
            },
            "optimizer_parameters": _optimizer_names(module, optimizer),
        },
    }


def _state_dict_reader(module: fq.Module, state: Mapping[str, Any]) -> None:
    module.load_state_dict(state)


def _named_copy_reader(module: fq.Module, state: Mapping[str, Any]) -> None:
    # The replacement is deliberately limited to the fake's buffer-free models.
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            parameter.copy_(state[name])
    module.set_extra_state(state["_extra_state"])


def _restore(
    payload: ReviewCheckpoint,
    module: fq.Module,
    optimizer: torch.optim.Optimizer | None,
    reader: StateReader,
) -> None:
    if payload["version"] != VERSION:
        raise TrainingStateError("review refuses an unknown checkpoint version")
    if payload["workload_signature"] != _identity(module):
        raise TrainingStateError("review workload mismatch")
    training = payload["parameter_training"]
    if set(training) != {"requires_grad", "optimizer_parameters"}:
        raise TrainingStateError("review training metadata fields mismatch")
    parameters = dict(module.named_parameters())
    flags = training["requires_grad"]
    if (
        not isinstance(flags, dict)
        or set(flags) != set(parameters)
        or any(type(value) is not bool for value in flags.values())
    ):
        raise TrainingStateError(
            "review parameter flags must exactly cover registered names"
        )
    groups = training["optimizer_parameters"]
    if groups is not None and (
        not isinstance(groups, list)
        or any(
            not isinstance(group, list) or any(type(name) is not str for name in group)
            for group in groups
        )
    ):
        raise TrainingStateError("review optimizer bindings must be ordered name lists")
    if groups != _optimizer_names(module, optimizer):
        raise TrainingStateError("review optimizer binding names or order mismatch")
    if (payload["optimizer_state"] is None) != (optimizer is None):
        raise TrainingStateError("review optimizer presence mismatch")
    state = payload["module_state"]
    if set(state) != set(module.state_dict()):
        raise TrainingStateError("review module state names mismatch")
    for name, parameter in parameters.items():
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != parameter.shape
            or value.dtype != parameter.dtype
            or value.layout != torch.strided
        ):
            raise TrainingStateError("review state shape or dtype mismatch")
    before = copy.deepcopy(module.state_dict())
    previous_optimizer = (
        None if optimizer is None else copy.deepcopy(optimizer.state_dict())
    )
    previous_training = {
        name: (value.requires_grad, value.grad) for name, value in parameters.items()
    }
    try:
        reader(module, state)
        for name, parameter in parameters.items():
            parameter.requires_grad_(flags[name])
            parameter.grad = None
        if optimizer is not None:
            assert payload["optimizer_state"] is not None
            optimizer.load_state_dict(payload["optimizer_state"])
    except (RuntimeError, ValueError, TrainingStateError):
        # Transaction boundary: keep caller-owned tensors, gradients and moments.
        module.load_state_dict(before)
        for name, parameter in parameters.items():
            enabled, gradient = previous_training[name]
            parameter.requires_grad_(enabled)
            parameter.grad = gradient
        if optimizer is not None:
            assert previous_optimizer is not None
            optimizer.load_state_dict(previous_optimizer)
        raise


def _step(module: fq.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    torch.autograd.backward(module().square().mean())
    optimizer.step()


def _assert_module_state(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    assert set(actual) == set(expected)
    for name, value in actual.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
        else:
            assert value == expected[name]


@pytest.mark.parametrize("reader", (_state_dict_reader, _named_copy_reader))
@pytest.mark.parametrize("symbolic", (False, True))
@pytest.mark.parametrize("kind", ("AdamW", "SGD"))
def test_same_consumer_restores_frozen_flags_and_continues_exact_updates(
    reader: StateReader, symbolic: bool, kind: str
) -> None:
    source, target = _model(symbolic=symbolic), _model(symbolic=symbolic)
    source.parameter_groups["encoder"].requires_grad_(False)

    def optimizer(module: fq.Module) -> torch.optim.Optimizer:
        return (
            torch.optim.AdamW(module.parameters(), lr=0.03, weight_decay=0.1)
            if kind == "AdamW"
            else torch.optim.SGD(module.parameters(), lr=0.03, momentum=0.8)
        )

    left, right = optimizer(source), optimizer(target)
    for _ in range(3):
        _step(source, left)
    target()
    torch.autograd.backward(target().square().mean())
    identities = {name: id(value) for name, value in target.named_parameters()}
    payload = _save(source, left)
    _restore(payload, target, right, reader)
    assert not target.parameter_groups["encoder"].requires_grad
    assert all(value.grad is None for value in target.parameters())
    for step in range(4):
        if step == 2:
            source.parameter_groups["encoder"].requires_grad_(True)
            target.parameter_groups["encoder"].requires_grad_(True)
        _step(source, left)
        _step(target, right)
        _assert_module_state(target.state_dict(), source.state_dict())
        torch.testing.assert_close(
            right.state_dict(), left.state_dict(), atol=0, rtol=0
        )
    assert identities == {name: id(value) for name, value in target.named_parameters()}
    frozen = payload["parameter_training"]["requires_grad"]
    assert (
        isinstance(frozen, dict) and frozen["named_parameter_groups.encoder"] is False
    )


@pytest.mark.parametrize("reader", (_state_dict_reader, _named_copy_reader))
@pytest.mark.parametrize("flat", (False, True))
def test_all_frozen_inference_without_optimizer(
    reader: StateReader, flat: bool
) -> None:
    source, target = _model(flat=flat), _model(flat=flat)
    source.requires_grad_(False)
    source(), target()
    _restore(_save(source, None), target, None, reader)
    assert all(not value.requires_grad for value in target.parameters())
    assert not target().requires_grad
    torch.testing.assert_close(target(), source())


def test_omitted_trainable_parameter_is_distinct_from_freezing() -> None:
    source, target = _model(), _model()
    left = torch.optim.AdamW([source.parameter_groups["readout"]], lr=0.03)
    right = torch.optim.AdamW([target.parameter_groups["readout"]], lr=0.03)
    source(), target()
    _restore(_save(source, left), target, right, _state_dict_reader)
    assert target.parameter_groups["encoder"].requires_grad
    before = target.parameter_groups["encoder"].detach().clone()
    _step(target, right)
    torch.testing.assert_close(
        target.parameter_groups["encoder"], before, atol=0, rtol=0
    )
    assert target.parameter_groups["encoder"].grad is not None


@pytest.mark.parametrize(
    "change",
    (
        "missing",
        "extra",
        "integer_flag",
        "group_reorder",
        "duplicate",
        "bad_group",
        "state_shape",
        "state_dtype",
        "version",
        "workload",
    ),
)
def test_malformed_or_mismatched_payload_does_not_mutate_target(change: str) -> None:
    source, target = _model(), _model()
    left, right = torch.optim.AdamW(source.parameters()), torch.optim.AdamW(
        target.parameters()
    )
    source(), target()
    payload = _save(source, left)
    flags = payload["parameter_training"]["requires_grad"]
    assert isinstance(flags, dict)
    names = list(flags)
    if change == "missing":
        flags.pop(names[0])
    elif change == "extra":
        flags["unknown"] = True
    elif change == "integer_flag":
        flags[names[0]] = 1
    elif change == "group_reorder":
        payload["parameter_training"]["optimizer_parameters"] = [list(reversed(names))]
    elif change == "duplicate":
        payload["parameter_training"]["optimizer_parameters"] = [[names[0], names[0]]]
    elif change == "bad_group":
        payload["parameter_training"]["optimizer_parameters"] = ["invalid"]
    elif change == "state_shape":
        payload["module_state"][names[0]] = torch.zeros(2, dtype=torch.float64)
    elif change == "state_dtype":
        payload["module_state"][names[0]] = torch.zeros((), dtype=torch.float32)
    elif change == "version":
        payload["version"] = "unsupported"
    else:
        payload["workload_signature"] = "changed"
    torch.autograd.backward(target().square().mean())
    before = copy.deepcopy(target.state_dict())
    gradients = {name: value.grad for name, value in target.named_parameters()}
    with pytest.raises(TrainingStateError):
        _restore(payload, target, right, _state_dict_reader)
    _assert_module_state(target.state_dict(), before)
    assert all(
        value.grad is gradients[name] for name, value in target.named_parameters()
    )


def test_actual_optimizer_reorder_is_refused_even_with_identical_group_sizes() -> None:
    source, target = _model(), _model()
    source(), target()
    left = torch.optim.AdamW(source.parameters())
    right = torch.optim.AdamW(list(reversed(list(target.parameters()))))
    with pytest.raises(TrainingStateError, match="binding"):
        _restore(_save(source, left), target, right, _state_dict_reader)


def test_optimizer_failure_rolls_back_values_flags_gradients_and_moments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target = _model(), _model()
    left, right = torch.optim.AdamW(source.parameters()), torch.optim.AdamW(
        target.parameters()
    )
    _step(source, left)
    _step(target, right)
    source.parameter_groups["encoder"].requires_grad_(False)
    source()
    payload = _save(source, left)
    torch.autograd.backward(target().square().mean())
    before, optimizer_before = copy.deepcopy(target.state_dict()), copy.deepcopy(
        right.state_dict()
    )
    gradients = {name: value.grad for name, value in target.named_parameters()}
    original = right.load_state_dict
    calls = 0

    def fail_once(state: dict[str, Any]) -> None:
        nonlocal calls
        calls += 1
        original(state)
        if calls == 1:
            raise ValueError("injected optimizer load failure")

    monkeypatch.setattr(right, "load_state_dict", fail_once)
    with pytest.raises(ValueError, match="injected"):
        _restore(payload, target, right, _named_copy_reader)
    _assert_module_state(target.state_dict(), before)
    torch.testing.assert_close(right.state_dict(), optimizer_before, atol=0, rtol=0)
    assert all(value.requires_grad for value in target.parameters())
    assert all(
        value.grad is gradients[name] for name, value in target.named_parameters()
    )


@pytest.mark.parametrize("symbolic", (False, True))
def test_identity_ignores_training_flags_but_preserves_fixed_program_values(
    symbolic: bool,
) -> None:
    first, changed = _model(symbolic=symbolic), _model(symbolic=symbolic, scale=2.0)
    first(), changed()
    identity = _identity(first)
    first.requires_grad_(False)
    with torch.no_grad():
        for parameter in first.parameters():
            parameter.add_(0.1)
    first()
    assert _identity(first) == identity
    assert _identity(changed) != identity


def test_historical_v2_version_admission_refuses_v3(tmp_path: Path) -> None:
    # Freeze the historical reader's version admission, not the new reader's
    # behavior. Later checks are unreachable for v3 in that reader.
    import runpy

    reader = runpy.run_path(
        str(
            Path(__file__).parents[1]
            / "fixtures"
            / "checkpoint_v2_version_admission.py"
        )
    )["admit_version"]
    source = _model()
    source()
    path = source.save_checkpoint(tmp_path / "source.pt", optimizer=None, seed=1)
    payload = torch.load(path, weights_only=True)
    assert payload["version"] == VERSION
    with pytest.raises(TrainingStateError, match="unsupported training state"):
        reader(payload)


def test_historical_v2_keeps_caller_owned_training_flags(tmp_path: Path) -> None:
    source, target = _model(), _model()
    source.parameter_groups["encoder"].requires_grad_(False)
    source(), target()
    path = source.save_checkpoint(tmp_path / "v2.pt", optimizer=None, seed=1)
    payload = torch.load(path, weights_only=True)
    payload["version"] = "flagquantum.training_state.v2"
    payload["workload_signature"] = source._checkpoint_signature()
    payload.pop("parameter_training", None)
    torch.save(payload, path)
    target.load_checkpoint(path, optimizer=None)
    assert target.parameter_groups["encoder"].requires_grad
