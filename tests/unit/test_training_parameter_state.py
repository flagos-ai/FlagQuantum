"""Production v3 conformance and transactional checkpoint workflows."""

from __future__ import annotations

import copy
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum.circuit import Circuit
from flagquantum.runtime import checkpointing as training_state
from flagquantum.runtime.checkpointing import TrainingStateError
from tests.unit.test_training_parameter_state_contract_fake import (
    _assert_module_state,
    _identity,
    _model,
    _restore,
    _save,
    _state_dict_reader,
    _step,
)

pytestmark = pytest.mark.unit


def _checkpoint(
    source: torch.nn.Module, path: Path, optimizer: torch.optim.Optimizer | None = None
) -> dict[str, Any]:
    owner = training_state._quantum_owner(source)
    training_state.save_training_checkpoint(
        path,
        module=source,
        optimizer=optimizer,
        seed=training_state.SeedContract(11, 12, 13, (0, 11)),
        precision=getattr(owner, "precision"),
        step=3,
    )
    return dict(torch.load(path, weights_only=True))


@pytest.mark.parametrize("codec", ("review", "production"))
@pytest.mark.parametrize(
    "symbolic,flat", ((True, False), (False, False), (False, True))
)
def test_same_training_consumer_resumes_through_both_codecs(
    tmp_path: Path, codec: str, symbolic: bool, flat: bool
) -> None:
    source, target = _model(symbolic=symbolic, flat=flat), _model(
        symbolic=symbolic, flat=flat
    )
    left, right = torch.optim.AdamW(source.parameters(), lr=0.03), torch.optim.AdamW(
        target.parameters(), lr=0.1
    )
    for _ in range(2):
        _step(source, left)
    frozen = next(source.parameters())
    frozen.requires_grad_(False)
    frozen.grad = None
    if not flat:
        _step(source, left)
    # A differently initialized, initially trainable target has stale gradients.
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.add_(0.8)
    target().sum().backward()
    identities = [id(parameter) for parameter in target.parameters()]
    source()
    path = tmp_path / "checkpoint.pt"
    if codec == "review":
        _restore(_save(source, left), target, right, _state_dict_reader)
    else:
        payload = _checkpoint(source, path, left)
        assert payload["workload_signature"] == _identity(source)
        assert (
            payload["parameter_training"] == _save(source, left)["parameter_training"]
        )
        if not symbolic:
            assert target._captured_builder is not None
        target.load_checkpoint(path, optimizer=right)
        assert target._captured_builder is None
    assert not next(target.parameters()).requires_grad
    assert all(parameter.grad is None for parameter in target.parameters())
    # Native unfreezing retains the restored optimizer moments.
    for model in (source, target):
        model.requires_grad_(True)
    for _ in range(3):
        _step(source, left)
        _step(target, right)
        _assert_module_state(target.state_dict(), source.state_dict())
        torch.testing.assert_close(
            right.state_dict(), left.state_dict(), atol=0, rtol=0
        )
    assert identities == [id(parameter) for parameter in target.parameters()]


@pytest.mark.parametrize(
    "symbolic,flat", ((True, False), (False, False), (False, True))
)
def test_all_frozen_checkpoint_restores_inference(
    tmp_path: Path, symbolic: bool, flat: bool
) -> None:
    source, target = _model(symbolic=symbolic, flat=flat), _model(
        symbolic=symbolic, flat=flat
    )
    source.requires_grad_(False)
    expected = source()
    path = tmp_path / "inference.pt"
    _checkpoint(source, path)
    target.load_checkpoint(path, optimizer=None)
    assert not target().requires_grad
    torch.testing.assert_close(target(), expected, atol=0, rtol=0)


def test_omitted_optimizer_membership_and_multiple_groups_resume(
    tmp_path: Path,
) -> None:
    source, target = _model(), _model()
    source(), target()
    left = torch.optim.AdamW([source.parameter_groups["readout"]], lr=0.03)
    right = torch.optim.AdamW([target.parameter_groups["readout"]], lr=0.09)
    _step(source, left)
    path = tmp_path / "partial.pt"
    _checkpoint(source, path, left)
    target.load_checkpoint(path, optimizer=right)
    assert target.parameter_groups["encoder"].requires_grad
    before = target.parameter_groups["encoder"].detach().clone()
    _step(source, left)
    _step(target, right)
    torch.testing.assert_close(
        target.parameter_groups["encoder"], before, atol=0, rtol=0
    )
    for model, optimizer in ((source, left), (target, right)):
        optimizer.add_param_group(
            {"params": [model.parameter_groups["encoder"]], "lr": 0.02}
        )
    _step(source, left)
    _checkpoint(source, path, left)
    target.load_checkpoint(path, optimizer=right)
    _step(source, left)
    _step(target, right)
    _assert_module_state(target.state_dict(), source.state_dict())
    torch.testing.assert_close(right.state_dict(), left.state_dict(), atol=0, rtol=0)


@pytest.mark.parametrize(
    "change",
    (
        "missing_training",
        "missing_policy",
        "extra_training",
        "missing_flag",
        "extra_flag",
        "integer_flag",
        "bad_group",
        "duplicate",
        "group_reorder",
        "state_shape",
        "state_dtype",
        "state_sparse",
        "missing_state",
        "extra_state",
        "optimizer_ids",
        "optimizer_boolean_ids",
        "optimizer_state_id",
        "negative_step",
        "bad_precision",
        "bad_topology",
        "missing_seed",
        "bad_seed",
        "unknown_version",
        "changed_workload",
    ),
)
def test_invalid_checkpoint_preserves_target_and_rng(
    tmp_path: Path, change: str
) -> None:
    source, target = _model(), _model()
    left, right = torch.optim.AdamW(source.parameters()), torch.optim.AdamW(
        target.parameters()
    )
    _step(source, left)
    _step(target, right)
    target().sum().backward()
    path = tmp_path / "invalid.pt"
    payload = _checkpoint(source, path, left)
    flags = payload["parameter_training"]["requires_grad"]
    names = list(flags)
    if change == "missing_training":
        payload.pop("parameter_training")
    elif change == "missing_policy":
        payload.pop("runtime_policy")
    elif change == "extra_training":
        payload["parameter_training"]["unknown"] = None
    elif change == "missing_flag":
        flags.pop(names[0])
    elif change == "extra_flag":
        flags["unknown"] = True
    elif change == "integer_flag":
        flags[names[0]] = 1
    elif change == "bad_group":
        payload["parameter_training"]["optimizer_parameters"] = ["bad"]
    elif change == "duplicate":
        payload["parameter_training"]["optimizer_parameters"] = [[names[0], names[0]]]
    elif change == "group_reorder":
        payload["parameter_training"]["optimizer_parameters"] = [names[::-1]]
    elif change == "state_shape":
        payload["module_state"][names[0]] = torch.zeros(2, dtype=torch.float64)
    elif change == "state_dtype":
        payload["module_state"][names[0]] = torch.zeros((), dtype=torch.float32)
    elif change == "state_sparse":
        payload["module_state"][names[0]] = torch.zeros(
            (), dtype=torch.float64
        ).to_sparse()
    elif change == "missing_state":
        payload["module_state"].pop(names[0])
    elif change == "extra_state":
        payload["module_state"]["unknown"] = torch.tensor(0.0)
    elif change == "optimizer_ids":
        payload["optimizer_state"]["param_groups"][0]["params"] = [1, 0]
    elif change == "optimizer_boolean_ids":
        payload["optimizer_state"]["param_groups"][0]["params"] = [False, True]
    elif change == "optimizer_state_id":
        payload["optimizer_state"]["state"][99] = {}
    elif change == "negative_step":
        payload["step"] = -1
    elif change == "bad_precision":
        payload["precision"] = None
    elif change == "bad_topology":
        payload["topology"] = None
    elif change == "missing_seed":
        payload.pop("seed_state")
    elif change == "bad_seed":
        payload["seed_state"]["torch_cpu"] = torch.tensor([0.5])
    elif change == "unknown_version":
        payload["version"] = "flagquantum.training_state.v99"
    else:
        payload["workload_signature"] = "changed"
    torch.save(payload, path)
    versions = [parameter._version for parameter in target.parameters()]
    before = copy.deepcopy(target.state_dict())
    optimizer_before = copy.deepcopy(right.state_dict())
    gradients = [parameter.grad for parameter in target.parameters()]
    rng, python_rng = torch.get_rng_state(), random.getstate()
    with pytest.raises(TrainingStateError):
        target.load_checkpoint(path, optimizer=right)
    assert versions == [parameter._version for parameter in target.parameters()]
    _assert_module_state(target.state_dict(), before)
    torch.testing.assert_close(right.state_dict(), optimizer_before, atol=0, rtol=0)
    assert all(
        parameter.grad is gradient
        for parameter, gradient in zip(target.parameters(), gradients, strict=True)
    )
    assert all(parameter.requires_grad for parameter in target.parameters())
    assert torch.equal(torch.get_rng_state(), rng) and random.getstate() == python_rng


@pytest.mark.parametrize("stage", ("optimizer", "rng"))
def test_failure_after_mutation_restores_flags_gradients_buffers_and_moments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    source, target = _model(), _model()
    source.register_buffer("statistics", torch.tensor([4.0]))
    target.register_buffer("statistics", torch.tensor([9.0]))
    left, right = torch.optim.AdamW(source.parameters()), torch.optim.AdamW(
        target.parameters()
    )
    _step(source, left)
    _step(target, right)
    source.requires_grad_(False)
    target().sum().backward()
    path = tmp_path / "rollback.pt"
    _checkpoint(source, path, left)
    before, moments = copy.deepcopy(target.state_dict()), copy.deepcopy(
        right.state_dict()
    )
    gradients = [parameter.grad for parameter in target.parameters()]
    identities = [id(value) for value in (*target.parameters(), *target.buffers())]
    rng = torch.get_rng_state()
    original = (
        right.load_state_dict if stage == "optimizer" else training_state._restore_rng
    )
    calls = 0

    def fail_once(state: Any) -> Any:
        nonlocal calls
        calls += 1
        result = original(state)
        if calls == 1:
            raise RuntimeError("injected restore failure")
        return result

    if stage == "optimizer":
        monkeypatch.setattr(right, "load_state_dict", fail_once)
    else:
        monkeypatch.setattr(training_state, "_restore_rng", fail_once)
    with pytest.raises(RuntimeError, match="injected"):
        target.load_checkpoint(path, optimizer=right)
    _assert_module_state(target.state_dict(), before)
    torch.testing.assert_close(right.state_dict(), moments, atol=0, rtol=0)
    assert all(parameter.requires_grad for parameter in target.parameters())
    assert all(
        parameter.grad is gradient
        for parameter, gradient in zip(target.parameters(), gradients, strict=True)
    )
    assert identities == [
        id(value) for value in (*target.parameters(), *target.buffers())
    ]
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize(
    "kind",
    ("module_hook", "optimizer_hook", "parameter_subclass", "alias", "shared_storage"),
)
def test_unsupported_targets_are_refused_before_mutation(
    tmp_path: Path, kind: str
) -> None:
    source, target = _model(), _model()
    source()
    path = tmp_path / "guard.pt"
    _checkpoint(source, path)
    called = []
    optimizer = None
    if kind == "module_hook":
        target.register_load_state_dict_pre_hook(lambda *args: called.append(True))
    elif kind == "optimizer_hook":
        optimizer = torch.optim.AdamW(target.parameters())
        optimizer.register_load_state_dict_pre_hook(lambda *args: called.append(True))
    elif kind == "parameter_subclass":

        class CustomParameter(torch.nn.Parameter):
            pass

        target.named_parameter_groups["encoder"] = CustomParameter(
            torch.tensor(0.9, dtype=torch.float64)
        )
    elif kind == "alias":
        target.register_parameter("alias", target.parameter_groups["encoder"])
    else:
        target.named_parameter_groups["readout"] = torch.nn.Parameter(
            target.parameter_groups["encoder"].detach()
        )
    before = [parameter.detach().clone() for parameter in target.parameters()]
    with pytest.raises(TrainingStateError):
        target.load_checkpoint(path, optimizer=optimizer)
    assert not called
    for parameter, value in zip(target.parameters(), before, strict=True):
        torch.testing.assert_close(parameter, value, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ("alias", "unregistered", "duplicate"))
def test_invalid_source_optimizer_or_alias_refuses_save(
    tmp_path: Path, kind: str
) -> None:
    source = _model()
    source()
    parameters = list(source.parameters())
    if kind == "alias":
        source.register_parameter("alias", parameters[0])
    elif kind == "unregistered":
        parameters.append(torch.nn.Parameter(torch.tensor(0.5)))
    else:
        parameters.append(parameters[0])
    with pytest.warns(UserWarning) if kind == "duplicate" else nullcontext():
        optimizer = torch.optim.AdamW(parameters)
    with pytest.raises(TrainingStateError):
        _checkpoint(source, tmp_path / "refused.pt", optimizer)
    assert not (tmp_path / "refused.pt").exists()


@pytest.mark.parametrize("symbolic", (False, True))
def test_stale_source_program_refuses_atomic_replacement(
    tmp_path: Path, symbolic: bool
) -> None:
    constant = torch.tensor(0.4)
    source = (
        _model(symbolic=True)
        if symbolic
        else fq.Module(lambda p: fq.Circuit(1).ry(0, p[0] + constant), 1)
    )
    source()
    path = tmp_path / "stale.pt"
    _checkpoint(source, path)
    before = path.read_bytes()
    if symbolic:
        source.circuit_builder.x(0)
    else:
        constant.add_(0.2)
    with pytest.raises(TrainingStateError, match="stale"):
        _checkpoint(source, path)
    assert path.read_bytes() == before


def test_unqualified_tensor_literal_keeps_v1_but_refuses_frozen_save(
    tmp_path: Path,
) -> None:
    circuit = fq.Circuit(1).ry(0, fq.Parameter("theta")).rx(0, torch.tensor(0.2))
    source = fq.Module(circuit, init={"theta": 0.4})
    source()
    path = tmp_path / "legacy.pt"
    payload = _checkpoint(source, path)
    assert payload["version"] == "flagquantum.training_state.v1"
    assert "parameter_training" not in payload
    source.requires_grad_(False)
    before = path.read_bytes()
    with pytest.raises(TrainingStateError, match="qualified v3"):
        _checkpoint(source, path)
    assert path.read_bytes() == before


def test_file_write_failure_preserves_previous_checkpoint_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _model()
    source()
    path = tmp_path / "atomic.pt"
    _checkpoint(source, path)
    before = path.read_bytes()

    def fail(payload: Any, target: Path) -> None:
        target.write_bytes(b"partial")
        raise OSError("injected disk failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="injected"):
        _checkpoint(source, path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_composite_restores_classical_and_quantum_flags_in_place(
    tmp_path: Path,
) -> None:
    source, target = torch.nn.ModuleDict(
        {"quantum": _model(), "classical": torch.nn.Linear(1, 1, dtype=torch.float64)}
    ), torch.nn.ModuleDict(
        {"quantum": _model(), "classical": torch.nn.Linear(1, 1, dtype=torch.float64)}
    )
    left, right = torch.optim.AdamW(source.parameters()), torch.optim.AdamW(
        target.parameters()
    )
    for _ in range(2):
        left.zero_grad(set_to_none=True)
        source["classical"](source["quantum"]()).square().mean().backward()
        left.step()
    source_quantum = source["quantum"]
    assert isinstance(source_quantum, fq.Module)
    source_quantum.parameter_groups["encoder"].requires_grad_(False)
    source["classical"].weight.requires_grad_(False)
    path = tmp_path / "composite.pt"
    _checkpoint(source, path, left)
    identities = [id(parameter) for parameter in target.parameters()]
    training_state.load_training_checkpoint(path, module=target, optimizer=right)
    target_quantum = target["quantum"]
    assert isinstance(target_quantum, fq.Module)
    assert not target_quantum.parameter_groups["encoder"].requires_grad
    assert not target["classical"].weight.requires_grad
    assert target["classical"].bias.requires_grad
    assert identities == [id(parameter) for parameter in target.parameters()]
    _assert_module_state(target.state_dict(), source.state_dict())


@pytest.mark.parametrize("symbolic", (False, True))
def test_custom_extra_state_retains_historical_format_without_dropping_freezing(
    tmp_path: Path, symbolic: bool
) -> None:
    class Composite(torch.nn.Module):
        def __init__(self, label: str) -> None:
            super().__init__()
            self.quantum = _model(symbolic=symbolic)
            self.label = label
            self.extra_state_reads = 0

        def get_extra_state(self) -> dict[str, str]:
            self.extra_state_reads += 1
            return {"label": self.label}

        def set_extra_state(self, state: dict[str, str]) -> None:
            self.label = state["label"]

    source, target = Composite("source"), Composite("target")
    expected = source.quantum()
    path = tmp_path / "composite.pt"
    payload = _checkpoint(source, path)
    assert payload["version"] == (
        "flagquantum.training_state.v1" if symbolic else "flagquantum.training_state.v2"
    )
    assert "parameter_training" not in payload
    training_state.load_training_checkpoint(path, module=target, optimizer=None)
    torch.testing.assert_close(target.quantum(), expected)
    assert target.label == source.label

    source.quantum.parameter_groups["encoder"].requires_grad_(False)
    original = path.read_bytes()
    reads = source.extra_state_reads
    with pytest.raises(
        TrainingStateError, match="frozen parameters require a qualified v3"
    ):
        _checkpoint(source, path)
    assert path.read_bytes() == original
    assert source.extra_state_reads == reads


def test_v3_restore_refuses_custom_extra_state_before_running_callbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, target = _model(), _model()
    source()
    path = tmp_path / "v3.pt"
    _checkpoint(source, path)
    before = [parameter.detach().clone() for parameter in target.parameters()]
    identities = [id(parameter) for parameter in target.parameters()]

    def unavailable_extra_state() -> dict[str, object]:
        raise AssertionError("unsupported extra-state callbacks must not execute")

    monkeypatch.setattr(target, "get_extra_state", unavailable_extra_state)
    with pytest.raises(TrainingStateError, match="custom state loading methods"):
        target.load_checkpoint(path, optimizer=None)
    for parameter, value in zip(target.parameters(), before, strict=True):
        torch.testing.assert_close(parameter, value, atol=0, rtol=0)
    assert identities == [id(parameter) for parameter in target.parameters()]


@pytest.mark.parametrize("symbolic", (False, True))
def test_fixed_program_changes_reject_production_restore(
    tmp_path: Path, symbolic: bool
) -> None:
    source, target = _model(symbolic=symbolic), _model(symbolic=symbolic, scale=2.0)
    source()
    path = tmp_path / "workload.pt"
    _checkpoint(source, path)
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(TrainingStateError, match="workload"):
        target.load_checkpoint(path, optimizer=None)
    _assert_module_state(target.state_dict(), before)


def test_actual_optimizer_parameter_reorder_is_rejected(tmp_path: Path) -> None:
    source, target = _model(), _model()
    source()
    left = torch.optim.AdamW(source.parameters())
    right = torch.optim.AdamW(list(target.parameters())[::-1])
    path = tmp_path / "order.pt"
    _checkpoint(source, path, left)
    with pytest.raises(TrainingStateError, match="binding names or order"):
        target.load_checkpoint(path, optimizer=right)


def test_same_parameter_reused_in_gates_restores_normally(tmp_path: Path) -> None:
    def model() -> fq.Module:
        return fq.Module(
            fq.Circuit(1).ry(0, fq.Parameter("theta")).rx(0, fq.Parameter("theta")),
            init={"theta": 0.4},
        )

    source, target = model(), model()
    source.requires_grad_(False)
    source()
    path = tmp_path / "shared_gate.pt"
    _checkpoint(source, path)
    target.load_checkpoint(path, optimizer=None)
    assert len(list(target.parameters())) == 1
    assert not next(target.parameters()).requires_grad
    torch.testing.assert_close(target(), source(), atol=0, rtol=0)


@pytest.mark.parametrize("source_frozen", (False, True))
def test_correctness_debug_restores_with_frozen_target_and_later_unfreeze(
    tmp_path: Path, source_frozen: bool
) -> None:
    source, target = _model(), _model()
    source.set_runtime_policy(fq.RuntimePolicy(correctness_debug=True))
    source.requires_grad_(not source_frozen)
    target.requires_grad_(False)
    source()
    path = tmp_path / "debug.pt"
    _checkpoint(source, path)
    target.load_checkpoint(path, optimizer=None)
    assert all(
        value.requires_grad == (not source_frozen) for value in target.parameters()
    )
    target.requires_grad_(True)
    with pytest.raises(training_state.NonFiniteTrainingError, match="gradient"):
        (target().sum() * torch.tensor(float("inf"))).backward()


@pytest.mark.parametrize("kind", ("custom_circuit", "construction_state"))
def test_unqualified_workload_retains_legacy_writer_until_frozen(
    tmp_path: Path, kind: str
) -> None:
    if kind == "custom_circuit":

        class CustomCircuit(Circuit):
            pass

        source = fq.Module(
            lambda p: CustomCircuit(1).ry(0, p[0]), 1, init=torch.tensor([0.4])
        )
    else:
        circuit = fq.Circuit(
            1, inputs=torch.tensor([0.0, 1.0], dtype=torch.complex64)
        ).ry(0, fq.Parameter("theta"))
        source = fq.Module(circuit, init={"theta": 0.4})
    source()
    path = tmp_path / "unqualified.pt"
    payload = _checkpoint(source, path)
    assert payload["version"] == "flagquantum.training_state.v1"
    assert "parameter_training" not in payload
    source.requires_grad_(False)
    with pytest.raises(TrainingStateError, match="qualified v3"):
        _checkpoint(source, path)


@pytest.mark.parametrize("symbolic", (False, True))
def test_prepared_but_unconsumed_program_cannot_replace_checkpoint(
    tmp_path: Path, symbolic: bool
) -> None:
    source = _model(symbolic=symbolic)
    source()
    path = tmp_path / "unconsumed.pt"
    _checkpoint(source, path)
    before = path.read_bytes()
    if symbolic:
        source.circuit_builder.x(0)
    else:
        source.circuit_builder = _model(scale=2.0).circuit_builder
    source._build(None, source._owned_parameters())
    source._builder_bindings.clear()
    with pytest.raises(TrainingStateError, match="stale"):
        _checkpoint(source, path)
    assert path.read_bytes() == before
    source()
    assert _checkpoint(source, path)["version"] == "flagquantum.training_state.v3"


def test_invalidated_capture_refuses_save_until_reexecuted(tmp_path: Path) -> None:
    source = _model()
    source()
    path = tmp_path / "invalidated.pt"
    _checkpoint(source, path)
    source.to(dtype=torch.float64)
    with pytest.raises(TrainingStateError, match="stale"):
        _checkpoint(source, path)
    source()
    assert _checkpoint(source, path)["version"] == "flagquantum.training_state.v3"


@pytest.mark.parametrize("kind", ("overlapping", "inference"))
def test_unsupported_parameter_storage_is_refused_before_any_copy(
    tmp_path: Path, kind: str
) -> None:
    source, target = _model(flat=True), _model(flat=True)
    source()
    path = tmp_path / "storage.pt"
    _checkpoint(source, path)
    if kind == "overlapping":
        target.parameters_tensor = torch.nn.Parameter(
            torch.tensor([0.9], dtype=torch.float64).expand(2)
        )
    else:
        with torch.inference_mode():
            target.parameters_tensor = torch.nn.Parameter(
                torch.tensor([0.9, 0.8], dtype=torch.float64)
            )
    before = target.parameters_tensor.detach().clone()
    parameter = target.parameters_tensor
    with pytest.raises(TrainingStateError, match="storage"):
        target.load_checkpoint(path, optimizer=None)
    assert target.parameters_tensor is parameter
    torch.testing.assert_close(parameter, before, atol=0, rtol=0)
