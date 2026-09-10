from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, fields

import pytest

from flagquantum.core.ir import CircuitIR
from flagquantum.core.runtime_config import RuntimeConfig
from flagquantum.runtime.options import ExecutionOptions
from flagquantum.runtime.options_resolver import (
    circuit_execution_constraints,
    resolve_execution_options,
)
from flagquantum.runtime.policy import RuntimePolicy

pytestmark = pytest.mark.unit


def test_execution_options_exact_shape_and_immutability() -> None:
    options = ExecutionOptions()

    assert [field.name for field in fields(options)] == [
        "mode",
        "backend",
        "device",
        "target",
        "batch_size",
        "precision",
        "shots",
        "seed",
        "memory_limit_bytes",
        "require_gradients",
        "allow_approximate",
        "allow_backend_fallback",
    ]
    assert all(getattr(options, field.name) is None for field in fields(options))
    assert not hasattr(options, "__dict__")
    with pytest.raises(FrozenInstanceError):
        options.mode = "mps"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"mode": "distributed_statevector"}, ValueError),
        ({"target": "full_state"}, ValueError),
        ({"precision": "float32"}, ValueError),
        ({"backend": ""}, ValueError),
        ({"batch_size": True}, TypeError),
        ({"shots": 0}, ValueError),
        ({"seed": -1}, ValueError),
        ({"memory_limit_bytes": 0}, ValueError),
        ({"require_gradients": 1}, TypeError),
    ],
)
def test_execution_options_reject_invalid_values(kwargs, error) -> None:
    with pytest.raises(error):
        ExecutionOptions(**kwargs)


def test_execution_options_serialization_is_strict_and_round_trips() -> None:
    options = ExecutionOptions(mode="mps", shots=100, allow_approximate=True)
    payload = options.to_dict()

    assert ExecutionOptions.from_dict(payload) == options
    with pytest.raises(ValueError, match="unknown execution options"):
        ExecutionOptions.from_dict({**payload, "extras": {}})
    incomplete = dict(payload)
    incomplete.pop("seed")
    with pytest.raises(ValueError, match="missing execution options"):
        ExecutionOptions.from_dict(incomplete)


def test_resolver_applies_one_field_level_precedence_chain() -> None:
    resolved = resolve_execution_options(
        ExecutionOptions(mode="mps", device="cuda:1"),
        policy_options=ExecutionOptions(
            mode="statevector", require_gradients=True, batch_size=2
        ),
        program_constraints=ExecutionOptions(batch_size=2, precision="complex128"),
        runtime_config=RuntimeConfig(backend="jax", device="cuda:0"),
    )

    assert resolved.mode == "mps"
    assert resolved.backend == "jax"
    assert resolved.device == "cuda:1"
    assert resolved.batch_size == 2
    assert resolved.precision == "complex128"
    assert resolved.require_gradients is True
    assert resolved.allow_approximate is False
    assert resolved.source_for("mode") == "call"
    assert resolved.source_for("precision") == "program_constraints"
    assert resolved.source_for("allow_approximate") == "framework_defaults"


def test_resolver_rejects_program_batch_conflict() -> None:
    with pytest.raises(ValueError, match="program batch constraint"):
        resolve_execution_options(
            ExecutionOptions(batch_size=3),
            program_constraints=ExecutionOptions(batch_size=2),
        )


def test_resolver_rejects_program_precision_demotion() -> None:
    with pytest.raises(ValueError, match="would demote a complex128 program"):
        resolve_execution_options(
            ExecutionOptions(precision="complex64"),
            program_constraints=ExecutionOptions(precision="complex128"),
        )


def test_circuit_ir_precision_is_a_program_constraint() -> None:
    constraints = circuit_execution_constraints(
        CircuitIR(n_wires=1, instructions=(), dtype="complex128")
    )

    assert constraints.precision == "complex128"
    assert constraints.batch_size is None
    resolved = resolve_execution_options(program_constraints=constraints)
    assert resolved.precision == "complex128"
    assert resolved.source_for("precision") == "program_constraints"


def test_resolver_requires_typed_options_instead_of_kwargs_escape_hatch() -> None:
    with pytest.raises(TypeError, match="ExecutionOptions"):
        resolve_execution_options({"mode": "mps"})  # type: ignore[arg-type]


def test_runtime_policy_has_one_execution_source_and_round_trips() -> None:
    policy = RuntimePolicy(
        execution_options=ExecutionOptions(mode="mps", require_gradients=True),
        observable="z_sum",
        observable_wires=(0, 1),
        correctness_debug=True,
    )

    assert [field.name for field in fields(policy)] == [
        "execution_options",
        "observable",
        "observable_wires",
        "correctness_debug",
    ]
    assert policy.mode == "mps"
    assert policy.backend == "pytorch"
    assert policy.allow_backend_fallback is False
    assert RuntimePolicy.from_dict(policy.to_dict()) == policy


@pytest.mark.parametrize("source", ("options", "policy_options", "program_constraints"))
@pytest.mark.parametrize("extra", (None, 0, 2))
def test_resolver_does_not_ignore_explicit_subclass_fields(
    source: str, extra: int | None
) -> None:
    @dataclass(frozen=True)
    class ExtendedOptions(ExecutionOptions):
        extra: int | None = None

    supplied = ExtendedOptions(mode="mps", extra=extra)
    if extra is None:
        resolved = resolve_execution_options(**{source: supplied})
        assert resolved.mode == "mps"
    else:
        with pytest.raises(TypeError, match="extra"):
            resolve_execution_options(**{source: supplied})
