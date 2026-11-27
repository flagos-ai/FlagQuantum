"""Shared input semantics across static specialization and dynamic lowering."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from flagquantum.compiler.frontends._structured import (
    BOOL,
    INDEX,
    HybridProgram,
    SpecializationError,
    capture_source,
    lower_dynamic_program,
    runtime_inputs,
    scalar_type,
    specialize,
    specialize_program,
    tensor_type,
)
from flagquantum.compiler.frontends._structured.diagnostics import (
    SpecializationError as SharedSpecializationError,
)

pytestmark = pytest.mark.unit

SCALAR_SIGNATURE = "f0b9233d72602fbb8d7d428c31bb5088b876c548f4d332e2f4b810a1d057cc5a"
TENSOR_SIGNATURE = "900733014ca0cb773e4bf3fbf1ed55e020a39b3cb0be253924d447ffd5d612f4"


def _program(dynamic: bool) -> HybridProgram:
    returned = (
        "    bit = qp.measure(wires=0)\n    return bit\n"
        if dynamic
        else "    return qp.expval(qp.PauliZ(0))\n"
    )
    return capture_source(
        "def rotate(theta, wire, enabled):\n"
        "    if enabled:\n"
        "        qp.RX(theta, wires=wire)\n" + returned,
        (scalar_type("float64"), INDEX, BOOL),
    )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("optimize", [False, True])
def test_input_signature_preserves_baseline_and_excludes_parameter_values(
    dynamic: bool, optimize: bool
) -> None:
    program = _program(dynamic)
    for angle in (0.2, -0.7):
        theta = torch.tensor(angle, dtype=torch.float64)
        inputs = (theta, 1, True)
        if dynamic:
            lowered = lower_dynamic_program(
                program, inputs, circuit_dtype="complex128", optimize=optimize
            )
            assert lowered.input_signature_identity == SCALAR_SIGNATURE
            instruction = lowered.bind().instructions[0]
            assert (instruction.name, instruction.wires) == ("rx", (1,))
            assert instruction.params["theta"] is theta
        else:
            trace = specialize_program(program, inputs, optimize=optimize)
            assert trace.input_signature_identity == SCALAR_SIGNATURE
            assert tuple(gate.structure() for gate in trace.gates) == (
                {"name": "rx", "wires": [1], "parameterized": True},
            )
            assert trace.gates[0].parameter is theta


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize(
    ("inputs", "code"),
    [
        ((), "input.arity"),
        ((torch.zeros(2, dtype=torch.float64), 1, True), "input.scalar"),
        ((torch.tensor(0.2, dtype=torch.float32), 1, True), "input.dtype"),
        ((0.2, True, True), "input.index"),
        ((0.2, 1.5, True), "input.index"),
        ((0.2, torch.tensor(1.5), True), "input.index"),
        ((0.2, torch.tensor([1]), True), "input.scalar"),
        ((0.2, 1, 1), "input.bool"),
        ((0.2, 1, torch.tensor(1.0)), "input.bool"),
        ((0.2, 1, torch.tensor([True])), "input.scalar"),
    ],
)
def test_both_entries_preserve_shared_input_rejections(
    dynamic: bool, optimize: bool, inputs: tuple[Any, ...], code: str
) -> None:
    with pytest.raises(specialize.SpecializationError) as caught:
        if dynamic:
            lower_dynamic_program(
                _program(dynamic),
                inputs,
                circuit_dtype="complex128",
                optimize=optimize,
            )
        else:
            specialize_program(_program(dynamic), inputs, optimize=optimize)

    assert type(caught.value) is SharedSpecializationError
    assert caught.value.code == code
    assert caught.value.location is None
    assert str(caught.value).startswith(f"{code}: ")


def test_tensor_signature_preserves_baseline_and_concrete_shape() -> None:
    declared = (tensor_type("float64", (None, 4)),)
    first = torch.zeros((2, 4), dtype=torch.float64)
    changed_values = torch.ones_like(first)
    changed_shape = torch.zeros((3, 4), dtype=torch.float64)

    assert runtime_inputs._input_signature((first,), declared) == TENSOR_SIGNATURE
    assert (
        runtime_inputs._input_signature((changed_values,), declared) == TENSOR_SIGNATURE
    )
    assert (
        runtime_inputs._input_signature((changed_shape,), declared) != TENSOR_SIGNATURE
    )


def test_retained_input_helpers_and_exception_are_direct_aliases() -> None:
    assert SpecializationError is SharedSpecializationError
    assert specialize.SpecializationError is SharedSpecializationError
    assert specialize._normalized_dtype is runtime_inputs._normalized_dtype
    assert specialize._runtime_shape is runtime_inputs._runtime_shape
    assert specialize._runtime_device_class is runtime_inputs._runtime_device_class
    assert specialize._input_descriptor is runtime_inputs._input_descriptor
    assert specialize._input_signature is runtime_inputs._input_signature
