"""Narrow parameter mappings preserve direct lookup and local execution."""

from collections.abc import Iterator, Mapping
from types import MappingProxyType

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms import pauli_term
from flagquantum.core.parameters import Parameter, bind_parameter_value
from flagquantum.runtime.executors.statevector.split_real_imag import (
    _canonical_parameter_bindings,
)
from flagquantum.runtime.executors.statevector.split_real_imag_device_double_single import (
    execute_split_real_imag_device_double_single_expectation,
    parameter_shift_split_real_imag_device_double_single_gradient,
)

pytestmark = pytest.mark.unit


class LookupOnlyBindings(Mapping[str, torch.Tensor]):
    """Reject enumeration so binding cannot quietly copy or scan the mapping."""

    def __init__(self, value: torch.Tensor) -> None:
        self.value = value
        self.lookups = 0

    def __contains__(self, key: object) -> bool:
        return key == "angle"

    def __getitem__(self, key: str) -> torch.Tensor:
        self.lookups += 1
        if key != "angle":
            raise KeyError(key)
        return self.value

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("Binding must not enumerate keys")

    def __len__(self) -> int:
        return 1


def test_binding_keeps_direct_lookup_and_tensor_identity() -> None:
    value = torch.tensor(0.3, requires_grad=True)
    bindings = LookupOnlyBindings(value)
    assert Parameter("angle").bind(bindings) is value
    assert bindings.lookups == 1
    result = bind_parameter_value(-Parameter("angle"), bindings)
    result.backward()
    torch.testing.assert_close(value.grad, torch.tensor(-1.0))
    assert bindings.lookups == 2


def test_narrow_and_readonly_mapping_types_preserve_values() -> None:
    parameter = Parameter("angle")
    value = torch.tensor(0.3, requires_grad=True)
    named: dict[str, torch.Tensor] = {"angle": value}
    symbolic: dict[Parameter, torch.Tensor] = {parameter: value}
    mixed: dict[str | Parameter, torch.Tensor] = {parameter: value, "unused": value}
    readonly: Mapping[str, torch.Tensor] = MappingProxyType(named)
    assert parameter.bind(named) is value
    assert parameter.bind(symbolic) is value
    assert parameter.bind(mixed) is value
    assert parameter.bind(readonly) is value
    assert bind_parameter_value(parameter, named) is value
    assert bind_parameter_value(parameter, symbolic) is value
    assert bind_parameter_value(parameter, mixed) is value
    assert bind_parameter_value(parameter, readonly) is value
    torch.testing.assert_close((parameter * 2).bind(named), value * 2)
    torch.testing.assert_close((parameter * 2).bind(symbolic), value * 2)
    torch.testing.assert_close((parameter * 2).bind(mixed), value * 2)
    torch.testing.assert_close((parameter * 2).bind(readonly), value * 2)


@pytest.mark.parametrize("object_first", [False, True])
def test_runtime_still_rejects_duplicate_parameter_names(object_first: bool) -> None:
    entries: list[tuple[str | Parameter, float]] = [
        ("angle", 0.2),
        (Parameter("angle"), 0.3),
    ]
    if object_first:
        entries.reverse()
    with pytest.raises(ValueError, match="duplicate or empty parameter binding"):
        _canonical_parameter_bindings(dict(entries))


def test_device_double_single_accepts_readonly_symbolic_bindings_on_cpu() -> None:
    parameter = Parameter("angle")
    circuit = Circuit(1).gate("ry", 0, theta=parameter)
    observable = pauli_term(1.0, "Z", (0,))
    angle = torch.tensor(0.23, dtype=torch.float32)
    bindings: Mapping[Parameter, torch.Tensor] = MappingProxyType({parameter: angle})
    expectation = execute_split_real_imag_device_double_single_expectation(
        circuit, observable, parameter_bindings=bindings, device="cpu", preflight=False
    )
    gradient = parameter_shift_split_real_imag_device_double_single_gradient(
        circuit, observable, parameter_bindings=bindings, device="cpu", preflight=False
    )
    reference = angle.to(torch.float64)
    assert torch.abs(expectation.cpu_float64() - torch.cos(reference)) < 1e-11
    assert torch.max(torch.abs(gradient.cpu_float64() + torch.sin(reference))) < 1e-10
