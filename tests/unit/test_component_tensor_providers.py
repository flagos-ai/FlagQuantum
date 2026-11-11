"""Explicit provider bindings drive both tensor-leaf selection and supply."""

from collections.abc import Iterable, Mapping
from typing import Any, cast

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.component_tensor_expansion import (
    expand_components_for_tensor,
)
from flagquantum.core.ir.program import FunctionRecord
from flagquantum.simulation.tensor_network.component_dense import (
    contract_component_superoperator,
    contract_component_tensors,
)
from flagquantum.simulation.tensor_network.component_tensor_providers import (
    ComponentTensorProviders,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from tests.unit.test_component_superoperator_nodes import mixed_graph, supplier
from tests.unit.test_component_tensor_nodes import fixture, h_supplier

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

Ports = Mapping[str, tuple[tuple[int, ...], ...]]


def test_one_explicit_binding_selects_and_supplies_the_same_leaf() -> None:
    record, source = fixture(1, ())
    calls: list[str] = []

    def provide(
        function: FunctionRecord, incoming: Ports, outgoing: Ports
    ) -> Iterable[TensorNetworkNode | int]:
        calls.append(function.name)
        return h_supplier(function, incoming, outgoing)

    original = {"h": provide}
    providers = ComponentTensorProviders(original)
    original.clear()
    assert providers.has_custom_tensor("h")
    assert not providers.has_custom_tensor("missing")

    selected, _instance_ids = expand_components_for_tensor(
        record, providers.has_custom_tensor, full_flatten=False
    )
    like = torch.empty((), dtype=torch.complex128)
    actual = contract_component_tensors(
        selected,
        tuple(i for i, _, _ in iter_component_calls(selected)),
        providers.supply,
        like,
    )
    np.testing.assert_allclose(
        actual.numpy(), source.tensor_contract(), atol=2e-14, rtol=0
    )
    assert calls == ["h", "h", "h"]
    assert len(selected.functions[0].blocks[0].operations) == 3


def test_exact_function_keys_do_not_use_python_method_name_heuristics() -> None:
    providers = ComponentTensorProviders({"Hadamard": h_supplier})
    assert providers.has_custom_tensor("Hadamard")
    assert not providers.has_custom_tensor("h")
    assert not providers.has_custom_tensor("hadamard")


def test_provider_table_rejects_invalid_keys_and_values() -> None:
    with pytest.raises(ValueError, match="non-empty function names"):
        ComponentTensorProviders({"": h_supplier})
    with pytest.raises(TypeError, match="must be callable"):
        ComponentTensorProviders({"h": cast(Any, object())})


def test_provider_replacement_preserves_consumer_and_tensor_gradient() -> None:
    record, _ = fixture(1, ())
    scale = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)

    def alternate(
        function: FunctionRecord, incoming: Ports, outgoing: Ports
    ) -> Iterable[TensorNetworkNode | int]:
        node = h_supplier(function, incoming, outgoing)[0]
        yield TensorNetworkNode(node.tensor * scale, node.labels)
        yield TensorNetworkNode(1 / scale, ())

    def consume(provider: ComponentTensorProviders[[Ports, Ports]]) -> torch.Tensor:
        selected, _instance_ids = expand_components_for_tensor(
            record, provider.has_custom_tensor, full_flatten=False
        )
        return contract_component_tensors(
            selected,
            tuple(i for i, _, _ in iter_component_calls(selected)),
            provider.supply,
            torch.empty((), dtype=torch.complex128),
        )

    baseline = consume(ComponentTensorProviders({"h": h_supplier}))
    replacement = consume(ComponentTensorProviders({"h": alternate}))
    torch.testing.assert_close(replacement, baseline, atol=1e-14, rtol=0)
    gradient = torch.autograd.grad(replacement.real.sum(), scale)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-14, rtol=0)


def test_missing_selected_provider_is_explicit_and_chained() -> None:
    record, _ = fixture(1, ())
    function = next(function for function in record.functions if function.name == "h")
    providers: ComponentTensorProviders[[Ports, Ports]] = ComponentTensorProviders({})
    with pytest.raises(
        ValueError, match="no tensor provider registered.*'h'"
    ) as failure:
        tuple(providers.supply(function, {}, {}))
    assert isinstance(failure.value.__cause__, KeyError)


def test_provider_callback_exception_identity_is_preserved() -> None:
    record, _ = fixture(1, ())
    function = next(function for function in record.functions if function.name == "h")
    failure = RuntimeError("provider callback failed")

    def fail(
        function: FunctionRecord, incoming: Ports, outgoing: Ports
    ) -> Iterable[TensorNetworkNode | int]:
        raise failure

    providers = ComponentTensorProviders({"h": fail})
    with pytest.raises(RuntimeError) as actual:
        tuple(providers.supply(function, {}, {}))
    assert actual.value is failure


def test_mixed_provider_receives_shared_fresh_label_callback() -> None:
    record, source = mixed_graph(discard=False)
    environments: list[int] = []
    selected_supplier = supplier(
        torch.empty((), dtype=torch.complex128), environments=environments
    )
    providers = ComponentTensorProviders(
        {"first": selected_supplier, "second": selected_supplier}
    )
    selected = expand_components_for_tensor(
        record, providers.has_custom_tensor, full_flatten=False
    )
    actual = contract_component_superoperator(
        selected[0],
        tuple(i for i, _, _ in iter_component_calls(selected[0])),
        providers.supply,
        torch.empty((), dtype=torch.complex128),
    )
    from qualtran.simulation.tensor._dense import bloq_to_dense

    np.testing.assert_allclose(
        actual.numpy(),
        bloq_to_dense(source, full_flatten=False, superoperator=True),
        atol=2e-14,
        rtol=0,
    )
    assert len(environments) == len(set(environments)) == 2
