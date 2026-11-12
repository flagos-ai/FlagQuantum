"""Fresh-label pure dense composition through the existing numerical consumers."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace

import pytest
import torch

from examples.algorithms.bell_tensor_workflow import bell_program, bell_providers
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.component_tensor_expansion import (
    expand_component_root_for_tensor,
)
from flagquantum.compiler.registers.layout.component_workspace import (
    allocate_word_function,
)
from flagquantum.core.ir.program import FunctionRecord
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.component_dense import (
    contract_component_tensors_with_labels,
)
from flagquantum.simulation.tensor_network.component_tensor_nodes import (
    component_tensor_nodes_with_labels,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    contract_register_tensors,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
Ports = Mapping[str, tuple[tuple[int, ...], ...]]


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("factorized", [False, True])
@pytest.mark.parametrize("atol", [None, 0.0, 1e-12])
@pytest.mark.parametrize("wide", [False, True])
def test_factorized_provider_direct_dense_entry_values_phase_and_replacement(
    prepared: bool,
    factorized: bool,
    atol: float | None,
    wide: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.simulation.tensor_network.full_simplification as policy

    record = bell_program(prepared=prepared)
    if wide:
        registers = record.functions[0].registers or ()
        dtype = RegisterType("uint", 9)
        anc = allocate_word_function("anc_zero", dtype, False)
        builder = ComponentProgramBuilder(
            Signature((*registers, Register("anc", dtype, side=RegisterSide.RIGHT))),
            (*record.functions, anc),
            name="wide_bell",
        )
        names = ("q0", "q1") if prepared else ("q1", "q2")
        result = builder.call_dict(
            "bell", {} if prepared else {name: builder.inputs[name] for name in names}
        )
        result["anc"] = builder.call_dict("anc_zero", {})["reg"]
        record = builder.finalize(result)
    like = torch.empty((), dtype=torch.complex128)
    providers = bell_providers(record, like, factorized_cnot=factorized)
    selected, _ = expand_component_root_for_tensor(
        record, providers.has_custom_tensor, "wide_bell" if wide else "bell"
    )
    order = tuple(i for i, _, _ in iter_component_calls(selected, "tensor_root"))
    phase = torch.tensor(0.29, dtype=torch.float64, requires_grad=True)
    allocated: list[int] = []

    def supply(
        function: FunctionRecord,
        incoming: Ports,
        outgoing: Ports,
        fresh: Callable[[], int],
    ) -> Iterable[TensorNetworkNode | int]:
        def allocate() -> int:
            label = fresh()
            allocated.append(label)
            return label

        nodes = tuple(providers.supply(function, incoming, outgoing, allocate))
        assert all(isinstance(node, TensorNetworkNode) for node in nodes)
        if function.name in ("h", "plus"):
            return tuple(
                (
                    replace(node, tensor=node.tensor * torch.exp(1j * phase))
                    if isinstance(node, TensorNetworkNode)
                    else node
                )
                for node in nodes
            )
        return nodes

    calls: list[int] = []
    original = policy.simplify_tensor_network

    def simplify(
        nodes: Sequence[TensorNetworkNode],
        outputs: Sequence[int] | None = None,
        *,
        atol: float = 1e-12,
    ) -> tuple[TensorNetworkNode, ...]:
        calls.append(1)
        return original(nodes, outputs, atol=atol)

    # The callable boundary is instrumented only; production keeps its typed owner.
    monkeypatch.setattr(policy, "simplify_tensor_network", simplify)
    actual = contract_component_tensors_with_labels(
        selected, order, supply, like, "tensor_root", simplification_atol=atol
    )
    nodes, incoming, outgoing = component_tensor_nodes_with_labels(
        selected, order, supply, like, "tensor_root"
    )
    expected = contract_register_tensors(
        nodes, Signature(selected.functions[0].registers or ()), incoming, outgoing
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-14)
    torch.testing.assert_close(
        torch.autograd.grad(actual.imag.sum(), phase, retain_graph=True)[0],
        torch.autograd.grad(expected.imag.sum(), phase)[0],
        rtol=0,
        atol=2e-14,
    )
    assert bool(allocated) == factorized
    # Wide preparation has11 outputs; wide unitary has11 outputs+2 inputs.
    assert len(calls) == int(wide and not prepared and atol is not None)
    if wide:
        assert actual.numel() == (2048 if prepared else 8192)
        assert not bool(actual.reshape(4, 512, -1)[:, 1:, :].any())


def test_supplier_exception_identity_and_no_retries() -> None:
    record = bell_program(prepared=True)
    providers = bell_providers(record, torch.empty((), dtype=torch.complex128))
    selected, _ = expand_component_root_for_tensor(
        record, providers.has_custom_tensor, "bell"
    )
    order = tuple(i for i, _, _ in iter_component_calls(selected, "tensor_root"))
    failure = RuntimeError("provider failure sentinel")
    count = 0

    def fail(
        function: FunctionRecord,
        incoming: Ports,
        outgoing: Ports,
        fresh: Callable[[], int],
    ) -> Iterable[TensorNetworkNode | int]:
        nonlocal count
        count += 1
        raise failure

    with pytest.raises(RuntimeError) as captured:
        contract_component_tensors_with_labels(
            selected,
            order,
            fail,
            torch.empty((), dtype=torch.complex128),
            "tensor_root",
            simplification_atol=0,
        )
    assert captured.value is failure and count == 1
