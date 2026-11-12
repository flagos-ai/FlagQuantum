"""Pinned COPY/XOR data, fresh edges, phase, gradients and physical consumers."""

from collections import Counter
from dataclasses import replace
from functools import partial
from itertools import permutations

import numpy as np
import pytest
import torch

from examples.algorithms.factorized_controlled_not import (
    controlled_not_chain,
    factorized_cnot_supply,
    main,
)
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.component_superoperator_nodes import (
    component_superoperator_nodes,
)
from flagquantum.simulation.tensor_network.component_tensor_nodes import (
    component_tensor_nodes_with_labels,
)
from flagquantum.simulation.tensor_network.controlled_not import controlled_not_nodes
from flagquantum.simulation.tensor_network.register_superoperator_binding import (
    contract_register_superoperator,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    contract_register_tensors,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
@pytest.mark.parametrize("labels", list(permutations((3, 9, 1, 7))))
def test_two_source_tensor_data_axes_and_names(
    dtype: torch.dtype, labels: tuple[int, ...]
) -> None:
    from qualtran.bloqs.basic_gates import CNOT

    incoming = {"ctrl": ((labels[0],),), "target": ((labels[1],),)}
    outgoing = {"ctrl": ((labels[2],),), "target": ((labels[3],),)}
    actual = controlled_not_nodes(
        incoming, outgoing, lambda: 21, torch.empty((), dtype=dtype)
    )
    source = CNOT().my_tensors(
        {"ctrl": "ci", "target": "ti"}, {"ctrl": "co", "target": "to"}
    )
    mapping = {
        ("ci", 0): labels[0],
        ("ti", 0): labels[1],
        ("co", 0): labels[2],
        ("to", 0): labels[3],
        source[0].inds[-1]: 21,
    }
    assert len(actual) == len(source) == 2
    for node, reference in zip(actual, source, strict=True):
        assert node.labels == tuple(mapping[index] for index in reference.inds)
        assert set(reference.tags) == {node.name}
        np.testing.assert_array_equal(node.tensor.numpy(), reference.data)
        assert node.tensor.shape == (2, 2, 2) and node.tensor.dtype == dtype
        assert node.tensor.device.type == "cpu"


@pytest.mark.parametrize("count", [1, 2, 3, 7])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_native_pure_mixed_complex_gradient_and_append(
    count: int, dtype: torch.dtype
) -> None:
    from qualtran import BloqBuilder
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    import flagquantum as fq
    from examples.algorithms.binary_field_arithmetic import FieldOperation

    bb, values = BloqBuilder.from_signature(SourceSignature.build(ctrl=1, target=1))
    for _ in range(count):
        values = bb.add_d(CNOT(), **values)
    original = bb.finalize(**values)
    from qualtran.simulation.tensor import cbloq_to_quimb

    assert len(cbloq_to_quimb(original).tensors) == 2 * count
    graph = controlled_not_chain(count)
    like = torch.empty((), dtype=dtype)
    order = tuple(i for i, _, _ in iter_component_calls(graph, "cnot_chain"))
    supplier = partial(factorized_cnot_supply, like=like)
    nodes, incoming, outgoing = component_tensor_nodes_with_labels(
        graph, order, supplier, like, "cnot_chain"
    )
    signature = Signature(graph.functions[0].registers or ())
    actual = contract_register_tensors(nodes, signature, incoming, outgoing)
    expected = torch.as_tensor(original.tensor_contract(), dtype=dtype)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(nodes) == 2 * count
    assert sum(node.tensor.numel() for node in nodes) == 16 * count
    shared = Counter(label for node in nodes for label in node.labels)
    assert all(n in (1, 2) for n in shared.values())
    mixed, mi, mo = component_superoperator_nodes(
        graph, order, supplier, like, "cnot_chain"
    )
    channel = contract_register_superoperator(mixed, signature, mi, mo)
    torch.testing.assert_close(
        channel,
        torch.as_tensor(original.tensor_contract(superoperator=True), dtype=dtype),
        atol=0,
        rtol=0,
    )
    phase = torch.tensor(
        0.31,
        dtype=torch.float32 if dtype == torch.complex64 else torch.float64,
        requires_grad=True,
    )
    state = torch.arange(1, 9, dtype=phase.dtype).reshape(4, 2).to(dtype)
    state = (state + 1j * state.flip(0)) * torch.exp(1j * phase)
    output = actual @ state
    target = expected @ state
    torch.testing.assert_close(output, target, atol=0, rtol=0)
    loss = output.real.sum() + 0.7 * output.imag.sum()
    gradient = torch.autograd.grad(loss, phase, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(
        target.real.sum() + 0.7 * target.imag.sum(), phase
    )[0]
    torch.testing.assert_close(gradient, expected_gradient, atol=0, rtol=0)
    bound = bind_register_program(graph, ((2,), (0,)), entry="cnot_chain")
    assert bound.circuit.instructions == tuple(
        Instruction("cx", (2, 0)) for _ in range(count)
    )
    assert Counter(g.name for g in bound.circuit.instructions) == {"cx": count}
    assert get_cost_value(original, QECGatesCost()).clifford == count
    assert get_cost_value(original, QubitCount()) == 2
    # Physical data wires are permuted; wire 1 is a spectator, wire 3 clean work.
    initial = torch.zeros(16, dtype=dtype)
    initial[::2] = torch.complex(
        torch.arange(1, 9, dtype=phase.dtype), torch.arange(9, 17, dtype=phase.dtype)
    )
    initial /= initial.norm()
    expected_state = initial.clone()
    if count % 2:
        for index in range(16):
            target_index = index ^ (8 if index & 2 else 0)
            expected_state[target_index] = initial[index]
    actual_state = (
        fq.Circuit(4, inputs=initial, dtype=dtype)
        .append(FieldOperation(4, bound.circuit.instructions))
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual_state, expected_state, atol=0, rtol=0)
    assert torch.count_nonzero(actual_state[1::2]) == 0


@pytest.mark.parametrize("internal", [-1, True, 1.0, 0, 1, 2, 3])
def test_bad_internal_label_is_refused(internal: int) -> None:
    with pytest.raises(ValueError, match="internal tensor label"):
        controlled_not_nodes(
            {"ctrl": ((0,),), "target": ((1,),)},
            {"ctrl": ((2,),), "target": ((3,),)},
            lambda: internal,
            torch.empty(()),
        )


def test_external_alias_and_provider_body_refusal() -> None:
    with pytest.raises(ValueError, match="external tensor labels"):
        controlled_not_nodes(
            {"ctrl": ((0,),), "target": ((1,),)},
            {"ctrl": ((0,),), "target": ((3,),)},
            lambda: 4,
            torch.empty(()),
        )
    graph = controlled_not_chain(1)
    leaf = next(f for f in graph.functions if f.name == "main")
    with pytest.raises(ValueError, match="declared primitive body"):
        factorized_cnot_supply(
            replace(leaf, name="unrelated"), {}, {}, lambda: 0, like=torch.empty(())
        )


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_repeat_count_is_not_coerced(count: int) -> None:
    with pytest.raises(ValueError):
        controlled_not_chain(count)


def test_example(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    assert "6 nodes, 48 logical tensor elements" in capsys.readouterr().out
