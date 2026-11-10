"""Register tensor indices feed existing native contraction without foreign graphs."""

from collections import Counter
from functools import partial
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
    contract_register_tensors,
    register_tensor_label_groups,
)
from tests.unit.test_register_tensor import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
Bindings = dict[str, tuple[tuple[int, ...], ...]]


def pauli_tensor(axis: str) -> NDArray[np.complex128]:
    return np.array(
        [[0, 1], [1, 0]] if axis == "x" else [[1, 0], [0, -1]], dtype=np.complex128
    )


def controlled_pauli(axis: str, control: int) -> tuple[Instruction, ...]:
    return (Instruction(f"c{axis}", (control, 2)),)


def source_bindings(
    signature: Any, prefix: str, labels: dict[tuple[str, int], int]
) -> tuple[dict[str, Any], Bindings]:
    source: dict[str, Any] = {}
    native: Bindings = {}
    for register in signature:
        words = []
        array = np.empty(register.shape, dtype=object)
        for index in register.all_idxs():
            token = f"{prefix}:{register.name}:{index}"
            array[index] = token
            bits = []
            for bit in range(register.bitsize):
                key = (token, bit)
                if key not in labels:
                    labels[key] = 1000 + 7 * len(labels)
                bits.append(labels[key])
            words.append(tuple(bits))
        source[register.name] = array if register.shape else array.item()
        native[register.name] = tuple(words)
    return source, native


def outer_reindex(signature: Any, ports: dict[str, Any], group: str) -> dict[Any, Any]:
    return {
        (ports[reg.name][idx] if idx else ports[reg.name], bit): (
            reg.name,
            idx,
            bit,
            group,
        )
        for reg in signature
        for idx in reg.all_idxs()
        for bit in range(reg.bitsize)
    }


@pytest.mark.parametrize("sides", list(product(("LEFT", "RIGHT", "THRU"), repeat=2)))
@pytest.mark.parametrize("shape", [(), (1, 2), (0,)])
def test_original_index_order_and_dense_outer_groups(
    sides: tuple[str, ...], shape: tuple[int, ...]
) -> None:
    import quimb.tensor as qtn
    from qualtran import QAny, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor._dense import (
        _order_incoming_outgoing_indices,
        quimb_to_dense,
    )

    source = SourceSignature(
        [
            SourceRegister("word", QAny(2), shape=shape, side=Side[sides[0]]),
            SourceRegister("bit", QAny(1), side=Side[sides[1]]),
            SourceRegister("zero", QAny(0)),
        ]
    )
    signature = native_signature(source)
    labels: dict[tuple[str, int], int] = {}
    source_in, incoming = source_bindings(source.lefts(), "in", labels)
    source_out, outgoing = source_bindings(source.rights(), "out", labels)
    ordered = _order_incoming_outgoing_indices(source, source_in, source_out)
    groups = register_tensor_label_groups(signature, incoming, outgoing)
    assert tuple(x for group in groups for x in group) == tuple(
        labels[i] for i in ordered
    )
    rng = np.random.default_rng(19)
    shape_bits = (2,) * len(ordered)
    data = rng.normal(size=shape_bits) + 1j * rng.normal(size=shape_bits)
    node = bind_register_tensor(
        torch.from_numpy(np.asarray(data)), signature, incoming, outgoing, name="leaf"
    )
    assert node.name == "leaf" and node.tensor.dtype == torch.complex128
    original = qtn.Tensor(data=data, inds=ordered)
    reindex = {
        **outer_reindex(source.lefts(), source_in, "l"),
        **outer_reindex(source.rights(), source_out, "r"),
    }
    original.reindex_(reindex)
    # Source sees a deliberately reversed outer-axis inventory.
    original.transpose_(*reversed(original.inds))
    expected = quimb_to_dense(qtn.TensorNetwork([original]), source)
    actual = contract_register_tensors((node,), signature, incoming, outgoing)
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-14, rtol=0)


@pytest.mark.parametrize("values", list(product((0, 1), repeat=2)))
def test_controlled_tensor_chain_original_contraction_append_and_all_branches(
    values: tuple[int, ...],
) -> None:
    import quimb.tensor as qtn
    from qualtran import Controlled, CtrlSpec
    from qualtran.bloqs.basic_gates import XGate, ZGate
    from qualtran.simulation.tensor._dense import quimb_to_dense

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.registers.control.control_predicate import ControlRegister
    from flagquantum.compiler.registers.control.controlled_component import (
        controlled_component_ir,
    )
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic
    from flagquantum.simulation.tensor_network.controlled_tensor import (
        controlled_register_tensor,
    )
    from tests.unit.measurement_checks import forced_branch

    sources = [Controlled(child, CtrlSpec(cvs=values)) for child in (XGate(), ZGate())]
    signature = native_signature(sources[0].signature)
    labels: dict[tuple[str, int], int] = {}
    in_src, incoming = source_bindings(sources[0].signature, "in", labels)
    mid_src, middle = source_bindings(sources[0].signature, "mid", labels)
    out_src, outgoing = source_bindings(sources[0].signature, "out", labels)
    natives = []
    originals = []
    for axis, source, old, new, old_source, new_source in zip(
        ("x", "z"),
        sources,
        (incoming, middle),
        (middle, outgoing),
        (in_src, mid_src),
        (mid_src, out_src),
        strict=True,
    ):
        dense = controlled_register_tensor(
            native_signature(source.ctrl_regs),
            Signature.build(q=1),
            (np.array(values),),
            partial(pauli_tensor, axis),
        )
        natives.append(
            bind_register_tensor(torch.from_numpy(dense), signature, old, new)
        )
        originals.extend(source.my_tensors(old_source, new_source))
    tn = qtn.TensorNetwork(originals)
    tn.reindex_(
        {
            **outer_reindex(sources[0].signature, in_src, "l"),
            **outer_reindex(sources[0].signature, out_src, "r"),
        }
    )
    expected = quimb_to_dense(tn, sources[0].signature)
    actual = contract_register_tensors(natives, signature, incoming, outgoing)
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-14, rtol=0)
    # Existing graph can replace the first child without changing the consumer.
    replacement = bind_register_tensor(
        torch.eye(8, dtype=torch.complex128), signature, incoming, middle
    )
    changed = contract_register_tensors(
        (replacement, natives[1]), signature, incoming, outgoing
    )
    np.testing.assert_allclose(
        changed.numpy(), sources[1].tensor_contract(), atol=1e-14, rtol=0
    )
    gates = tuple(
        gate
        for axis in ("x", "z")
        for gate in controlled_component_ir(
            ((0, 1),),
            (ControlRegister(1, values),),
            partial(controlled_pauli, axis),
            (3,),
            (0,),
        )
    )
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    originals_gates: list[Instruction] = []
    models = []
    for child in (XGate(), ZGate()):
        decomposed = ControlledViaAnd(child, CtrlSpec(cvs=values))
        originals_gates.extend(
            reference_arithmetic_gates(
                decomposed,
                {decomposed.ctrl_reg_names[0]: (0, 1), "q": (2,)},
                (3,),
                {3: 0},
            )
        )
        models.append(get_cost_value(decomposed, QECGatesCost()))
    assert_source(gates, tuple(originals_gates))
    counts = Counter(g.name for g in gates)
    assert counts["t"] + counts["tdg"] == sum(m.total_t_count() for m in models) == 8
    assert counts["measure"] == sum(m.measurement for m in models) == 2
    inverse = tuple(
        gate
        for axis in ("z", "x")
        for gate in controlled_component_ir(
            ((0, 1),),
            (ControlRegister(1, values),),
            partial(controlled_pauli, axis),
            (3,),
            (0,),
        )
    )
    state = torch.arange(1, 17, dtype=torch.float64).to(torch.complex128).reshape(8, 2)
    state += 1j * torch.flip(state.real, (0,))
    state /= state.norm()
    initial = torch.zeros((8, 2, 2), dtype=torch.complex128)
    initial[:, 0, :] = state
    target = actual @ state
    for history in product((0, 1), repeat=2):
        branch = forced_branch(initial.flatten(), gates, history).reshape(8, 2, 2)
        torch.testing.assert_close(branch[:, 0, :], target / 2, atol=2e-12, rtol=0)
        assert torch.count_nonzero(branch[:, 1, :]) == 0
        for reverse_history in product((0, 1), repeat=2):
            restored = forced_branch(branch.flatten(), inverse, reverse_history)
            torch.testing.assert_close(
                restored, initial.flatten() / 4, atol=2e-12, rtol=0
            )
    circuit = DynamicCircuit(5, inputs=initial.flatten(), dtype=torch.complex128)
    circuit.append(FieldOperation(5, gates))
    result = run_dynamic(circuit, shots=4, seed=29, strategy="trajectory")
    final = result.final_states.reshape(4, 8, 2, 2)
    torch.testing.assert_close(
        final[:, :, 0, :], target.expand(4, 8, 2), atol=2e-12, rtol=0
    )
    assert torch.count_nonzero(final[:, :, 1, :]) == 0
    circuit.append(FieldOperation(5, inverse))
    roundtrip = run_dynamic(circuit, shots=4, seed=31, strategy="trajectory")
    torch.testing.assert_close(
        roundtrip.final_states.reshape(4, -1),
        initial.flatten().expand(4, -1),
        atol=2e-12,
        rtol=0,
    )


def test_binding_retains_autograd_and_caller_storage() -> None:
    theta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    c, s = torch.cos(theta), torch.sin(theta)
    matrix = torch.stack((c, -s, s, c)).reshape(2, 2).to(torch.complex128)
    signature = Signature.build(q=1)
    node = bind_register_tensor(matrix, signature, {"q": ((101,),)}, {"q": ((205,),)})
    assert node.tensor.data_ptr() == matrix.data_ptr()
    initial = TensorNetworkNode(torch.tensor([1, 0], dtype=torch.complex128), (101,))
    from flagquantum.core.ir.program.registers import Register, RegisterSide

    state_signature = Signature(
        [Register("q", signature[0].dtype, side=RegisterSide.RIGHT)]
    )
    result = contract_register_tensors(
        (node, initial), state_signature, {}, {"q": ((205,),)}
    )
    torch.testing.assert_close(result, torch.stack((c, s)).to(torch.complex128))
    (gradient,) = torch.autograd.grad(result[1].abs().square(), (theta,))
    torch.testing.assert_close(gradient, 2 * torch.sin(theta) * torch.cos(theta))


def test_scalar_and_zero_bit_ports_need_no_connections() -> None:
    from flagquantum.core.ir.program.registers import Register, RegisterType

    signature = Signature(
        [
            Register("zero", RegisterType("opaque", 0)),
            Register("empty", RegisterType("opaque", 2), (0,)),
        ]
    )
    node = bind_register_tensor(
        torch.tensor(2 + 3j, dtype=torch.complex128), signature, {}, {}
    )
    assert node.labels == ()
    actual = contract_register_tensors((node,), signature, {}, {})
    assert actual.shape == () and actual == 2 + 3j
    with pytest.raises(ValueError, match="at least one"):
        contract_register_tensors((), signature, {}, {})


@pytest.mark.parametrize(
    "bindings", [{}, {"q": ()}, {"q": ((1, 2),)}, {"q": ((-1,),)}, {"q": ((True,),)}]
)
def test_invalid_word_bindings_fail_before_contraction(bindings: Bindings) -> None:
    with pytest.raises((ValueError, KeyError)):
        bind_register_tensor(
            torch.eye(2), Signature.build(q=1), bindings, {"q": ((9,),)}
        )


def test_invalid_data_size_does_not_enter_the_contraction_engine() -> None:
    with pytest.raises(ValueError, match="dense tensor size"):
        bind_register_tensor(
            torch.zeros(3), Signature.build(q=1), {"q": ((1,),)}, {"q": ((2,),)}
        )
