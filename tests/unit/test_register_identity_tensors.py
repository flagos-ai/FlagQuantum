"""Direct-through words retain independent outer axes and joint amplitudes."""

from collections import Counter
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_identity import (
    register_identity_nodes,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
    contract_register_tensors,
)
from tests.unit.test_register_tensor import native_signature
from tests.unit.test_register_tensor_binding import source_bindings

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", [0, 1, 2, 3])
@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,)])
def test_original_placeholder_count_and_dense_identity(
    width: int, shape: tuple[int, ...]
) -> None:
    from qualtran import BloqBuilder, QAny
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor._dense import quimb_to_dense
    from qualtran.simulation.tensor._quimb import cbloq_to_quimb

    source_sig = SourceSignature([SourceRegister("idle", QAny(width), shape=shape)])
    bb, data = BloqBuilder.from_signature(source_sig)
    source = bb.finalize(**data)
    original = cbloq_to_quimb(source)
    labels: dict[tuple[str, int], int] = {}
    _, incoming = source_bindings(source_sig, "in", labels)
    _, outgoing = source_bindings(source_sig, "out", labels)
    signature = native_signature(source_sig)
    nodes = register_identity_nodes(
        signature, incoming, outgoing, torch.empty((), dtype=torch.complex128)
    )
    count = width * len(tuple(source_sig[0].all_idxs()))
    assert len(nodes) == len(original.tensors) == count
    assert all(node.tensor.shape == (2, 2) for node in nodes)
    assert all(node.tensor.dtype == torch.complex128 for node in nodes)
    assert all(np.array_equal(t.data, np.eye(2)) for t in original.tensors)
    if not count:
        assert nodes == ()
        return
    actual = contract_register_tensors(nodes, signature, incoming, outgoing)
    expected = quimb_to_dense(original, source_sig)
    np.testing.assert_array_equal(actual.numpy(), expected)
    torch.testing.assert_close(actual, torch.eye(1 << count, dtype=torch.complex128))


@pytest.mark.parametrize("values", list(product((0, 1), repeat=2)))
def test_idle_word_with_controlled_operation_all_branches_and_append(
    values: tuple[int, ...],
) -> None:
    from qualtran import BloqBuilder, Controlled, CtrlSpec, QUInt
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.simulation.tensor._dense import quimb_to_dense
    from qualtran.simulation.tensor._quimb import cbloq_to_quimb

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.registers.control.control_predicate import ControlRegister
    from flagquantum.compiler.registers.control.controlled_component import (
        controlled_component_ir,
    )
    from flagquantum.core.ir import Instruction
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic
    from flagquantum.simulation.tensor_network.controlled_tensor import (
        controlled_register_tensor,
    )
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    child = Controlled(XGate(), CtrlSpec(cvs=values))
    source_sig = SourceSignature([*child.signature, SourceRegister("idle", QUInt(2))])
    bb, data = BloqBuilder.from_signature(source_sig)
    returned = bb.add_d(
        child, **{name: data[name] for name in (child.ctrl_reg_names[0], "q")}
    )
    source = bb.finalize(**returned, idle=data["idle"])
    signature = native_signature(source_sig)
    labels: dict[tuple[str, int], int] = {}
    _, incoming = source_bindings(source_sig, "in", labels)
    _, outgoing = source_bindings(source_sig, "out", labels)
    dense = controlled_register_tensor(
        native_signature(child.ctrl_regs),
        Signature.build(q=1),
        (np.array(values),),
        lambda: np.array([[0, 1], [1, 0]], dtype=np.complex128),
    )
    active = bind_register_tensor(
        torch.from_numpy(dense), native_signature(child.signature), incoming, outgoing
    )
    idle_signature = Signature([signature[-1]])
    placeholders = register_identity_nodes(
        idle_signature, incoming, outgoing, active.tensor
    )
    actual = contract_register_tensors(
        (active, *placeholders), signature, incoming, outgoing
    )
    original = cbloq_to_quimb(source)
    assert len(original.tensors) == 3  # one child and two direct bit identities
    expected = quimb_to_dense(original, source_sig)
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-14, rtol=0)
    # Replacing the active child with identity preserves the unused-word consumer.
    replacement = bind_register_tensor(
        torch.eye(8, dtype=torch.complex128),
        native_signature(child.signature),
        incoming,
        outgoing,
    )
    torch.testing.assert_close(
        contract_register_tensors(
            (replacement, *placeholders), signature, incoming, outgoing
        ),
        torch.eye(32, dtype=torch.complex128),
    )
    gates = controlled_component_ir(
        ((0, 1),),
        (ControlRegister(1, values),),
        lambda c: (Instruction("cx", (c, 2)),),
        (5,),
        (0,),
    )
    decomposed = ControlledViaAnd(XGate(), child.ctrl_spec)
    oracle = reference_arithmetic_gates(
        decomposed, {decomposed.ctrl_reg_names[0]: (0, 1), "q": (2,)}, (5,), {5: 0}
    )
    assert_source(gates, oracle)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(decomposed, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4
    assert counts["measure"] == model.measurement == 1
    assert get_cost_value(decomposed, QubitCount()) == 4
    state = torch.complex(
        torch.arange(1, 33, dtype=torch.float64),
        torch.arange(32, 0, -1, dtype=torch.float64),
    )
    state /= state.norm()
    initial = torch.zeros(64, dtype=torch.complex128)
    initial[::2] = state
    target = actual @ state
    for history in ((0,), (1,)):
        branch = forced_branch(initial, gates, history)
        torch.testing.assert_close(branch[::2], target / np.sqrt(2), atol=2e-12, rtol=0)
        assert torch.count_nonzero(branch[1::2]) == 0
        for reverse in ((0,), (1,)):
            torch.testing.assert_close(
                forced_branch(branch, gates, reverse), initial / 2, atol=2e-12, rtol=0
            )
    circuit = DynamicCircuit(6, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(6, gates))
    first = run_dynamic(circuit, shots=4, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        first.final_states.reshape(4, 64)[:, ::2],
        target.expand(4, -1),
        atol=2e-12,
        rtol=0,
    )
    assert torch.count_nonzero(first.final_states.reshape(4, 64)[:, 1::2]) == 0
    circuit.append(FieldOperation(6, gates))
    second = run_dynamic(circuit, shots=4, seed=43, strategy="trajectory")
    torch.testing.assert_close(
        second.final_states.reshape(4, 64), initial.expand(4, -1), atol=2e-12, rtol=0
    )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_identity_nodes_preserve_caller_dtype_device_and_autograd(
    dtype: torch.dtype,
) -> None:
    signature = Signature.build(q=1)
    value = torch.tensor([0.2, 0.7], dtype=dtype, requires_grad=True)
    nodes = register_identity_nodes(signature, {"q": ((10,),)}, {"q": ((20,),)}, value)
    assert nodes[0].tensor.dtype == dtype and nodes[0].tensor.device == value.device
    ket = TensorNetworkNode(value, (10,))
    state_signature = Signature(
        [Register("q", signature[0].dtype, side=RegisterSide.RIGHT)]
    )
    result = contract_register_tensors(
        (*nodes, ket), state_signature, {}, {"q": ((20,),)}
    )
    torch.testing.assert_close(result, value)
    (gradient,) = torch.autograd.grad(result.abs().square().sum(), (value,))
    torch.testing.assert_close(gradient, 2 * value)


@pytest.mark.parametrize("side", [RegisterSide.LEFT, RegisterSide.RIGHT])
def test_directional_port_is_not_a_direct_through_connection(
    side: RegisterSide,
) -> None:
    with pytest.raises(ValueError, match="direct-through"):
        register_identity_nodes(
            Signature([Register("q", RegisterType("bit", 1), side=side)]),
            {},
            {},
            torch.empty(()),
        )


@pytest.mark.parametrize(
    "outgoing", [{"q": ((10,),)}, {"q": ((20, 20),)}, {"q": ()}, {}]
)
def test_invalid_binding_or_external_alias_rejects(outgoing: Any) -> None:
    with pytest.raises((ValueError, KeyError)):
        register_identity_nodes(
            Signature.build(q=1), {"q": ((10,),)}, outgoing, torch.empty(())
        )
