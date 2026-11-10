"""Per-bit zero kets connect native initialized graphs to original pure states."""

from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.component_tensor_expansion import (
    expand_components_for_tensor,
)
from flagquantum.compiler.registers.component_zero_initialization import (
    initialize_component_from_zero,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.component_tensor_nodes import (
    component_tensor_nodes,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
    contract_register_tensors,
)
from flagquantum.simulation.tensor_network.register_zero_state import (
    register_zero_state_nodes,
)
from tests.unit.test_component_tensor_nodes import h_supplier
from tests.unit.test_component_zero_initialization import phase_fixture

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def supplier(
    function: FunctionRecord, inc: Any, out: Any
) -> tuple[TensorNetworkNode, ...]:
    sig = Signature(function.registers or ())
    if function.name.startswith("__fq_zero_"):
        return register_zero_state_nodes(
            sig, out, torch.empty((), dtype=torch.complex128)
        )
    if function.name == "h":
        return h_supplier(function, inc, out)
    assert function.name == "s"
    return (
        bind_register_tensor(
            torch.diag(torch.tensor([1, 1j], dtype=torch.complex128)), sig, inc, out
        ),
    )


def contract(record: Any, supply: Any = supplier) -> torch.Tensor:
    order = tuple(i for i, _, _ in iter_component_calls(record, "from_zero"))
    nodes, inc, out = component_tensor_nodes(
        record,
        order,
        supply,
        torch.empty((), dtype=torch.complex128),
        entry="from_zero",
    )
    assert inc == {}
    sig = Signature(
        next(f for f in record.functions if f.name == "from_zero").registers or ()
    )
    return contract_register_tensors(nodes, sig, inc, out)


@pytest.mark.parametrize("width", [0, 1, 2])
@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,)])
def test_initialized_phase_arrays_full_complex_state_and_append(
    width: int, shape: tuple[int, ...]
) -> None:
    from qualtran.simulation.tensor._dense import bloq_to_dense
    from qualtran.simulation.tensor._quimb import initialize_from_zero

    import flagquantum as fq
    from examples.algorithms.binary_field_arithmetic import FieldOperation

    record, source = phase_fixture(width, shape)
    initialized = initialize_component_from_zero(record)
    selected, _ = expand_components_for_tensor(
        initialized,
        lambda name: name.startswith("__fq_zero_") or name in {"h", "s"},
        entry="from_zero",
        full_flatten=False,
    )
    actual = contract(selected)
    if shape == (0,):
        with pytest.raises(IndexError, match="out of bounds"):
            initialize_from_zero(source)
        expected = np.kron(
            np.array([1 + 1j, 1 - 1j]) / 2, np.array([1, 1j]) / np.sqrt(2)
        )
    else:
        original = initialize_from_zero(source)
        # Full decomposition visits source ZeroState and Join for IntState.
        if width == 0:
            from qualtran._infra.composite_bloq import BloqError

            with pytest.raises(BloqError, match="dtypes are not consistent"):
                bloq_to_dense(original, full_flatten=True)
            expected = np.kron(
                np.array([1 + 1j, 1 - 1j]) / 2, np.array([1, 1j]) / np.sqrt(2)
            )
        else:
            expected = bloq_to_dense(original, full_flatten=True)
    np.testing.assert_allclose(actual.numpy(), expected, atol=3e-14, rtol=0)
    n = 2 + width * int(np.prod(shape))
    bound = bind_register_program(
        initialized,
        (),
        entry="from_zero",
        clean=tuple(range(n + 1)),
        allow_workspace_outputs=True,
    )
    state = (
        fq.Circuit(n + 1, dtype=torch.complex128)
        .append(FieldOperation(n + 1, bound.circuit.instructions))
        .state()
        .flatten()
    )
    returned = tuple(wire for word in bound.outputs for wire in word)
    unused = tuple(w for w in range(n + 1) if w not in returned)
    reordered = (
        state.reshape((2,) * (n + 1)).permute(*returned, *unused).reshape(1 << n, -1)
    )
    torch.testing.assert_close(reordered[:, 0], actual, atol=3e-14, rtol=0)
    assert torch.count_nonzero(reordered[:, 1:]) == 0


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_original_zero_state_nodes_linear_storage_and_autograd(
    dtype: torch.dtype,
) -> None:
    from qualtran.bloqs.basic_gates import ZeroState

    sig = Signature(
        [Register("word", RegisterType("uint", 3), (2,), RegisterSide.RIGHT)]
    )
    labels = {"word": ((2, 7, 11), (15, 20, 29))}
    like = torch.empty((), dtype=dtype)
    nodes = register_zero_state_nodes(sig, labels, like)
    original = ZeroState().my_tensors({}, {"q": "output"})
    assert len(nodes) == 6 and sum(node.tensor.numel() for node in nodes) == 12
    for node in nodes:
        assert node.tensor.dtype == dtype and node.tensor.device == like.device
        np.testing.assert_array_equal(node.tensor.numpy(), original[0].data)
    value = torch.tensor(
        [1 + 2j, 3 - 4j] if dtype.is_complex else [1, 3],
        dtype=dtype,
        requires_grad=True,
    )
    covector = TensorNetworkNode(value, (2,))
    result = contract_register_tensors((nodes[0], covector), Signature([]), {}, {})
    (gradient,) = torch.autograd.grad(result.abs().square(), (value,))
    torch.testing.assert_close(
        gradient, torch.stack((2 * value.detach()[0], torch.zeros((), dtype=dtype)))
    )


def test_empty_zero_width_alias_and_direction_failures() -> None:
    from flagquantum.core.ir.program.register_types import classical_bit_type

    like = torch.empty(())
    for width, shape in ((0, ()), (3, (0,))):
        sig = Signature(
            [Register("word", RegisterType("uint", width), shape, RegisterSide.RIGHT)]
        )
        assert register_zero_state_nodes(sig, {}, like) == ()
    for side in (RegisterSide.LEFT, RegisterSide.THRU):
        with pytest.raises(ValueError, match="pure-output"):
            register_zero_state_nodes(
                Signature([Register("q", RegisterType("uint", 2), side=side)]), {}, like
            )
    with pytest.raises(ValueError, match="pure-output"):
        register_zero_state_nodes(
            Signature([Register("c", classical_bit_type(), side=RegisterSide.RIGHT)]),
            {},
            like,
        )
    sig = Signature([Register("q", RegisterType("uint", 2), side=RegisterSide.RIGHT)])
    with pytest.raises(ValueError, match="distinct"):
        register_zero_state_nodes(sig, {"q": ((1, 1),)}, like)
    with pytest.raises(KeyError):
        register_zero_state_nodes(sig, {}, like)
    with pytest.raises(ValueError, match="counts"):
        register_zero_state_nodes(sig, {"q": ((1,),)}, like)


def test_replacement_supplier_keeps_initialized_consumer_and_gradient() -> None:
    record, _ = phase_fixture(2, ())
    initialized = initialize_component_from_zero(record)
    selected, _ = expand_components_for_tensor(
        initialized,
        lambda name: name.startswith("__fq_zero_") or name in {"h", "s"},
        entry="from_zero",
        full_flatten=False,
    )
    theta = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)

    def alternative(
        function: FunctionRecord, inc: Any, out: Any
    ) -> tuple[TensorNetworkNode, ...]:
        nodes = supplier(function, inc, out)
        if function.name == "s":
            return tuple(
                replace(node, tensor=node.tensor * torch.exp(1j * theta))
                for node in nodes
            )
        return nodes

    baseline = contract(selected)
    changed = contract(selected, alternative)
    torch.testing.assert_close(
        changed, baseline * torch.exp(2j * theta), atol=3e-14, rtol=0
    )
    (gradient,) = torch.autograd.grad(changed.real.sum(), (theta,))
    torch.testing.assert_close(
        gradient, (2j * baseline * torch.exp(2j * theta)).real.sum()
    )


def test_initialized_measured_add_all_branches_and_roundtrip() -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.registers.component_adjoint import (
        adjoint_component_hierarchy,
    )
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_component_hierarchy_adjoint import wrapped_add
    from tests.unit.test_component_tensor_expansion import (
        test_measured_add_consumer_preserves_gates_all_history_pairs_and_work,
    )

    record, _ = wrapped_add(2)
    initialized = initialize_component_from_zero(record)
    forward = bind_register_program(
        initialized,
        (),
        entry="from_zero",
        clean=tuple(range(8)),
        allow_workspace_outputs=True,
    )
    inverse = adjoint_component_hierarchy(record, {"add": "add_adjoint"})
    backward = bind_register_program(
        inverse,
        forward.outputs,
        clean=tuple(
            w
            for w in range(8)
            if w not in {b for word in forward.outputs for b in word}
        ),
    )
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[0] = 1
    histories = tuple(product((0, 1), repeat=4))
    for first in histories:
        state = forced_branch(initial, forward.circuit.instructions, first)
        torch.testing.assert_close(state, initial / 4, atol=2e-12, rtol=0)
        for second in histories:
            result = forced_branch(state, backward.circuit.instructions, second)
            torch.testing.assert_close(result, initial / 16, atol=2e-12, rtol=0)
    circuit = DynamicCircuit(8, dtype=torch.complex128)
    circuit.append(FieldOperation(8, forward.circuit.instructions))
    circuit.append(FieldOperation(8, backward.circuit.instructions))
    execution = run_dynamic(circuit, shots=8, seed=37, strategy="trajectory")
    torch.testing.assert_close(
        execution.final_states.reshape(8, -1), initial.expand(8, -1), atol=2e-12, rtol=0
    )
    # Also retain the nonzero arbitrary-joint-input source/gate/phase check.
    test_measured_add_consumer_preserves_gates_all_history_pairs_and_work(False)
