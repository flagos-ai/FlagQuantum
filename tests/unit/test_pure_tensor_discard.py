"""Pure trace refusal follows source sentinels without changing mixed numerics."""

from collections import Counter
from collections.abc import Callable, Iterable
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    ProgramRecord,
    ValueRef,
)
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network import component_dense
from flagquantum.simulation.tensor_network.component_dense import (
    contract_component_superoperator,
    contract_component_tensors,
)
from flagquantum.simulation.tensor_network.component_tensor_nodes import (
    component_tensor_nodes,
)
from flagquantum.simulation.tensor_network.measurement_copy import (
    z_measurement_copy_node,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    contract_register_tensors,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_tensor_nodes import order
from tests.unit.test_component_trace_only import graph, source_graph, trace_supplier
from tests.unit.test_tensor_root_entry import Ports

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_dephasing(events: list[str]) -> Any:
    """Supply unchanged MeasureZ tensors on a quantum output instead of a CBit."""
    from qualtran import Bloq
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import MeasureZ

    def tensors(self: object, incoming: Any, outgoing: Any) -> Iterable[Any]:
        node, trace = MeasureZ().my_tensors({"q": incoming["q"]}, {"c": outgoing["q"]})
        events.append("node")
        yield node
        events.append("trace")
        yield trace
        events.append("tail")

    factory: Any = type(
        "QuantumDephasingLeaf",
        (Bloq,),
        {
            "signature": property(lambda self: SourceSignature.build(q=1)),
            "my_tensors": tensors,
        },
    )
    return factory()


def dephasing_graph() -> ProgramRecord:
    # Only a numerical leaf descriptor: its body is not a physical dephasing recipe.
    q = ValueRef("entry", 0, "register")
    leaf = FunctionRecord(
        "dephase",
        (BlockRecord("entry", (q,), return_values=(q,)),),
        registers=(Register("q", quantum_bit_type()),),
    )
    builder = ComponentProgramBuilder(Signature.build(q=1, spectator=1), (leaf,))
    return builder.finalize(
        {
            "q": builder.call_dict("dephase", {"q": builder.inputs["q"]})["q"],
            "spectator": builder.inputs["spectator"],
        }
    )


def source_dephasing_graph(events: list[str]) -> Any:
    from qualtran import BloqBuilder
    from qualtran import Signature as SourceSignature

    bb, values = BloqBuilder.from_signature(SourceSignature.build(q=1, spectator=1))
    return bb.finalize(
        q=bb.add(source_dephasing(events), q=values["q"]), spectator=values["spectator"]
    )


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,)])
def test_original_direct_quantum_discard_rejected_at_first_request(
    dense: bool, shape: tuple[int, ...]
) -> None:
    from qualtran.simulation.tensor._dense import bloq_to_dense
    from qualtran.simulation.tensor._quimb import cbloq_to_quimb

    record = graph(False, shape, True)
    events: list[str] = []
    like = torch.empty((), dtype=torch.complex128)

    def pure(
        function: FunctionRecord, inc: Ports, out: Ports
    ) -> Iterable[TensorNetworkNode | int]:
        events.append(function.name)
        for word in inc["q"]:
            yield from word
            events.append("after_trace")

    original = source_graph(False, shape, True, False)
    if int(np.prod(shape)):
        with pytest.raises(ValueError, match="tried to discard information") as failure:
            if dense:
                contract_component_tensors(record, order(record), pure, like)
            else:
                component_tensor_nodes(record, order(record), pure, like)
        assert "drop" in str(failure.value)
        assert "component_superoperator_nodes" in str(failure.value)
        assert "contract_component_superoperator" in str(failure.value)
        assert events == ["drop"]
        with pytest.raises(
            ValueError, match="tried to discard information"
        ) as source_failure:
            if dense:
                bloq_to_dense(original)
            else:
                cbloq_to_quimb(original)
        assert "superoperator=True" in str(source_failure.value)
    else:
        if dense:
            actual = contract_component_tensors(record, order(record), pure, like)
        else:
            nodes, inc, out = component_tensor_nodes(record, order(record), pure, like)
            actual = contract_register_tensors(
                nodes, Signature(record.functions[0].registers or ()), inc, out
            )
        np.testing.assert_array_equal(actual.numpy(), bloq_to_dense(original))
        assert not events


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_tensor_then_trace_refusal_stops_generator_before_tail_or_contraction(
    dense: bool, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran.simulation.tensor._dense import bloq_to_dense
    from qualtran.simulation.tensor._quimb import cbloq_to_quimb

    events: list[str] = []
    source_events: list[str] = []
    record = dephasing_graph()
    like = torch.empty((), dtype=dtype)

    def pure(
        function: FunctionRecord, inc: Ports, out: Ports
    ) -> Iterable[TensorNetworkNode | int]:
        # This finite refused graph leases edge 17 explicitly; no shared allocator is inferred.
        events.append("node")
        yield z_measurement_copy_node(inc["q"][0][0], out["q"][0][0], 17, like)
        events.append("trace")
        yield 17
        events.append("tail")

    def forbidden_contraction(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("trace refusal must precede contraction")

    monkeypatch.setattr(
        component_dense, "contract_register_tensors", forbidden_contraction
    )
    with pytest.raises(ValueError, match="dephase tried to discard information"):
        if dense:
            contract_component_tensors(record, order(record), pure, like)
        else:
            component_tensor_nodes(record, order(record), pure, like)
    assert events == ["node", "trace"]
    original = source_dephasing_graph(source_events)
    with pytest.raises(ValueError, match="tried to discard information"):
        if dense:
            bloq_to_dense(original, full_flatten=False)
        else:
            cbloq_to_quimb(original)
    assert source_events == events


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_same_trace_protocol_remains_valid_for_mixed_channel_and_relative_gradient(
    dtype: torch.dtype,
) -> None:
    from qualtran.simulation.tensor._dense import bloq_to_dense

    events: list[str] = []
    source_events: list[str] = []
    record = dephasing_graph()
    like = torch.empty((), dtype=dtype)

    def mixed(
        function: FunctionRecord, inc: Ports, out: Ports, fresh: Callable[[], int]
    ) -> Iterable[TensorNetworkNode | int]:
        environment = fresh()
        events.append("node")
        yield z_measurement_copy_node(inc["q"][0][0], out["q"][0][0], environment, like)
        events.append("trace")
        yield environment
        events.append("tail")

    actual = contract_component_superoperator(record, order(record), mixed, like)
    original = bloq_to_dense(
        source_dephasing_graph(source_events), full_flatten=False, superoperator=True
    )
    assert events == source_events == ["node", "trace", "tail"]
    np.testing.assert_array_equal(actual.numpy(), original)
    theta = torch.tensor(0.27, dtype=torch.float64, requires_grad=True)
    state = torch.arange(1, 5) * torch.exp(1j * theta * torch.arange(4))
    state = state / state.norm()
    density = torch.outer(state, state.conj())
    observed = torch.einsum("abij,ij->ab", actual.to(torch.complex128), density)
    expected = torch.zeros_like(density).reshape(2, 2, 2, 2)
    for bit in (0, 1):
        expected[bit, :, bit, :] = density.reshape(2, 2, 2, 2)[bit, :, bit, :]
    torch.testing.assert_close(observed, expected.reshape(4, 4), rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(observed.real.sum(), theta, retain_graph=True)[0],
        torch.autograd.grad(expected.real.sum(), theta)[0],
        rtol=0,
        atol=2e-15,
    )


@pytest.mark.parametrize("width", [1, 2, 3])
def test_pure_refusal_preserves_measure_and_forget_append_every_branch_and_work(
    width: int,
) -> None:
    from qualtran.simulation.tensor._dense import bloq_to_dense

    record = graph(False, (width,), True)
    inputs = tuple((i,) for i in range(width + 1))
    bound = bind_register_program(record, inputs, clean=(width + 1,))

    def pure(
        function: FunctionRecord, inc: Ports, out: Ports
    ) -> Iterable[TensorNetworkNode | int]:
        yield from (label for word in inc["q"] for label in word)

    with pytest.raises(ValueError, match="tried to discard information"):
        contract_component_tensors(record, order(record), pure, torch.empty(()))
    assert bind_register_program(record, inputs, clean=(width + 1,)) == bound
    gates = (
        Circuit(width + 2)
        .append(FieldOperation(width + 2, bound.circuit.instructions))
        .to_ir()
        .instructions
    )
    assert gates == bound.circuit.instructions
    assert Counter(g.name for g in gates) == {"measure": width}
    for i, gate in enumerate(gates):
        assert gate.wires == (i,) and gate.metadata["classical_bit"] == i
    state = torch.arange(1, (1 << (width + 1)) + 1, dtype=torch.float64) * torch.exp(
        1j * 0.31 * torch.arange(1 << (width + 1), dtype=torch.float64)
    )
    state = state / state.norm()
    initial = torch.zeros(1 << (width + 2), dtype=torch.complex128)
    initial[::2] = state
    total = torch.zeros((2, 2), dtype=torch.complex128)
    for bits in product((0, 1), repeat=width):
        branch = forced_branch(initial, gates, bits).reshape(1 << width, 2, 2)
        index = sum(bit << (width - i - 1) for i, bit in enumerate(bits))
        expected = torch.zeros_like(branch)
        expected[index, :, 0] = state.reshape(1 << width, 2)[index]
        torch.testing.assert_close(branch, expected, rtol=0, atol=0)
        assert torch.count_nonzero(branch[..., 1]) == 0
        total += torch.outer(branch[index, :, 0], branch[index, :, 0].conj())
    channel = contract_component_superoperator(
        record, order(record), trace_supplier, torch.empty((), dtype=torch.complex128)
    )
    density = torch.outer(state, state.conj())
    torch.testing.assert_close(
        total, torch.einsum("abij,ij->ab", channel, density), rtol=0, atol=0
    )
    # The source's direct-input superoperator failure stays separate from its pure refusal.
    with pytest.raises(KeyError):
        bloq_to_dense(source_graph(False, (width,), True, False), superoperator=True)
    np.testing.assert_array_equal(
        channel.numpy(),
        bloq_to_dense(source_graph(False, (width,), True, True), superoperator=True),
    )
