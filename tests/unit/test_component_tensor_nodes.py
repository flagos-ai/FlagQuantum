"""Selected SSA wiring matches original composite tensor networks."""

from collections import Counter
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
)
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    quantum_bit_type,
)
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
from flagquantum.simulation.tensor_network.register_tensor_indices import (
    register_tensor_outer_indices,
)
from tests.unit.test_component_expansion import _hierarchy

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def fixture(width: int, shape: tuple[int, ...]) -> tuple[ProgramRecord, Any]:
    from qualtran import BloqBuilder, QAny
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import Hadamard

    base, _ = _hierarchy(1, 1)
    leaf = base.functions[1]
    signature = Signature(
        [
            Register("q", quantum_bit_type(), (2,)),
            Register("idle", RegisterType("uint", width), shape),
        ]
    )
    builder = ComponentProgramBuilder(signature, (leaf,))
    words = np.asarray(builder.inputs["q"]).copy()
    for index in (0, 1, 0):
        words[index] = builder.call_dict("h", {"q": words[index]})["q"]
    record = builder.finalize({"q": words, "idle": builder.inputs["idle"]})
    source_signature = SourceSignature(
        [
            SourceRegister("q", QAny(1), shape=(2,)),
            SourceRegister("idle", QAny(width), shape=shape),
        ]
    )
    bb, values = BloqBuilder.from_signature(source_signature)
    for index in (0, 1, 0):
        values["q"][index] = bb.add(Hadamard(), q=values["q"][index])
    return record, bb.finalize(**values)


def order(record: ProgramRecord) -> tuple[int, ...]:
    return tuple(position for position, _, _ in iter_component_calls(record))


def h_supplier(
    function: FunctionRecord, incoming: Any, outgoing: Any
) -> tuple[TensorNetworkNode, ...]:
    matrix = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / np.sqrt(2)
    return (
        bind_register_tensor(
            matrix,
            Signature(function.registers or ()),
            incoming,
            outgoing,
            name=function.name,
        ),
    )


@pytest.mark.parametrize("width", [0, 1, 2])
@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,)])
def test_array_edges_partial_direct_ports_and_original_dense(
    width: int, shape: tuple[int, ...]
) -> None:
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.simulation.tensor._dense import quimb_to_dense
    from qualtran.simulation.tensor._quimb import cbloq_to_quimb

    import flagquantum as fq
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.registers.register_binding import bind_register_program

    record, source = fixture(width, shape)
    nodes, incoming, outgoing = component_tensor_nodes(
        record, order(record), h_supplier, torch.empty((), dtype=torch.complex128)
    )
    sig = Signature(record.functions[0].registers or ())
    actual = contract_register_tensors(nodes, sig, incoming, outgoing)
    original = cbloq_to_quimb(source)
    np.testing.assert_allclose(
        actual.numpy(), quimb_to_dense(original, source.signature), atol=2e-14, rtol=0
    )
    assert len(nodes) == len(original.tensors) == 3 + width * int(np.prod(shape))
    # Match each callee tensor in source visitation order, not just its product.
    for native, expected in zip(nodes[:3], original.tensors[:3], strict=True):
        np.testing.assert_array_equal(native.tensor.numpy(), expected.data)
    labels = Counter(label for node in nodes for label in node.labels)
    externals = [
        label
        for ports in (incoming, outgoing)
        for words in ports.values()
        for word in words
        for label in word
    ]
    assert len(externals) == len(set(externals))
    assert all(
        count == (1 if label in externals else 2) for label, count in labels.items()
    )
    assert set(externals) <= labels.keys()
    external_names = register_tensor_outer_indices(sig, incoming, outgoing)

    def canonical(rows: Any, external: Any) -> Any:
        internal: dict[Any, int] = {}
        result = []
        for row in rows:
            result.append(
                tuple(
                    (
                        external[label]
                        if label in external
                        else ("internal", internal.setdefault(label, len(internal)))
                    )
                    for label in row
                )
            )
        return result

    source_rows = [
        tensor.inds if i < 3 else tensor.inds[::-1]
        for i, tensor in enumerate(original.tensors)
    ]
    source_external = {
        label: label
        for row in source_rows
        for label in row
        if isinstance(label, tuple) and len(label) == 4 and label[-1] in {"l", "r"}
    }
    assert canonical([node.labels for node in nodes], external_names) == canonical(
        source_rows, source_external
    )
    physical_bindings = tuple(
        tuple(range(start, start + size))
        for start, size in [
            (0, 1),
            (1, 1),
            *((2 + i * width, width) for i in range(int(np.prod(shape)))),
        ]
    )
    n = 2 + width * int(np.prod(shape))
    bound = bind_register_program(record, physical_bindings, clean=(n,))
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts == {"h": 3}
    assert get_cost_value(source, QECGatesCost()).clifford == 3
    initial = torch.complex(
        torch.arange(1, 2**n + 1, dtype=torch.float64),
        torch.arange(2**n, 0, -1, dtype=torch.float64),
    )
    initial /= initial.norm()
    target = actual @ initial
    with_work = torch.zeros(2 ** (n + 1), dtype=torch.complex128)
    with_work[::2] = initial
    state = (
        fq.Circuit(n + 1, inputs=with_work, dtype=torch.complex128)
        .append(FieldOperation(n + 1, bound.circuit.instructions))
        .state()
        .flatten()
    )
    torch.testing.assert_close(state[::2], target, atol=2e-14, rtol=0)
    assert torch.count_nonzero(state[1::2]) == 0


@pytest.mark.parametrize("full", [False, True])
def test_expansion_and_supplier_replacement_without_consumer_changes(
    full: bool,
) -> None:
    from qualtran.simulation.tensor._dense import bloq_to_dense

    from flagquantum.compiler.registers.component_tensor_expansion import (
        expand_components_for_tensor,
    )
    from tests.unit.test_component_tensor_expansion import fixture as nested_fixture

    record, source = nested_fixture(3, 2)
    selected, _ = expand_components_for_tensor(
        record, lambda name: name == "custom", full_flatten=full
    )
    seen = []

    def supply(
        function: FunctionRecord, incoming: Any, outgoing: Any
    ) -> tuple[TensorNetworkNode, ...]:
        seen.append(function.name)
        if function.name == "custom":
            return (
                bind_register_tensor(
                    torch.eye(2, dtype=torch.complex128),
                    Signature(function.registers or ()),
                    incoming,
                    outgoing,
                ),
            )
        return h_supplier(function, incoming, outgoing)

    nodes, inc, out = component_tensor_nodes(
        selected, order(selected), supply, torch.empty((), dtype=torch.complex128)
    )
    actual = contract_register_tensors(
        nodes, Signature(selected.functions[0].registers or ()), inc, out
    )
    np.testing.assert_allclose(
        actual.numpy(), bloq_to_dense(source, full_flatten=full), atol=2e-14, rtol=0
    )
    assert set(seen) == ({"h"} if full else {"custom"})

    # A second implementation emits two binary tensors for the same leaf.
    def alternative(
        function: FunctionRecord, incoming: Any, outgoing: Any
    ) -> tuple[TensorNetworkNode, ...]:
        first = h_supplier(function, incoming, outgoing)[0]
        return (
            replace(first, tensor=first.tensor / 2),
            replace(
                first, tensor=torch.ones((), dtype=torch.complex128) * 2, labels=()
            ),
        )

    a, inc, out = component_tensor_nodes(
        selected, order(selected), alternative, torch.empty((), dtype=torch.complex128)
    )
    b, _, _ = component_tensor_nodes(
        selected, order(selected), h_supplier, torch.empty((), dtype=torch.complex128)
    )
    sig = Signature(selected.functions[0].registers or ())
    torch.testing.assert_close(
        contract_register_tensors(a, sig, inc, out),
        contract_register_tensors(b, sig, inc, out),
        atol=2e-14,
        rtol=0,
    )


def test_direct_renamed_directional_ports_and_empty_graph() -> None:
    signature = Signature(
        [
            Register("before", RegisterType("uint", 2), (2,), RegisterSide.LEFT),
            Register("after", RegisterType("uint", 2), (2,), RegisterSide.RIGHT),
        ]
    )
    builder = ComponentProgramBuilder(signature)
    record = builder.finalize({"after": np.asarray(builder.inputs["before"])[::-1]})

    def fail(*args: Any) -> Any:
        raise AssertionError("no calls")

    nodes, inc, out = component_tensor_nodes(
        record, (), fail, torch.empty((), dtype=torch.complex64)
    )
    tensor = contract_register_tensors(nodes, signature, inc, out)
    expected = torch.zeros(16, 16, dtype=torch.complex64)
    for value in range(16):
        expected[((value & 3) << 2) | (value >> 2), value] = 1
    torch.testing.assert_close(tensor, expected)
    assert all(node.tensor.dtype == torch.complex64 for node in nodes)
    empty = ComponentProgramBuilder(Signature([])).finalize({})
    assert component_tensor_nodes(empty, (), fail, torch.empty(())) == ((), {}, {})


def test_callback_autograd_and_error_propagation() -> None:
    record, _ = fixture(1, ())
    weight = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)

    def supply(
        function: FunctionRecord, inc: Any, out: Any
    ) -> tuple[TensorNetworkNode, ...]:
        return tuple(
            replace(node, tensor=node.tensor * weight)
            for node in h_supplier(function, inc, out)
        )

    nodes, inc, out = component_tensor_nodes(record, order(record), supply, weight)
    result = contract_register_tensors(
        nodes, Signature(record.functions[0].registers or ()), inc, out
    )
    (gradient,) = torch.autograd.grad(result.abs().square().sum(), (weight,))
    torch.testing.assert_close(gradient, 48 * weight.detach() ** 5)

    def fail(*args: Any) -> Any:
        raise RuntimeError("supplier unavailable")

    with pytest.raises(RuntimeError, match="supplier unavailable"):
        component_tensor_nodes(record, order(record), fail, weight)


@pytest.mark.parametrize("selected", [(), (0, 0, 1), (2, 1, 0), (0, 1, 3)])
def test_invalid_order_rejected_before_callback(selected: tuple[int, ...]) -> None:
    record, _ = fixture(1, ())

    def fail(*args: Any) -> Any:
        raise AssertionError("callback must not run")

    with pytest.raises(ValueError, match="order"):
        component_tensor_nodes(record, selected, fail, torch.empty(()))


def test_entry_and_profile_rejections() -> None:
    record, _ = fixture(1, ())
    with pytest.raises(ValueError, match="unknown"):
        component_tensor_nodes(
            record, order(record), h_supplier, torch.empty(()), "missing"
        )
    with pytest.raises(ValueError, match="typed calls"):
        component_tensor_nodes(
            record,
            tuple(range(len(record.functions[1].blocks[0].operations))),
            h_supplier,
            torch.empty(()),
            "h",
        )
    invalid = replace(record, functions=record.functions + (record.functions[0],))
    with pytest.raises(ValueError, match="invalid component"):
        component_tensor_nodes(invalid, (), h_supplier, torch.empty(()))
    builder = ComponentProgramBuilder(Signature([Register("c", classical_bit_type())]))
    classical = builder.finalize(builder.inputs)
    with pytest.raises(ValueError, match="classical ports"):
        component_tensor_nodes(classical, (), h_supplier, torch.empty(()))


def test_measured_add_graph_tensor_and_all_history_pairs() -> None:
    from qualtran.simulation.tensor._dense import bloq_to_dense

    from flagquantum.compiler.registers.component_tensor_expansion import (
        expand_components_for_tensor,
    )
    from tests.unit.test_component_hierarchy_adjoint import wrapped_add
    from tests.unit.test_component_tensor_expansion import (
        test_measured_add_consumer_preserves_gates_all_history_pairs_and_work,
    )

    record, source = wrapped_add(2)
    selected, _ = expand_components_for_tensor(
        record, lambda name: name == "add", full_flatten=False
    )

    def supply(
        function: FunctionRecord, inc: Any, out: Any
    ) -> tuple[TensorNetworkNode, ...]:
        assert function.name == "add"
        sig = Signature(function.registers or ())
        matrix = torch.zeros(32, 32, dtype=torch.complex128)
        for a in range(4):
            for b in range(8):
                matrix[a * 8 + ((a + b) % 8), a * 8 + b] = 1
        return (bind_register_tensor(matrix, sig, inc, out),)

    nodes, inc, out = component_tensor_nodes(
        selected, order(selected), supply, torch.empty((), dtype=torch.complex128)
    )
    actual = contract_register_tensors(
        nodes, Signature(selected.functions[0].registers or ()), inc, out
    )
    np.testing.assert_allclose(
        actual.numpy(), bloq_to_dense(source), atol=2e-14, rtol=0
    )
    # Same selected graph is lowered through the established append consumer:
    # 16 forward branches, 256 inverse pairs and clean workspace are checked.
    test_measured_add_consumer_preserves_gates_all_history_pairs_and_work(False)


def test_fanin_and_disjoint_call_order_match_original_connections() -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import CNOT, Hadamard
    from qualtran.simulation.tensor._dense import quimb_to_dense
    from qualtran.simulation.tensor._quimb import cbloq_to_quimb

    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.core.ir.program import BlockRecord, Operation, ValueRef

    values = tuple(ValueRef("entry", i, "register") for i in range(4))
    bits = tuple(ValueRef("entry", i, "qubit") for i in range(4, 8))
    ports = (
        Register("ctrl", quantum_bit_type()),
        Register("target", quantum_bit_type()),
    )
    cx = FunctionRecord(
        "cx",
        (
            BlockRecord(
                "entry",
                values[:2],
                operations=(
                    Operation("split_register", (values[0],), (bits[0],)),
                    Operation("split_register", (values[1],), (bits[1],)),
                    Operation("cx", bits[:2], bits[2:]),
                    Operation(
                        "join_register",
                        (bits[2],),
                        (values[2],),
                        register_type=quantum_bit_type(),
                    ),
                    Operation(
                        "join_register",
                        (bits[3],),
                        (values[3],),
                        register_type=quantum_bit_type(),
                    ),
                ),
                return_values=values[2:],
            ),
        ),
        registers=ports,
    )
    base, _ = _hierarchy(1, 1)
    builder = ComponentProgramBuilder(Signature(ports), (cx, base.functions[1]))
    control = builder.call_dict("h", {"q": builder.inputs["ctrl"]})["q"]
    target = builder.call_dict("h", {"q": builder.inputs["target"]})["q"]
    returned = builder.call_dict("cx", {"ctrl": control, "target": target})
    record = builder.finalize(returned)
    bb, source_inputs = BloqBuilder.from_signature(CNOT().signature)
    source_inputs = {
        name: bb.add(Hadamard(), q=value) for name, value in source_inputs.items()
    }
    source = bb.finalize(**bb.add_d(CNOT(), **source_inputs))

    def supply(
        function: FunctionRecord, inc: Any, out: Any
    ) -> tuple[TensorNetworkNode, ...]:
        if function.name == "h":
            return h_supplier(function, inc, out)
        matrix = torch.eye(4, dtype=torch.complex128)[[0, 1, 3, 2]]
        return (bind_register_tensor(matrix, Signature(ports), inc, out),)

    expected = quimb_to_dense(cbloq_to_quimb(source), source.signature)
    for selected in (order(record), (1, 0, 2)):
        nodes, inc, out = component_tensor_nodes(
            record, selected, supply, torch.empty((), dtype=torch.complex128)
        )
        actual = contract_register_tensors(nodes, Signature(ports), inc, out)
        np.testing.assert_allclose(actual.numpy(), expected, atol=2e-14, rtol=0)
    assert Counter(
        g.name for g in bind_register_program(record, ((0,), (1,))).circuit.instructions
    ) == {"h": 2, "cx": 1}


@pytest.mark.parametrize("width", [0, 1, 2])
def test_directional_ket_bra_calls_and_scalar_network(width: int) -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.bookkeeping import Allocate, Free
    from qualtran.simulation.tensor._quimb import cbloq_to_quimb

    from flagquantum.compiler.registers.layout.component_workspace import (
        allocate_word_function,
        free_word_function,
    )

    dtype = RegisterType("uint", width)
    builder = ComponentProgramBuilder(
        Signature([]),
        (
            allocate_word_function("ket", dtype, False),
            free_word_function("bra", dtype, False),
        ),
    )
    value = builder.call_dict("ket", {})["reg"]
    builder.call_dict("bra", {"reg": value})
    record = builder.finalize({})

    def supply(
        function: FunctionRecord, inc: Any, out: Any
    ) -> tuple[TensorNetworkNode, ...]:
        ket = torch.zeros(1 << width, dtype=torch.complex128)
        ket[0] = 1
        return (
            bind_register_tensor(ket, Signature(function.registers or ()), inc, out),
        )

    nodes, inc, out = component_tensor_nodes(
        record, order(record), supply, torch.empty((), dtype=torch.complex128)
    )
    actual = contract_register_tensors(nodes, Signature([]), inc, out)
    assert actual.item() == 1
    bb = BloqBuilder()
    # Original finite pure-state Allocate/Free tensors implement zero projection.
    from qualtran import QAny

    bb.add(Free(QAny(width)), reg=bb.add(Allocate(QAny(width))))
    source = bb.finalize()
    original = cbloq_to_quimb(source)
    if width:
        assert original.contract() == actual.item()
    else:
        # Source emits no zero-bit kets/bras and cannot contract the empty TN.
        # Native scalar one comes explicitly from the two supplied scalar nodes.
        assert len(original.tensors) == 0
        with pytest.raises(ValueError, match="not enough values"):
            original.contract()
    assert len(nodes) == 2
