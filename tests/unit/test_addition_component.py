"""Factorized Add graphs preserve the source gates and bounded tensor example."""

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import replace
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.arithmetic.addition import add_ir
from flagquantum.compiler.registers.arithmetic.addition_component import (
    addition_component_program,
)
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.component_tensor_expansion import (
    expand_components_for_tensor,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.classical_addition import register_add_classical_values
from flagquantum.simulation.classical_and import and_classical_values
from flagquantum.simulation.classical_controlled_bits import cnot_classical_values
from flagquantum.simulation.tensor_network.classical_action_tensor import (
    classical_action_tensor,
)
from flagquantum.simulation.tensor_network.component_dense import (
    contract_component_tensors,
)
from flagquantum.simulation.tensor_network.component_tensor_nodes import (
    component_tensor_nodes,
)
from flagquantum.simulation.tensor_network.contraction import (
    _contract_nodes_with_slicing_plan,
    _cotengra_slicing_plan,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_axis_groups import (
    register_tensor_label_groups,
)
from flagquantum.simulation.tensor_network.register_basis_vectors import (
    register_basis_vector_nodes,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_classical_vectors import tracked_flank
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
Connections = Mapping[str, tuple[tuple[int, ...], ...]]


def tensor_supplier(
    vectors: dict[str, tuple[int, bool]],
) -> Callable[
    [FunctionRecord, Connections, Connections], tuple[TensorNetworkNode, ...]
]:
    like = torch.empty((), dtype=torch.complex128)
    # Enumerate only a two-control primitive; never a whole n-bit Add matrix.
    from flagquantum.core.ir.program.registers import (
        Register,
        RegisterSide,
        RegisterType,
    )

    bit = RegisterType("bit", 1)
    and_sig = Signature(
        (Register("ctrl", bit, (2,)), Register("target", bit, side=RegisterSide.RIGHT))
    )
    compute = torch.as_tensor(
        classical_action_tensor(
            and_sig,
            lambda **values: tuple(and_classical_values(values["ctrl"]).values()),
            operation_name="and",
        ),
        dtype=like.dtype,
    ).reshape(8, 4)
    cnot_sig = Signature((Register("ctrl", bit), Register("target", bit)))
    cnot = torch.as_tensor(
        classical_action_tensor(
            cnot_sig,
            lambda **values: cnot_classical_values(values["ctrl"], values["target"]),
            operation_name="cx",
        ),
        dtype=like.dtype,
    )

    def supply(
        function: FunctionRecord, incoming: Connections, outgoing: Connections
    ) -> tuple[TensorNetworkNode, ...]:
        sig = Signature(function.registers or ())
        if function.name in vectors:
            value, state = vectors[function.name]
            return register_basis_vector_nodes(
                sig, outgoing if state else incoming, {"val": value}, like, state=state
            )
        matrix = {
            "add_cx": cnot,
            "add_and": compute,
            "add_and_adjoint": compute.mH,
        }.get(function.name)
        if matrix is not None:
            return (bind_register_tensor(matrix, sig, incoming, outgoing),)
        assert all(
            op.opcode in {"split_register", "join_register"}
            for op in function.blocks[0].operations
        )
        right, left = register_tensor_label_groups(sig, incoming, outgoing)
        return tuple(
            TensorNetworkNode(
                torch.eye(2, dtype=like.dtype), (output_label, input_label)
            )
            for output_label, input_label in zip(right, left, strict=True)
        )

    return supply


def selected_graph(
    record: ProgramRecord, vectors: dict[str, tuple[int, bool]], entry: str
) -> tuple[ProgramRecord, tuple[int, ...]]:
    selected, _ = expand_components_for_tensor(
        record, lambda name: name != "main", entry, full_flatten=False
    )
    return selected, tuple(i for i, _, _ in iter_component_calls(selected, entry))


@pytest.mark.parametrize(
    "left,right", [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3), (5, 5), (32, 32)]
)
def test_typed_graph_binds_exact_original_gate_and_feedback_stream(
    left: int, right: int
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    graph = addition_component_program(left, right)
    assert verify_program_record(graph) == ()
    a, b, carries = (
        tuple(range(left)),
        tuple(range(left, left + right)),
        tuple(range(left + right, left + 2 * right - 1)),
    )
    bound = bind_register_program(graph, (a, b), clean=carries)
    source = Add(QUInt(left), QUInt(right))
    original = reference_arithmetic_gates(
        source,
        {"a": a, "b": b},
        carries,
        dict(zip(carries, range(len(carries)), strict=True)),
    )
    slot_names = {
        g.metadata["classical_bit"]: carries.index(g.wires[0])
        for g in bound.circuit.instructions
        if g.name == "measure"
    }
    normalized = []
    for gate in bound.circuit.instructions:
        metadata = dict(gate.metadata)
        if gate.name == "measure":
            metadata["classical_bit"] = slot_names[metadata["classical_bit"]]
        if "conditions" in metadata:
            metadata["conditions"] = tuple(
                (slot_names[slot], value) for slot, value in metadata["conditions"]
            )
        normalized.append(replace(gate, metadata=metadata))
    assert_source(tuple(normalized), original)
    assert_source(tuple(normalized), add_ir(a, b, carries, tuple(range(len(carries)))))
    assert bound.outputs == (a, b)
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts["t"] + counts["tdg"] == 4 * (right - 1)
    assert counts["measure"] == right - 1
    model = get_cost_value(source, QECGatesCost())
    assert model.and_bloq == right - 1
    assert get_cost_value(source, QubitCount()) == left + 2 * right - 1


@pytest.mark.parametrize("left,right", [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)])
def test_full_factorized_matrix_complex_entanglement_all_measurement_branches(
    left: int, right: int
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add

    graph = addition_component_program(left, right)
    selected, order = selected_graph(graph, {}, "main")
    actual = contract_component_tensors(
        selected, order, tensor_supplier({}), torch.empty((), dtype=torch.complex128)
    )
    expected = torch.as_tensor(
        Add(QUInt(left), QUInt(right)).tensor_contract(), dtype=torch.complex128
    )
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=0)
    n = left + right
    work = right - 1
    data = torch.arange(1, 2 ** (n + 1) + 1, dtype=torch.float64)
    state = torch.complex(data, 2 * data - 7)
    state /= state.norm()
    indices = ((torch.arange(len(data)) >> 1) << (work + 1)) | (
        torch.arange(len(data)) & 1
    )
    initial = torch.zeros(1 << (n + work + 1), dtype=torch.complex128)
    initial[indices] = state
    expected_state = torch.zeros_like(initial)
    expected_state[indices] = (actual @ state.reshape(-1, 2)).flatten()
    bound = bind_register_program(
        graph,
        (tuple(range(left)), tuple(range(left, n))),
        clean=tuple(range(n, n + work)),
    )
    for history in product((0, 1), repeat=work):
        torch.testing.assert_close(
            forced_branch(initial, bound.circuit.instructions, history),
            expected_state / (2 ** (work / 2)),
            atol=2e-12,
            rtol=0,
        )


@pytest.mark.parametrize(
    "width,a,b,valid",
    [
        (5, 2, 3, True),
        (5, 2, 3, False),
        (32, 5555, 6666, True),
        (32, 5555, 6666, False),
    ],
)
def test_original_notebook_closed_inputs_use_only_factorized_native_tensors(
    monkeypatch: pytest.MonkeyPatch, width: int, a: int, b: int, valid: bool
) -> None:
    graph = addition_component_program(width)
    outputs = {"a": a, "b": (a + b + int(not valid)) % (1 << width)}
    closed, vectors = tracked_flank(monkeypatch, graph, {"a": a, "b": b}, outputs)
    selected, order = selected_graph(closed, vectors, "classical_vectors")
    supply = tensor_supplier(vectors)
    like = torch.empty((), dtype=torch.complex128)
    nodes, incoming, outgoing = component_tensor_nodes(
        selected, order, supply, like, "classical_vectors"
    )
    assert incoming == outgoing == {}
    assert max(node.tensor.numel() for node in nodes) <= 32
    plan = _cotengra_slicing_plan(
        nodes, (), target_peak_elements=1024, methods=("greedy",), max_repeats=1
    )
    assert plan.n_slices == 1
    result = _contract_nodes_with_slicing_plan(nodes, (), plan)
    torch.testing.assert_close(
        result, torch.tensor(int(valid), dtype=like.dtype), atol=2e-12, rtol=0
    )
    assert plan.peak_size <= 1024
    # The input/output prediction is independent of tensor contraction.
    assert register_add_classical_values(a, b, width) == (a, (a + b) % (1 << width))


@pytest.mark.parametrize(
    "left,right", [(0, None), (-1, None), (True, 2), (2, 1), (1, 2.0)]
)
def test_invalid_native_widths_fail_before_graph_construction(
    left: Any, right: Any
) -> None:
    with pytest.raises(ValueError, match="widths"):
        addition_component_program(left, right)


@pytest.mark.parametrize("width", [2, 3])
def test_original_append_path_executes_the_factorized_closed_graph(
    monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    graph = addition_component_program(width)
    closed, _ = tracked_flank(
        monkeypatch,
        graph,
        {"a": 2, "b": 3},
        evaluate=lambda values: dict(
            zip(
                ("a", "b"),
                register_add_classical_values(values["a"], values["b"], width),
                strict=True,
            )
        ),
    )
    total = 3 * width - 1
    bound = bind_register_program(
        closed, (), entry="classical_vectors", clean=tuple(range(total))
    )
    assert bound.outputs == ()
    circuit = DynamicCircuit(total, dtype=torch.complex128)
    circuit.append(FieldOperation(total, bound.circuit.instructions))
    actual = run_dynamic(circuit, shots=4, seed=37, strategy="trajectory")
    zero = torch.zeros(1 << total, dtype=torch.complex128)
    zero[0] = 1
    torch.testing.assert_close(
        actual.final_states.reshape(4, -1), zero.expand(4, -1), atol=2e-12, rtol=0
    )
