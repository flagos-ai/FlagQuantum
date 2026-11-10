"""Tensor-preserving expansion reuses the source graph policy and native owners."""

from collections import Counter
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_tensor_expansion import (
    expand_components_for_tensor,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
    contract_register_tensors,
)
from tests.unit.test_component_expansion import _hierarchy

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_custom() -> Any:
    import quimb.tensor as qtn
    from qualtran import Bloq
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import Hadamard

    def signature(self: object) -> Any:
        return SourceSignature.build(q=1)

    def build(self: object, bb: Any, q: Any) -> dict[str, Any]:
        return {"q": bb.add(Hadamard(), q=bb.add(Hadamard(), q=q))}

    def tensors(self: object, incoming: Any, outgoing: Any) -> list[Any]:
        return [qtn.Tensor(np.eye(2), inds=((outgoing["q"], 0), (incoming["q"], 0)))]

    factory: Any = type(
        "IdentityWithDecomposition",
        (Bloq,),
        {
            "signature": property(signature),
            "build_composite_bloq": build,
            "my_tensors": tensors,
        },
    )
    return factory()


def fixture(depth: int, repeats: int) -> tuple[ProgramRecord, Any]:
    from qualtran import BloqBuilder

    base, _ = _hierarchy(1, 2)
    custom = base.functions[0]
    from dataclasses import replace

    definitions = (replace(custom, name="custom"), *base.functions[1:])
    previous = "custom"
    source = source_custom()
    for level in range(depth):
        name = "main" if level == depth - 1 else f"outer{level}"
        builder = ComponentProgramBuilder(
            Signature(custom.registers or ()), definitions, name=name
        )
        values = dict(builder.inputs)
        bb, src = BloqBuilder.from_signature(source.signature)
        for _ in range(repeats):
            values = builder.call_dict(previous, values)
            src = bb.add_d(source, **src)
        record = builder.finalize(values)
        definitions, previous, source = record.functions, name, bb.finalize(**src)
    return record, source


def native_tensor(record: ProgramRecord) -> torch.Tensor:
    from flagquantum.simulation.tensor_network.models import TensorNetworkNode

    nodes: list[TensorNetworkNode] = []
    signature = Signature.build(q=1)
    block = record.functions[0].blocks[0]
    labels = {block.arguments[0]: 101}
    for position, operation in enumerate(block.operations):
        output = 110 + 7 * position
        labels[operation.results[0]] = output
        matrix = (
            torch.eye(2, dtype=torch.complex128)
            if operation.callee == "custom"
            else torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / np.sqrt(2)
        )
        nodes.append(
            bind_register_tensor(
                matrix,
                signature,
                {"q": ((labels[operation.operands[0]],),)},
                {"q": ((output,),)},
            )
        )
    assert block.return_values is not None
    return contract_register_tensors(
        nodes, signature, {"q": ((101,),)}, {"q": ((labels[block.return_values[0]],),)}
    )


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("repeats", [1, 2])
@pytest.mark.parametrize("full", [False, True])
def test_pinned_custom_tensor_precedence_graph_labels_contraction_and_append(
    depth: int, repeats: int, full: bool
) -> None:
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.simulation.tensor._dense import bloq_to_dense
    from qualtran.simulation.tensor._flattening import flatten_for_tensor_contraction

    import flagquantum as fq
    from examples.algorithms.binary_field_arithmetic import FieldOperation

    record, source = fixture(depth, repeats)
    seen: list[str] = []

    def custom(name: str) -> bool:
        seen.append(name)
        return name in {"custom", "h"}

    expanded, labels = expand_components_for_tensor(record, custom, full_flatten=full)
    expected = flatten_for_tensor_contraction(source, full_flatten=full)
    assert verify_program_record(expanded) == ()
    operations = expanded.functions[0].blocks[0].operations
    source_instances = list(expected.iter_bloqsoqs())
    assert labels == tuple(i.i for i, _, _ in source_instances)
    assert [op.callee for op in operations] == [
        "custom" if type(i.bloq).__name__ == "IdentityWithDecomposition" else "h"
        for i, _, _ in source_instances
    ]
    assert bool(seen) is not full
    assert all(
        a is b
        for a, b in zip(record.functions[1:], expanded.functions[1:], strict=True)
    )
    actual = native_tensor(expanded)
    np.testing.assert_allclose(
        actual.numpy(), bloq_to_dense(expected, full_flatten=False), atol=2e-14, rtol=0
    )
    np.testing.assert_allclose(
        actual.numpy(), bloq_to_dense(source, full_flatten=full), atol=2e-14, rtol=0
    )
    bound = bind_register_program(expanded, ((0,),), clean=(2,))
    baseline = bind_register_program(record, ((0,),), clean=(2,))
    assert bound == baseline
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts == {"h": 2 * repeats**depth}
    assert get_cost_value(source, QECGatesCost()).clifford == counts["h"]
    initial = torch.tensor([1, 0, 1j, 0, -2j, 0, 3, 0], dtype=torch.complex128)
    initial /= initial.norm()
    state = (
        fq.Circuit(3, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(3, bound.circuit.instructions))
        .state()
        .flatten()
    )
    torch.testing.assert_close(state, initial, atol=2e-14, rtol=0)
    assert torch.count_nonzero(state[1::2]) == 0


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("bound", [0, -1, 1, 2])
def test_existing_depth_bound_and_input_immutability(full: bool, bound: int) -> None:
    record, source = fixture(3, 1)
    original = record
    with pytest.raises(ValueError, match="Max recursion depth"):
        expand_components_for_tensor(
            record,
            lambda name: name in {"custom", "h"},
            full_flatten=full,
            max_depth=bound,
        )
    with pytest.raises(ValueError, match="Max recursion depth"):
        source.flatten(
            lambda i: full
            or type(i.bloq).__name__ not in {"IdentityWithDecomposition", "Hadamard"},
            max_depth=bound,
        )
    assert record == original


def test_full_expansion_never_queries_classifier_and_partial_errors_propagate() -> None:
    record, _ = fixture(1, 1)

    def fail(name: str) -> bool:
        raise RuntimeError("tensor classification failed: " + name)

    expanded, _ = expand_components_for_tensor(record, fail)
    assert [op.callee for op in expanded.functions[0].blocks[0].operations] == [
        "h",
        "h",
    ]
    with pytest.raises(RuntimeError, match="tensor classification failed: custom"):
        expand_components_for_tensor(record, fail, full_flatten=False)


@pytest.mark.parametrize("custom", [False, True])
def test_atomic_leaf_remains_when_no_decomposition_is_available(custom: bool) -> None:
    record, source = _hierarchy(1, 1)
    selected, labels = expand_components_for_tensor(
        record, lambda name: custom, full_flatten=False
    )
    assert selected == record and labels == (0,)
    expected = source.flatten(lambda i: not custom)
    assert len(expected.bloq_instances) == 1


@pytest.mark.parametrize("full", [False, True])
def test_measured_add_consumer_preserves_gates_all_history_pairs_and_work(
    full: bool,
) -> None:
    from itertools import product

    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.addition import Add
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.registers.component_adjoint import (
        adjoint_component_hierarchy,
    )
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_component_hierarchy_adjoint import wrapped_add
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    record, source = wrapped_add(2)
    inverse = adjoint_component_hierarchy(record, {"add": "add_adjoint"})
    outputs = []
    for module in (record, inverse):
        expanded, _ = expand_components_for_tensor(
            module, lambda name: name in {"add", "add_adjoint"}, full_flatten=full
        )
        bound = bind_register_program(expanded, ((0, 1), (2, 3, 4), (5,)), clean=(6, 7))
        baseline = bind_register_program(
            module, ((0, 1), (2, 3, 4), (5,)), clean=(6, 7)
        )
        assert bound == baseline
        outputs.append(bound)
    forward, backward = outputs
    for is_inverse, bound in enumerate(outputs):
        child = Add(QUInt(2), QUInt(3))
        if is_inverse:
            child = child.adjoint()
        measurements = [g for g in bound.circuit.instructions if g.name == "measure"]
        expected = tuple(
            g
            for offset in (0, 2)
            for g in reference_arithmetic_gates(
                child,
                {"a": (0, 1), "b": (2, 3, 4)},
                (6, 7),
                {
                    g.wires[0]: g.metadata["classical_bit"]
                    for g in measurements[offset : offset + 2]
                },
            )
        )
        assert_source(bound.circuit.instructions, expected)
        model = get_cost_value(child, QECGatesCost())
        counts = Counter(g.name for g in bound.circuit.instructions)
        assert counts["t"] + counts["tdg"] == 2 * model.total_t_count() == 16
        assert counts["measure"] == 2 * model.measurement == 4
    assert get_cost_value(source, QubitCount()) == 8
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[::4] = torch.complex(
        torch.arange(1, 65, dtype=torch.float64),
        torch.arange(64, 0, -1, dtype=torch.float64),
    )
    initial /= initial.norm()
    target = torch.zeros_like(initial)
    for prefix in range(64):
        a, b, spectator = prefix >> 4, (prefix >> 1) & 7, prefix & 1
        result_index = (a << 4) | (((b + 2 * a) & 7) << 1) | spectator
        target[result_index << 2] = initial[prefix << 2]
    histories = tuple(product((0, 1), repeat=4))
    for first in histories:
        state = forced_branch(initial, forward.circuit.instructions, first)
        torch.testing.assert_close(state, target / 4, atol=2e-12, rtol=0)
        assert torch.count_nonzero(state.reshape(-1, 4)[:, 1:]) == 0
        for second in histories:
            restored = forced_branch(state, backward.circuit.instructions, second)
            torch.testing.assert_close(restored, initial / 16, atol=2e-12, rtol=0)
    circuit = DynamicCircuit(8, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(8, forward.circuit.instructions))
    result = run_dynamic(circuit, shots=4, seed=23, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(4, -1), target.expand(4, -1), atol=2e-12, rtol=0
    )
    circuit.append(FieldOperation(8, backward.circuit.instructions))
    result = run_dynamic(circuit, shots=4, seed=29, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(4, -1), initial.expand(4, -1), atol=2e-12, rtol=0
    )
