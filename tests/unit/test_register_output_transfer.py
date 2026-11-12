"""Transfer real allocated output wires without closing their live lifetime."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_binding_fixtures import allocated_circuit

pytestmark = pytest.mark.unit


def output_program(
    circuit: CircuitIR, data_width: int, reverse: bool = False, dirty: bool = False
) -> ProgramRecord:
    module = allocated_circuit(circuit, data_width)
    main, leaf = module.functions
    block = main.blocks[0]
    work_join = next(op for op in block.operations if op.opcode == "join_register")
    operations = []
    for operation in block.operations:
        if operation.opcode == "free_register":
            continue
        if operation.opcode == "allocate_register":
            operation = replace(operation, workspace_dirty=dirty)
        elif reverse and operation is work_join:
            operation = replace(operation, operands=operation.operands[::-1])
        operations.append(operation)
    assert block.return_values is not None
    assert main.registers is not None
    main = replace(
        main,
        blocks=(
            replace(
                block,
                operations=tuple(operations),
                return_values=(*block.return_values, *work_join.results),
            ),
        ),
        registers=(
            *main.registers,
            Register(
                "out",
                RegisterType("opaque", circuit.n_wires - data_width),
                side=RegisterSide.RIGHT,
            ),
        ),
    )
    return replace(module, functions=(main, leaf))


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [0, 1, 3])
@pytest.mark.parametrize("dirty", [False, True])
def test_allocated_right_port_matches_original_source_contract(
    width: int, dirty: bool
) -> None:
    from tests.unit.test_register_output_transfer_contract import (
        test_source_allocated_output_retains_manager_ownership_and_wire_order,
    )

    test_source_allocated_output_retains_manager_ownership_and_wire_order(width, dirty)
    builder = ComponentProgramBuilder(
        Signature(
            [
                Register("data", RegisterType("opaque", 1)),
                Register("out", RegisterType("opaque", width), side=RegisterSide.RIGHT),
            ]
        )
    )
    out = builder.allocate(width, dirty=dirty)
    out = builder.join(builder.split(out)[::-1])
    module = builder.finalize({"data": builder.inputs["data"], "out": out})
    pool = tuple(range(1, width + 1))
    bound = bind_register_program(
        module,
        ((0,),),
        clean=() if dirty else pool,
        borrowed=pool if dirty else (),
        allow_workspace_outputs=True,
    )
    assert bound.outputs == ((0,), pool[::-1])
    assert bound.circuit.instructions == ()
    assert [(event.kind, event.wires, event.dirty) for event in bound.events] == [
        ("allocate", pool, dirty)
    ]
    if width:
        with pytest.raises(ValueError, match="ownership transfer"):
            bind_register_program(
                module,
                ((0,),),
                clean=() if dirty else pool,
                borrowed=pool if dirty else (),
            )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("dirty", [False, True])
def test_returned_work_preserves_source_complex_joint_state_and_permutation(
    dirty: bool,
) -> None:
    import cirq
    from qualtran import (
        BloqBuilder,
        QAny,
        Side,
    )
    from qualtran import (
        Register as SourceRegister,
    )
    from qualtran import (
        Signature as SourceSignature,
    )
    from qualtran.bloqs.basic_gates import CNOT, Hadamard, TGate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    gates = (Instruction("h", (0,)), Instruction("cx", (0, 1)), Instruction("t", (2,)))
    module = output_program(CircuitIR(3, gates), 1, reverse=True, dirty=dirty)
    bound = bind_register_program(
        module,
        ((2,),),
        clean=() if dirty else (0, 1),
        borrowed=(0, 1) if dirty else (),
        allow_workspace_outputs=True,
    )
    assert bound.outputs == ((2,), (1, 0))
    assert [(event.kind, event.wires) for event in bound.events] == [
        ("allocate", (0, 1))
    ]
    events = []

    class Manager(cirq.SimpleQubitManager):
        def qalloc(self, n: int, dim: int = 2) -> list[cirq.Qid]:
            assert n == dim == 2
            events.append("allocate")
            return list(cirq.LineQubit.range(2))

        def qborrow(self, n: int, dim: int = 2) -> list[cirq.Qid]:
            return self.qalloc(n, dim)

        def qfree(self, qubits: Iterable[cirq.Qid]) -> None:
            events.append("free")

    source, inputs = BloqBuilder.from_signature(
        SourceSignature(
            [
                SourceRegister("data", QAny(1)),
                SourceRegister("out", QAny(2), side=Side.RIGHT),
            ]
        )
    )
    work = source.split(source.allocate(dtype=QAny(2), dirty=dirty))
    data = source.add(Hadamard(), q=inputs["data"])
    data, work[0] = source.add(CNOT(), ctrl=data, target=work[0])
    work[1] = source.add(TGate(), q=work[1])
    composite = source.finalize(data=data, out=source.join(work[::-1]))
    reference, returned = composite.to_cirq_circuit_and_quregs(
        qubit_manager=Manager(), data=np.array([cirq.LineQubit(2)])
    )
    assert events == ["allocate"]
    assert tuple(returned["out"]) == tuple(cirq.LineQubit(i) for i in (1, 0))
    expected_gates = (
        Instruction("h", (2,)),
        Instruction("cx", (2, 0)),
        Instruction("t", (1,)),
    )
    from tests.unit.test_register_subtraction import assert_source

    assert_source(bound.circuit.instructions, expected_gates)
    state = torch.complex(
        torch.arange(1, 17, dtype=torch.float64),
        torch.arange(16, dtype=torch.float64).remainder(7) - 3,
    )
    if not dirty:
        state[4:] = 0
    state /= state.norm()
    import flagquantum as fq
    from examples.algorithms.binary_field_arithmetic import FieldOperation

    actual = (
        fq.Circuit(4, inputs=state, dtype=state.dtype)
        .append(FieldOperation(4, bound.circuit.instructions))
        .state()
        .flatten()
    )
    expected = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(
            reference, initial_state=state.numpy(), qubit_order=cirq.LineQubit.range(4)
        )
        .final_state_vector
    )
    torch.testing.assert_close(actual, torch.as_tensor(expected), rtol=0, atol=2e-12)
    assert Counter(gate.name for gate in bound.circuit.instructions) == {
        "h": 1,
        "cx": 1,
        "t": 1,
    }
    model = get_cost_value(composite, QECGatesCost())
    assert model.clifford == 2 and model.total_t_count() == 1 and model.measurement == 0
    assert get_cost_value(composite, QubitCount()) == 3


def test_nested_output_transfer_keeps_pool_lease_until_outer_return() -> None:
    module = output_program(CircuitIR(2, (Instruction("cx", (0, 1)),)), 1)
    functions = tuple(
        replace(f, name="allocate_leaf") if f.name == "main" else f
        for f in module.functions
    )
    builder = ComponentProgramBuilder(
        Signature(
            [
                Register("data", RegisterType("opaque", 1)),
                Register("out", RegisterType("opaque", 1), side=RegisterSide.RIGHT),
            ]
        ),
        functions,
    )
    data, out = builder.call_tuple("allocate_leaf", {"data": builder.inputs["data"]})
    extra = builder.allocate()
    builder.free(extra)
    record = builder.finalize({"data": data, "out": out})
    with pytest.raises(ValueError, match="Insufficient clean"):
        bind_register_program(record, ((0,),), clean=(1,), allow_workspace_outputs=True)
    bound = bind_register_program(
        record, ((0,),), clean=(1, 2), allow_workspace_outputs=True
    )
    assert bound.outputs == ((0,), (1,))
    assert [(event.kind, event.wires) for event in bound.events] == [
        ("allocate", (1,)),
        ("allocate", (2,)),
        ("free", (2,)),
    ]


@pytest.mark.ftqc_reference
def test_output_transfer_keeps_all_add_measurement_branches_and_clean_work() -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add

    from flagquantum.compiler.registers.arithmetic.addition import add_ir
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates

    gates = add_ir((0, 1), (2, 3, 4), (5, 6), (0, 1))
    module = output_program(CircuitIR(7, gates), 5)
    bound = bind_register_program(
        module, ((0, 1, 2, 3, 4),), clean=(5, 6), allow_workspace_outputs=True
    )
    assert bound.outputs == ((0, 1, 2, 3, 4), (5, 6))
    assert [(event.kind, event.wires) for event in bound.events] == [
        ("allocate", (5, 6))
    ]
    original = reference_arithmetic_gates(
        Add(QUInt(2), QUInt(3)),
        {"a": (0, 1), "b": (2, 3, 4)},
        (6, 5),
        {5: 0, 6: 1},
        cirq_style=True,
    )
    state = torch.zeros(128, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for word in range(32):
        amplitude = complex(word % 5 - 2, word % 3 - 1)
        state[word << 2] = torch.tensor(amplitude, dtype=state.dtype)
        left, right = word >> 3, word & 7
        expected[((left << 3) | ((left + right) & 7)) << 2] = torch.tensor(
            amplitude, dtype=expected.dtype
        )
    expected /= state.norm()
    state /= state.norm()
    assert Counter(gate.name for gate in bound.circuit.instructions) == Counter(
        gate.name for gate in original
    )
    for history in product((0, 1), repeat=2):
        actual = forced_branch(state, bound.circuit.instructions, history)
        torch.testing.assert_close(
            actual, forced_branch(state, original, history), rtol=0, atol=2e-12
        )
        torch.testing.assert_close(actual, expected / 2, rtol=0, atol=2e-12)
        assert torch.count_nonzero(actual.reshape(-1, 4)[:, 1:]) == 0


@pytest.mark.parametrize("policy", [None, 0, 1, "yes", np.bool_(True)])
def test_transfer_policy_requires_a_builtin_boolean(policy: Any) -> None:
    # Invalid API inputs intentionally cross the runtime validation boundary.
    with pytest.raises(ValueError, match="built-in boolean"):
        bind_register_program(ProgramRecord(()), (), allow_workspace_outputs=policy)


@pytest.mark.parametrize("dirty", [False, True])
def test_joined_input_and_allocated_output_keep_each_physical_wire(dirty: bool) -> None:
    builder = ComponentProgramBuilder(
        Signature(
            [
                Register("data", RegisterType("opaque", 1), side=RegisterSide.LEFT),
                Register("out", RegisterType("opaque", 2), side=RegisterSide.RIGHT),
            ]
        )
    )
    work = builder.allocate(dirty=dirty)
    data_binding = builder.inputs["data"]
    assert isinstance(data_binding, RegisterBinding)
    out = builder.join([*builder.split(work), *builder.split(data_binding)])
    bound = bind_register_program(
        builder.finalize({"out": out}),
        ((2,),),
        clean=() if dirty else (0,),
        borrowed=(0,) if dirty else (),
        allow_workspace_outputs=True,
    )
    assert bound.outputs == ((0, 2),)
    assert bound.circuit.instructions == ()
    assert len(bound.events) == 1 and bound.events[0].kind == "allocate"


@pytest.mark.parametrize("dirty", [False, True])
def test_output_permission_keeps_unmanaged_input_release_diagnostic(
    dirty: bool,
) -> None:
    builder = ComponentProgramBuilder(
        Signature(
            [
                Register("data", RegisterType("opaque", 1), side=RegisterSide.LEFT),
            ]
        )
    )
    data_binding = builder.inputs["data"]
    assert isinstance(data_binding, RegisterBinding)
    builder.free(data_binding, dirty=dirty)
    with pytest.raises(ValueError, match="caller input wires"):
        bind_register_program(
            builder.finalize({}), ((0,),), allow_workspace_outputs=True
        )


@pytest.mark.parametrize("dirty", [False, True])
def test_partial_output_transfer_keeps_remaining_lease_and_reuses_released_bit(
    dirty: bool,
) -> None:
    builder = ComponentProgramBuilder(
        Signature(
            [
                Register("out", RegisterType("opaque", 2), side=RegisterSide.RIGHT),
            ]
        )
    )
    bits = builder.split(builder.allocate(3, dirty=dirty))
    builder.free(bits[1], dirty=dirty)
    out = builder.join([bits[2], bits[0]])
    temporary = builder.allocate(dirty=dirty)
    builder.free(temporary, dirty=dirty)
    pool = (4, 1, 3)
    bound = bind_register_program(
        builder.finalize({"out": out}),
        (),
        clean=() if dirty else pool,
        borrowed=pool if dirty else (),
        allow_workspace_outputs=True,
    )
    assert bound.outputs == ((3, 4),)
    assert bound.circuit.instructions == ()
    assert [(event.kind, event.wires) for event in bound.events] == [
        ("allocate", (4, 1, 3)),
        ("free", (1,)),
        ("allocate", (1,)),
        ("free", (1,)),
    ]
