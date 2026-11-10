"""Explicit physical measurement retirement preserves branch phase and snapshots."""

from collections import Counter
from itertools import product
from typing import Any, Literal

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_program_owned_measurement import measured_function
from tests.unit.test_register_classical_discard import discard_feedback_program

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("policy", ["retire", "reset"])
@pytest.mark.parametrize("dirty", [False, True])
def test_allocated_measurement_consumes_work_owner(
    policy: Literal["retire", "reset"], dirty: bool
) -> None:
    builder = ComponentProgramBuilder(
        Signature([Register("c", classical_bit_type(), side=RegisterSide.RIGHT)]),
        [measured_function("measurement")],
    )
    work = builder.allocate(dtype=quantum_bit_type(), dirty=dirty)
    result = builder.call_dict("measurement", {"q": work})
    record = builder.finalize(result)
    if dirty:
        with pytest.raises(ValueError, match="borrowed"):
            bind_register_program(
                record,
                (),
                measured_workspace=policy,
                clean=() if dirty else (0,),
                borrowed=(0,) if dirty else (),
            )
        return
    bound = bind_register_program(
        record,
        (),
        measured_workspace=policy,
        clean=() if dirty else (0,),
        borrowed=(0,) if dirty else (),
    )
    assert bound.outputs == ((),) and bound.classical_outputs == ((0,),)
    assert [e.kind for e in bound.events] == [
        "allocate",
        "free" if policy == "reset" else "retire",
    ]
    assert all(not e.dirty for e in bound.events)
    assert Counter(g.name for g in bound.circuit.instructions) == (
        {"measure": 1, "reset": 1} if policy == "reset" else {"measure": 1}
    )


@pytest.mark.parametrize("policy", ["retire", "reset"])
def test_reset_reuses_input_wire_but_retirement_exhausts_pool(
    policy: Literal["retire", "reset"],
) -> None:
    builder = ComponentProgramBuilder(
        Signature(
            [
                Register("q", quantum_bit_type(), side=RegisterSide.LEFT),
                Register("c", classical_bit_type(), side=RegisterSide.RIGHT),
                Register("out", quantum_bit_type(), side=RegisterSide.RIGHT),
            ]
        ),
        [measured_function("measurement")],
    )
    c = builder.call_dict("measurement", {"q": builder.inputs["q"]})["c"]
    out = builder.allocate(dtype=quantum_bit_type())
    record = builder.finalize({"c": c, "out": out})
    if policy == "retire":
        with pytest.raises(ValueError, match="Insufficient"):
            bind_register_program(
                record,
                ((0,),),
                input_ownership=((0, False),),
                measured_workspace=policy,
                allow_workspace_outputs=True,
            )
        return
    bound = bind_register_program(
        record,
        ((0,),),
        input_ownership=((0, False),),
        measured_workspace=policy,
        allow_workspace_outputs=True,
    )
    assert bound.outputs == ((), (0,)) and bound.classical_outputs == ((0,), ())
    assert [(e.kind, e.wires) for e in bound.events] == [
        ("free", (0,)),
        ("allocate", (0,)),
    ]
    for outcome in (0, 1):
        state = torch.tensor([1, 1j], dtype=torch.complex128) / np.sqrt(2)
        branch = forced_branch(state, bound.circuit.instructions, (outcome,))
        torch.testing.assert_close(branch, torch.tensor([state[outcome], 0]))


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("polarity", [False, True])
@pytest.mark.parametrize("policy", ["retire", "reset"])
def test_all_branches_match_source_instrument_with_snapshot_feedback(
    count: int, polarity: bool, policy: Literal["retire", "reset"]
) -> None:
    from qualtran.bloqs.basic_gates import MeasureZ
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    record = discard_feedback_program(count, polarity)
    inputs = tuple((i,) for i in range(count + 3))
    bound = bind_register_program(
        record,
        inputs,
        clean=(count + 3,),
        input_ownership=tuple((i, False) for i in range(count)),
        measured_workspace=policy,
    )
    reference = bind_register_program(record, inputs, clean=(count + 3,))
    gates = (
        Circuit(count + 4)
        .append(FieldOperation(count + 4, bound.circuit.instructions))
        .to_ir()
        .instructions
    )
    assert gates == bound.circuit.instructions
    assert (
        tuple(g for g in gates if g.name != "reset") == reference.circuit.instructions
    )
    assert Counter(g.name for g in gates) == Counter(
        {
            "measure": count,
            "classical_xor": 1,
            "z": 1,
            "x": 1,
            **({"reset": count} if policy == "reset" else {}),
        }
    )
    model = get_cost_value(MeasureZ(), QECGatesCost())
    assert model.measurement == 1 and model.total_t_count() == 0
    copy, _ = MeasureZ().my_tensors({"q": "q"}, {"c": "c"})
    state = torch.zeros(2 ** (count + 4), dtype=torch.complex128)
    state[::2] = torch.randn(
        2 ** (count + 3),
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(108431),
    )
    state /= state.norm()
    initial = state.numpy().reshape((2,) * (count + 4))
    for history in product((0, 1), repeat=count):
        expected = initial
        # Contract the unchanged original MeasureZ copy tensors, retaining
        # every joint target/spectator amplitude and the untouched work axis.
        for wire, outcome in enumerate(history):
            expected = np.moveaxis(
                np.tensordot(copy.data[:, :, outcome].T, expected, axes=([1], [wire])),
                0,
                wire,
            )
        if (sum(history) % 2) == polarity:
            phase = np.array([1, -1]).reshape((1,) * count + (2, 1, 1, 1))
            expected = np.flip(expected * phase, axis=count + 1)
        branch = forced_branch(state, gates, history).numpy().reshape(initial.shape)
        if policy == "reset":
            projected = expected[history]
            expected = np.zeros_like(expected)
            expected[(0,) * count] = projected
        np.testing.assert_allclose(branch, expected, rtol=0, atol=2e-15)
        assert np.count_nonzero(branch[..., 1]) == 0
    assert all(e.dirty is False for e in bound.events)
    assert sum(e.kind == "retire" for e in bound.events) == (
        count if policy == "retire" else 0
    )


@pytest.mark.parametrize("policy", ["retire", "reset"])
def test_borrowed_adopted_wire_refuses_even_with_output_transfer(
    policy: Literal["retire", "reset"],
) -> None:
    with pytest.raises(ValueError, match="borrowed"):
        bind_register_program(
            ProgramRecord((measured_function(),)),
            ((0,),),
            input_ownership=((0, True),),
            measured_workspace=policy,
            allow_workspace_outputs=True,
        )


@pytest.mark.parametrize("policy", [None, True, "free", 1, []])
def test_invalid_policy_fails_before_program_validation(policy: Any) -> None:
    with pytest.raises(ValueError, match="workspace policy"):
        bind_register_program(ProgramRecord(()), (), measured_workspace=policy)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 2, 3, 8])
@pytest.mark.parametrize("policy", ["retire", "reset"])
def test_field_erasure_preserves_source_stream_and_all_small_branch_phases(
    width: int, policy: Literal["retire", "reset"]
) -> None:
    from flagquantum.compiler.registers.finite_field.product_erasure_program import (
        product_erasure_program,
    )
    from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
    from tests.unit.test_field_erasure_program import state_with_known_product

    inputs = tuple(tuple(range(i * width, (i + 1) * width)) for i in range(3))
    record = product_erasure_program(FiniteFieldEncoding(2, width).register_type())
    plain = bind_register_program(record, inputs, clean=(3 * width + 1,))
    bound = bind_register_program(
        record,
        inputs,
        clean=(3 * width + 1,),
        input_ownership=tuple((wire, False) for wire in inputs[2]),
        measured_workspace=policy,
    )
    gates = bound.circuit.instructions
    assert tuple(g for g in gates if g.name != "reset") == plain.circuit.instructions
    assert Counter(g.name for g in gates) == Counter(
        {
            "h": width,
            "measure": width,
            "classical_xor": width,
            "cz": width**2,
            **({"reset": width} if policy == "reset" else {}),
        }
    )
    assert len(bound.events) == width
    assert all(
        e.kind == ("free" if policy == "reset" else "retire") for e in bound.events
    )
    if width == 8:
        return  # The eight-bit witness checks gate resources, small fields exhaust branches.
    state, logical = state_with_known_product(width)
    measured = [g.wires[0] for g in gates if g.name == "measure"]
    for history in product((0, 1), repeat=width):
        branch = forced_branch(state, gates, history).reshape((2,) * (3 * width + 2))
        indices: list[int | slice] = [slice(None)] * branch.ndim
        for wire, outcome in zip(measured, history, strict=True):
            indices[wire] = 0 if policy == "reset" else outcome
        indices[-1] = 0
        actual = branch[tuple(indices)].flatten()
        torch.testing.assert_close(
            actual, logical / 2 ** (width / 2), rtol=0, atol=2e-15
        )
        assert torch.count_nonzero(branch) == torch.count_nonzero(actual)


@pytest.mark.parametrize("strategy", ["trajectory", "batched"])
@pytest.mark.parametrize("policy", ["retire", "reset"])
def test_actual_runtime_keeps_discarded_snapshot_after_workspace_retirement(
    strategy: str, policy: Literal["retire", "reset"]
) -> None:
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    bound = bind_register_program(
        discard_feedback_program(2, True),
        tuple((i,) for i in range(5)),
        clean=(5,),
        input_ownership=((0, False), (1, False)),
        measured_workspace=policy,
    )
    state = torch.zeros(64, dtype=torch.complex128)
    state[::2] = torch.randn(
        32, dtype=torch.complex128, generator=torch.Generator().manual_seed(108431)
    )
    state /= state.norm()
    circuit = DynamicCircuit(6, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(6, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=32, seed=1084, strategy=strategy)
    for bits, actual in zip(
        result.classical_bits, result.final_states.reshape(32, 64), strict=True
    ):
        history = tuple(int(bits[i]) for i in range(2))
        expected = forced_branch(state, bound.circuit.instructions, history)
        expected /= expected.norm()
        torch.testing.assert_close(actual, expected, rtol=0, atol=3e-15)
        assert int(bits[2]) == sum(history) % 2
        assert torch.count_nonzero(actual[1::2]) == 0
        if policy == "reset":
            assert torch.count_nonzero(actual[16:]) == 0
