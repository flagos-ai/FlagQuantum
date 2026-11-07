"""Prove negative masks, alias-safe parity and shared quantum predicate binding."""

from itertools import product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseProgram
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.global_phase import ccz_ir, scalar_sign_ir
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import expand_reset_branches, forced_branch

pytestmark = pytest.mark.unit


def _assignment(
    sources: tuple[int, ...],
    target: int,
    constant: bool,
    conditions: tuple[tuple[int, int], ...],
) -> Instruction:
    base = ClassicalXor(sources, target, constant).instruction()
    return Instruction(
        base.name,
        base.wires,
        base.params,
        metadata={**base.metadata, "conditions": conditions},
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("polarity", list(product((0, 1), repeat=3)))
@pytest.mark.parametrize("target", range(3))
@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize("sources", [(), (0,), (0, 1), (1, 0, 1), (0, 1, 2, 2)])
def test_native_complete_atomic_truth_tables(
    polarity: tuple[int, ...], target: int, constant: bool, sources: tuple[int, ...]
) -> None:
    km = pytest.importorskip("kickmix")
    assert hasattr(km, "Simulator")
    setup = tuple(ClassicalXor((), bit).instruction() for bit in range(3))
    conditions = tuple(enumerate(polarity))
    program = lower_reversible_phase(
        CircuitIR(1, (*setup, _assignment(sources, target, constant, conditions)))
    )
    assert program.n_bits == 3 + polarity.count(0)
    fragment = ReversiblePhaseProgram(
        1, program.n_bits, program.operations[3:], input_bits=(0, 1, 2)
    )
    assert decode_kmx(encode_kmx(fragment)) == fragment
    assert not any(op.qubits for op in fragment.operations)
    bits = tuple(product((0, 1), repeat=3))
    sim = km.Simulator(batch_size=64, seed=31)
    for bit in range(3):
        sim.write_across_shots(
            km.b(bit), sum(bits[shot % 8][bit] << shot for shot in range(64))
        )
    sim.do(km.Circuit(encode_kmx(fragment)))
    columns = [sim.read_across_shots(km.b(bit), out=int) for bit in range(3)]
    for shot in range(64):
        old = bits[shot % 8]
        expected = list(old)
        if all(old[bit] == value for bit, value in conditions):
            expected[target] = (sum(old[bit] for bit in sources) + constant) % 2
        assert [(column >> shot) & 1 for column in columns] == expected


def _gates(
    name: str, conditions: tuple[tuple[int, int], ...]
) -> tuple[Instruction, ...]:
    if name == "neg":
        return scalar_sign_ir(0, conditions=conditions)
    if name == "ccz":
        return ccz_ir((0, 1, 2), conditions=conditions)
    wires = {
        "x": (0,),
        "cx": (0, 1),
        "ccx": (0, 1, 2),
        "swap": (0, 1),
        "cswap": (0, 1, 2),
        "z": (0,),
        "cz": (0, 1),
    }[name]
    return (Instruction(name, wires, metadata={"conditions": conditions}),)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "name", ["x", "cx", "ccx", "swap", "cswap", "z", "cz", "ccz", "neg"]
)
@pytest.mark.parametrize("polarity", list(product((0, 1), repeat=2)))
def test_native_quantum_maps_and_exact_phases(
    name: str, polarity: tuple[int, ...]
) -> None:
    from fractions import Fraction

    km = pytest.importorskip("kickmix")
    assert hasattr(km, "Simulator")
    conditions = tuple(enumerate(polarity))
    ir = CircuitIR(
        3,
        (
            *(ClassicalXor((), bit).instruction() for bit in range(2)),
            *_gates(name, conditions),
        ),
    )
    program = lower_reversible_phase(ir)
    fragment = ReversiblePhaseProgram(
        3, program.n_bits, program.operations[2:], input_bits=(0, 1)
    )
    sim = km.Simulator(batch_size=64, seed=41)
    for wire in range(3):
        sim.write_across_shots(
            km.q(wire), sum(((shot >> wire) & 1) << shot for shot in range(64))
        )
    for bit in range(2):
        sim.write_across_shots(
            km.b(bit), sum(((shot >> (3 + bit)) & 1) << shot for shot in range(64))
        )
    sim.do(km.Circuit(encode_kmx(fragment)))
    columns = [sim.read_across_shots(km.q(wire), out=int) for wire in range(3)]
    for shot in range(64):
        q = [(shot >> wire) & 1 for wire in range(3)]
        phase = 0
        if all(((shot >> (3 + bit)) & 1) == value for bit, value in conditions):
            if name == "x":
                q[0] ^= 1
            elif name == "cx":
                q[1] ^= q[0]
            elif name == "ccx":
                q[2] ^= q[0] & q[1]
            elif name == "swap":
                q[0], q[1] = q[1], q[0]
            elif name == "cswap" and q[0]:
                q[1], q[2] = q[2], q[1]
            elif name == "z":
                phase = q[0]
            elif name == "cz":
                phase = q[0] & q[1]
            elif name == "ccz":
                phase = q[0] & q[1] & q[2]
            elif name == "neg":
                phase = 1
        assert [(column >> shot) & 1 for column in columns] == q
        assert Fraction(sim.read_shot_phase(shot)) % 2 == phase
    for bit in range(2):
        assert sim.read_across_shots(km.b(bit), out=int) == sum(
            ((shot >> (3 + bit)) & 1) << shot for shot in range(64)
        )


@pytest.mark.parametrize(
    "name", ["x", "cx", "ccx", "swap", "cswap", "z", "cz", "ccz", "neg"]
)
@pytest.mark.parametrize("polarity", [(0, 0), (0, 1), (1, 0)])
def test_local_complex_quantum_amplitudes(name: str, polarity: tuple[int, ...]) -> None:
    generator = torch.Generator().manual_seed(123)
    initial = torch.complex(
        torch.randn(8, generator=generator, dtype=torch.float64),
        torch.randn(8, generator=generator, dtype=torch.float64),
    )
    initial /= initial.norm()
    for values in product((False, True), repeat=2):
        setup = tuple(
            ClassicalXor((), bit, value).instruction()
            for bit, value in enumerate(values)
        )
        ir = CircuitIR(3, (*setup, *_gates(name, tuple(enumerate(polarity)))))
        lowered = decode_kmx(encode_kmx(lower_reversible_phase(ir)))
        pair = [
            DynamicCircuit(3, inputs=initial, dtype=torch.complex128).append(
                FieldOperation(3, gates)
            )
            for gates in (ir.instructions, bind_reversible_phase_execution(lowered))
        ]
        results = [
            run_dynamic(circuit, shots=2, seed=31, strategy="batched")
            for circuit in pair
        ]
        torch.testing.assert_close(
            results[0].final_states, results[1].final_states, atol=2e-14, rtol=2e-14
        )
        assert torch.equal(results[0].classical_bits, results[1].classical_bits[:, :2])


@pytest.mark.parametrize("value", [0, 1])
@pytest.mark.parametrize("polarity", [0, 1])
@pytest.mark.parametrize("alias", [False, True])
def test_every_guarded_measure_reset_branch_recomputes_scratch(
    value: int, polarity: int, alias: bool
) -> None:
    setup = [
        ClassicalXor((), 0, bool(value)).instruction(),
        ClassicalXor((), 1).instruction(),
    ]
    guard, result = (3, 0) if alias else (0, 1)
    if alias:
        setup.append(ClassicalXor((0,), 3).instruction())
    conditions = ((guard, polarity),)
    instructions = (
        *setup,
        Instruction("h", (0,), metadata={"conditions": conditions}),
        Instruction(
            "measure",
            (0,),
            metadata={
                "is_dynamic": True,
                "classical_bit": result,
                "guarded_effect": True,
                "conditions": conditions,
            },
        ),
        Instruction(
            "x",
            (0,),
            metadata={"conditions": tuple(sorted((*conditions, (result, 1))))},
        ),
        Instruction("h", (1,), metadata={"conditions": ((result, 0),)}),
        Instruction(
            "reset",
            (1,),
            metadata={
                "is_dynamic": True,
                "guarded_effect": True,
                "conditions": ((result, 0),),
            },
        ),
    )
    ir = CircuitIR(2, instructions)
    lowered = lower_reversible_phase(ir)
    assert lowered.n_bits == (5 if alias else 3)
    generator = torch.Generator().manual_seed(311)
    initial = torch.complex(
        torch.randn(4, generator=generator, dtype=torch.float64),
        torch.randn(4, generator=generator, dtype=torch.float64),
    )
    initial /= initial.norm()
    pair = [
        expand_reset_branches(gates, first_slot=1000)
        for gates in (ir.instructions, bind_reversible_phase_execution(lowered))
    ]
    probability = 0.0
    for outcomes in product((0, 1), repeat=2):
        outputs = [forced_branch(initial, gates, outcomes) for gates in pair]
        torch.testing.assert_close(outputs[0], outputs[1], atol=2e-14, rtol=2e-14)
        probability += float(outputs[0].abs().square().sum())
    assert abs(probability - 1) < 1e-14


def test_scratch_reuse_and_enclosing_caller_holes() -> None:
    setup = tuple(ClassicalXor((), bit).instruction() for bit in (0, 1, 4))
    ir = CircuitIR(
        1,
        (
            *setup,
            _assignment((1,), 0, True, ((0, 0), (1, 0))),
            _assignment((0,), 4, False, ((1, 0),)),
            Instruction("x", (0,), metadata={"conditions": ((4, 0),)}),
        ),
    )
    program = lower_reversible_phase(ir)
    assert program.n_bits == 7  # original width5 plus maximum two complements
    assert program.input_bits == (2, 3)
    complement_stores = [
        op.result_bit
        for op in program.operations
        if op.opcode == "bit_store0"
        and op.result_bit is not None
        and op.result_bit >= 5
    ]
    assert complement_stores == [5, 6, 5, 5]
    assert not any(op.qubits for op in program.operations if op.result_bit is not None)


def test_disjunction_uses_an_owned_single_gate_mask() -> None:
    ir = CircuitIR(
        1,
        (
            ClassicalXor((), 0).instruction(),
            ClassicalXor((), 1).instruction(),
            Instruction(
                "x", (0,), metadata={"condition_clauses": (((0, 0),), ((1, 1),))}
            ),
        ),
    )
    program = lower_reversible_phase(ir)
    assert program.n_bits == 5
    assert program.operations[-1].opcode == "x"
    assert program.operations[-1].condition_bit == 3


def test_runnable_alias_and_negative_guard_example() -> None:
    from examples.algorithms.negative_classical_predicates import main

    main()
