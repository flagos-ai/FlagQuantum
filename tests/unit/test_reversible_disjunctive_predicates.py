"""Verify Boolean OR masks, overlapping clauses and single quantum application."""

from itertools import combinations, product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseProgram
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import ValidationError
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_reversible_negative_predicates import _gates

pytestmark = pytest.mark.unit
Clause = tuple[tuple[int, int], ...]
Clauses = tuple[Clause, ...]


def _canonical(clauses: Clauses) -> Clauses:
    return tuple(sorted(clauses, key=lambda clause: (len(clause), clause)))


FAMILIES: tuple[Clauses, ...] = tuple(
    _canonical(clauses)
    for clauses in (
        (((0, 1),), ((1, 1),)),
        (((0, 0),), ((1, 1),)),
        (((0, 0),), ((0, 1),)),
        (((0, 1), (1, 1)), ((1, 1), (2, 1))),
        (((0, 0), (1, 1)), ((0, 1), (1, 0))),
        (((0, 0), (1, 1)), ((0, 1), (2, 1)), ((1, 0), (2, 0))),
        (((0, 0),), ((1, 1),), ((2, 0),)),
        (((0, 0), (1, 0)), ((0, 0), (2, 0))),
    )
)


def _quantum(name: str, clauses: Clauses) -> tuple[Instruction, ...]:
    base = _gates(name, ((0, 1),))
    return tuple(
        (
            Instruction(
                gate.name,
                gate.wires,
                gate.params,
                metadata={
                    **{
                        key: value
                        for key, value in gate.metadata.items()
                        if key != "conditions"
                    },
                    "condition_clauses": clauses,
                },
            )
            if "conditions" in gate.metadata
            else gate
        )
        for gate in base
    )


def _pairs() -> tuple[Clauses, ...]:
    clauses = [
        tuple((bit, value) for bit, value in enumerate(values) if value != -1)
        for values in product((-1, 0, 1), repeat=3)
        if values != (-1, -1, -1)
    ]
    return tuple(
        _canonical((left, right))
        for left, right in combinations(clauses, 2)
        if not (set(left) <= set(right) or set(right) <= set(left))
    )


@pytest.mark.ftqc_reference
def test_native_all_minimal_three_bit_clause_pairs() -> None:
    km = pytest.importorskip("kickmix")
    assert hasattr(km, "Simulator")
    families = _pairs()
    assert len(families) == 253
    setup = tuple(ClassicalXor((), bit).instruction() for bit in range(3))
    for clauses in families:
        program = lower_reversible_phase(
            CircuitIR(1, (*setup, *_quantum("x", clauses)))
        )
        negative = {bit for clause in clauses for bit, value in clause if not value}
        assert program.n_bits == 3 + len(negative) + 2
        fragment = ReversiblePhaseProgram(
            1, program.n_bits, program.operations[3:], input_bits=(0, 1, 2)
        )
        assert decode_kmx(encode_kmx(fragment)) == fragment
        assert len([op for op in fragment.operations if op.qubits]) == 1
        sim = km.Simulator(batch_size=64, seed=17)
        original = [
            sum(((shot >> bit) & 1) << shot for shot in range(64)) for bit in range(3)
        ]
        for bit, packed in enumerate(original):
            sim.write_across_shots(km.b(bit), packed)
        sim.write_across_shots(
            km.q(0), sum(((shot >> 3) & 1) << shot for shot in range(64))
        )
        sim.do(km.Circuit(encode_kmx(fragment)))
        expected = sum(
            (
                ((shot >> 3) & 1)
                ^ int(
                    any(
                        all(((shot >> bit) & 1) == value for bit, value in clause)
                        for clause in clauses
                    )
                )
            )
            << shot
            for shot in range(64)
        )
        assert sim.read_across_shots(km.q(0), out=int) == expected
        assert [
            sim.read_across_shots(km.b(bit), out=int) for bit in range(3)
        ] == original


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "name", ["x", "cx", "ccx", "swap", "cswap", "z", "cz", "ccz", "neg"]
)
@pytest.mark.parametrize("clauses", FAMILIES)
def test_native_quantum_basis_and_fraction_phases(name: str, clauses: Clauses) -> None:
    from fractions import Fraction

    km = pytest.importorskip("kickmix")
    assert hasattr(km, "Simulator")
    program = lower_reversible_phase(
        CircuitIR(
            3,
            (
                *(ClassicalXor((), bit).instruction() for bit in range(3)),
                *_quantum(name, clauses),
            ),
        )
    )
    fragment = ReversiblePhaseProgram(
        3, program.n_bits, program.operations[3:], input_bits=(0, 1, 2)
    )
    sim = km.Simulator(batch_size=64, seed=21)
    for q in range(3):
        sim.write_across_shots(
            km.q(q), sum(((shot >> q) & 1) << shot for shot in range(64))
        )
    for bit in range(3):
        sim.write_across_shots(
            km.b(bit), sum(((shot >> (bit + 3)) & 1) << shot for shot in range(64))
        )
    sim.do(km.Circuit(encode_kmx(fragment)))
    columns = [sim.read_across_shots(km.q(q), out=int) for q in range(3)]
    for shot in range(64):
        q = [(shot >> wire) & 1 for wire in range(3)]
        phase = 0
        active = any(
            all(((shot >> (bit + 3)) & 1) == value for bit, value in clause)
            for clause in clauses
        )
        if active:
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


@pytest.mark.parametrize(
    "name", ["x", "cx", "ccx", "swap", "cswap", "z", "cz", "ccz", "neg"]
)
@pytest.mark.parametrize("clauses", FAMILIES)
def test_local_full_complex_vectors(name: str, clauses: Clauses) -> None:
    generator = torch.Generator().manual_seed(813)
    initial = torch.complex(
        torch.randn(8, generator=generator, dtype=torch.float64),
        torch.randn(8, generator=generator, dtype=torch.float64),
    )
    initial /= initial.norm()
    for values in product((False, True), repeat=3):
        setup = tuple(
            ClassicalXor((), bit, value).instruction()
            for bit, value in enumerate(values)
        )
        ir = CircuitIR(3, (*setup, *_quantum(name, clauses)))
        artifact = lower_reversible_phase(ir)
        circuits = [
            DynamicCircuit(3, inputs=initial, dtype=torch.complex128).append(
                FieldOperation(3, gates)
            )
            for gates in (ir.instructions, bind_reversible_phase_execution(artifact))
        ]
        pair = [
            run_dynamic(circuit, shots=2, seed=42, strategy="batched")
            for circuit in circuits
        ]
        torch.testing.assert_close(
            pair[0].final_states, pair[1].final_states, atol=2e-14, rtol=2e-14
        )
        assert torch.equal(pair[0].classical_bits, pair[1].classical_bits[:, :3])


def test_resource_counts_shared_literals_and_scratch_reuse() -> None:
    clauses = FAMILIES[-1]  # NOT b0 appears in both clauses.
    setup = tuple(ClassicalXor((), bit).instruction() for bit in range(3))
    program = lower_reversible_phase(
        CircuitIR(1, (*setup, *_quantum("x", clauses), *_quantum("z", FAMILIES[0])))
    )
    assert (
        program.n_bits == 8
    )  # three inputs + three negative literals + accumulator/unset
    assert len([op for op in program.operations if op.qubits]) == 2
    # The first predicate binds b0's complement once, not once per occurrence.
    stores = [
        op.result_bit for op in program.operations[3:] if op.opcode == "bit_store0"
    ]
    assert stores[:4] == [3, 4, 5, 6]
    assert stores.count(7) == 2
    assert stores[-3:] == [3, 4, 4]


@pytest.mark.parametrize(
    "clauses",
    [(((0, 1),), ((0, 1),)), (((0, 1),), ((0, 1), (1, 0))), (((1, 0),), ((0, 1),))],
)
def test_noncanonical_disjunction_is_not_silently_normalized(clauses: Clauses) -> None:
    setup = tuple(ClassicalXor((), bit).instruction() for bit in range(2))
    with pytest.raises(ValidationError):
        lower_reversible_phase(CircuitIR(1, (*setup, *_quantum("x", clauses))))


def test_classical_assignment_disjunction_remains_outside_core_grammar() -> None:
    with pytest.raises(ValidationError):
        Instruction(
            "classical_xor",
            (),
            params={"sources": ()},
            metadata={
                "is_dynamic": True,
                "classical_bit": 0,
                "condition_clauses": FAMILIES[0],
            },
        )


@pytest.mark.parametrize("strategy", ["trajectory", "batched"])
def test_updated_predicate_example_preserves_rng_and_caller_values(
    strategy: str,
) -> None:
    from examples.algorithms.disjunctive_classical_predicates import verify

    verify(strategy)


def test_guarded_effect_disjunction_stays_outside_core_grammar() -> None:
    setup = tuple(ClassicalXor((), bit).instruction() for bit in range(3))
    effect = Instruction(
        "measure",
        (0,),
        metadata={
            "is_dynamic": True,
            "guarded_effect": True,
            "classical_bit": 2,
            "condition_clauses": FAMILIES[0],
        },
    )
    with pytest.raises(ValidationError, match="guarded effect"):
        lower_reversible_phase(CircuitIR(1, (*setup, effect)))


@pytest.mark.parametrize("clauses", FAMILIES)
def test_all_measurement_branches_clear_quantum_work_and_recompute_mask(
    clauses: Clauses,
) -> None:
    from flagquantum.compiler.registers.primitives.measurement import measure_x_clear_ir
    from tests.unit.measurement_checks import forced_branch

    ir = CircuitIR(
        4,
        (
            *(gate for bit in range(3) for gate in measure_x_clear_ir(bit, bit)),
            Instruction("x", (3,), metadata={"condition_clauses": clauses}),
            ClassicalXor((0,), 0, True).instruction(),
            Instruction("z", (3,), metadata={"condition_clauses": clauses}),
        ),
    )
    program = decode_kmx(encode_kmx(lower_reversible_phase(ir)))
    restored = bind_reversible_phase_execution(program)
    generator = torch.Generator().manual_seed(263)
    initial = torch.complex(
        torch.randn(16, generator=generator, dtype=torch.float64),
        torch.randn(16, generator=generator, dtype=torch.float64),
    )
    initial /= initial.norm()
    probability = 0.0
    for outcomes in product((0, 1), repeat=3):
        direct = forced_branch(initial, ir.instructions, outcomes)
        rebound = forced_branch(initial, restored, outcomes)
        torch.testing.assert_close(direct, rebound, atol=2e-14, rtol=2e-14)
        assert torch.count_nonzero(rebound[2:]) == 0
        probability += float(rebound.abs().square().sum())
    assert abs(probability - 1) < 1e-14


def test_disjunction_scratch_does_not_claim_original_classical_holes() -> None:
    ir = CircuitIR(
        1,
        (
            *(ClassicalXor((), bit).instruction() for bit in (0, 2, 5)),
            Instruction(
                "x",
                (0,),
                metadata={"condition_clauses": (((0, 1),), ((2, 0), (5, 1)))},
            ),
        ),
    )
    program = lower_reversible_phase(ir)
    assert program.input_bits == (1, 3, 4)
    assert program.n_bits == 9
    stores = [
        op.result_bit for op in program.operations[3:] if op.opcode == "bit_store0"
    ]
    assert set(stores) == {6, 7, 8}
