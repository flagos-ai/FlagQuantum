"""Prove atomic guarded assignments and measured-channel reverse legalization."""

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
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import CapabilityError
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.density_matrix import density_matrix_from_ir
from tests.unit.measurement_checks import expand_reset_branches, forced_branch

pytestmark = pytest.mark.unit


def _assignment(
    sources: tuple[int, ...], target: int, constant: bool, guards: tuple[int, ...]
) -> Instruction:
    base = ClassicalXor(sources, target, constant).instruction()
    return Instruction(
        base.name,
        base.wires,
        base.params,
        metadata={**base.metadata, "conditions": tuple((bit, 1) for bit in guards)},
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("target", range(3))
@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize("sources", [(), (0,), (0, 1), (1, 0, 1), (0, 1, 2, 2)])
@pytest.mark.parametrize("guards", [(0,), (0, 1), (0, 1, 2)])
def test_native_atomic_old_reads_and_guard_snapshots(
    target: int, constant: bool, sources: tuple[int, ...], guards: tuple[int, ...]
) -> None:
    km = pytest.importorskip("kickmix")
    assert hasattr(km, "Simulator")
    setup = tuple(ClassicalXor((), bit).instruction() for bit in range(3))
    result = lower_reversible_phase(
        CircuitIR(1, (*setup, _assignment(sources, target, constant, guards)))
    )
    fragment = ReversiblePhaseProgram(1, 3, result.operations[3:], input_bits=(0, 1, 2))
    assert result.n_bits == 3
    assert not any(op.qubits for op in fragment.operations)
    assert not any(op.opcode == "bit_store0" for op in fragment.operations)
    assert [
        op.condition_bit for op in fragment.operations if op.opcode == "push_condition"
    ] == list(guards)
    assert decode_kmx(encode_kmx(fragment)) == fragment
    inputs = list(product((0, 1), repeat=3))
    sim = km.Simulator(batch_size=64, seed=42)
    for bit in range(3):
        sim.write_across_shots(
            km.b(bit), sum(inputs[shot % 8][bit] << shot for shot in range(64))
        )
    sim.do(km.Circuit(encode_kmx(fragment)))
    columns = [sim.read_across_shots(km.b(bit), out=int) for bit in range(3)]
    for shot in range(64):
        old = inputs[shot % 8]
        expected = list(old)
        if all(old[guard] for guard in guards):
            expected[target] = (sum(old[bit] for bit in sources) + constant) % 2
        assert [(column >> shot) & 1 for column in columns] == expected


@pytest.mark.parametrize("strategy", ["trajectory", "batched"])
def test_local_round_trip_aliases_and_literal_replacement(strategy: str) -> None:
    for old in product((0, 1), repeat=3):
        setup = tuple(
            ClassicalXor((), bit, bool(value)).instruction()
            for bit, value in enumerate(old)
        )
        instructions = (*setup, _assignment((0, 1, 0, 2), 0, True, (0, 1)))
        ir = CircuitIR(1, instructions)
        lowered = lower_reversible_phase(ir)
        rebound = bind_reversible_phase_execution(decode_kmx(encode_kmx(lowered)))
        circuits = [
            DynamicCircuit(1).append(FieldOperation(1, gates))
            for gates in (instructions, rebound)
        ]
        results = [
            run_dynamic(circuit, shots=8, seed=11, strategy=strategy)
            for circuit in circuits
        ]
        assert torch.equal(results[0].classical_bits, results[1].classical_bits[:, :3])
        torch.testing.assert_close(
            results[0].final_states, results[1].final_states, rtol=0, atol=0
        )
        assert (
            results[1].statistics["measurement_count"]
            == results[1].statistics["reset_count"]
            == 0
        )


@pytest.mark.parametrize("guards", [(), (0,), (0, 1)])
@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_guarded_hmr_and_r_round_trip_density(
    guards: tuple[int, ...], alias: bool, enabled: bool
) -> None:
    result = 0 if alias else 2
    conditions = tuple((bit, 1) for bit in guards)
    setup = tuple(
        ClassicalXor((), bit, enabled if bit == 0 else True).instruction()
        for bit in range(3)
    )
    from flagquantum.compiler.artifacts.reversible_phase import (
        ReversiblePhaseOperation as Op,
    )

    markers = tuple(Op("push_condition", (), condition_bit=guard) for guard in guards)
    source = ReversiblePhaseProgram(
        3,
        3,
        (
            *markers,
            Op("hmr", (0,), result_bit=result),
            Op("r", (1,)),
            *(Op("pop_condition", ()) for _ in guards),
        ),
        input_bits=(0, 1, 2),
    )
    original = (*setup, *bind_reversible_phase_execution(source))
    lower = lower_reversible_phase(CircuitIR(3, original))
    assert [op.opcode for op in lower.operations if op.qubits] == ["hmr", "r"]
    generator = torch.Generator().manual_seed(129)
    initial = torch.complex(
        torch.randn(8, generator=generator, dtype=torch.float64),
        torch.randn(8, generator=generator, dtype=torch.float64),
    )
    initial /= initial.norm()
    circuits = [
        DynamicCircuit(3, inputs=initial, dtype=torch.complex128).append(
            FieldOperation(3, gates)
        )
        for gates in (original, bind_reversible_phase_execution(lower))
    ]
    torch.testing.assert_close(
        density_matrix_from_ir(circuits[0]),
        density_matrix_from_ir(circuits[1]),
        rtol=2e-14,
        atol=2e-14,
    )
    expanded = [
        expand_reset_branches(tuple(circuit.to_ir().instructions), first_slot=1000)
        for circuit in circuits
    ]
    events = sum(gate.name == "measure" for gate in expanded[0])
    assert events == sum(gate.name == "measure" for gate in expanded[1])
    probability = 0.0
    for outcomes in product((0, 1), repeat=events):
        branches = [forced_branch(initial, gates, outcomes) for gates in expanded]
        torch.testing.assert_close(branches[0], branches[1], rtol=3e-14, atol=3e-14)
        probability += float(branches[0].abs().square().sum())
    assert abs(probability - 1) < 2e-14
    assert not conditions or any(
        op.opcode == "push_condition" or op.condition_bit is not None
        for op in lower.operations
    )


def test_unowned_classical_holes_are_not_overwritten_on_rebind() -> None:
    ir = CircuitIR(1, (ClassicalXor((), 4, True).instruction(),))
    program = lower_reversible_phase(ir)
    assert program.input_bits == (0, 1, 2, 3)
    assert decode_kmx(encode_kmx(program)) == program
    rebound = bind_reversible_phase_execution(program)
    assert len(rebound) == 3  # initialize owned slot, zero assignment, invert
    assert {gate.metadata["classical_bit"] for gate in rebound} == {4}


@pytest.mark.parametrize("name", ["hmr", "r"])
def test_mismatched_effect_guards_fail_closed(name: str) -> None:
    setup = (ClassicalXor((), 0, True).instruction(), ClassicalXor((), 1).instruction())
    basis = Instruction("h", (0,), metadata={"conditions": ((0, 1),)})
    if name == "r":
        tail = (
            Instruction(
                "reset",
                (0,),
                metadata={
                    "is_dynamic": True,
                    "guarded_effect": True,
                    "conditions": ((1, 1),),
                },
            ),
        )
    else:
        tail = (
            Instruction(
                "measure", (0,), metadata={"is_dynamic": True, "classical_bit": 1}
            ),
            Instruction("x", (0,), metadata={"conditions": ((1, 1),)}),
        )
    with pytest.raises(CapabilityError):
        lower_reversible_phase(CircuitIR(1, (*setup, basis, *tail)))


def test_negative_parity_predicate_uses_owned_classical_scratch() -> None:
    assignment = ClassicalXor((0,), 0, True).instruction()
    guarded = Instruction(
        assignment.name,
        (),
        assignment.params,
        metadata={**assignment.metadata, "conditions": ((0, 0),)},
    )
    program = lower_reversible_phase(
        CircuitIR(1, (ClassicalXor((), 0).instruction(), guarded))
    )
    assert program.n_bits == 2
    assert not any(op.qubits for op in program.operations)


def _unlookup_core(
    n: int, mode: int, classical: tuple[int, int]
) -> tuple[CircuitIR, torch.Tensor]:
    from flagquantum.compiler.registers.finite_field import (
        ClassicalFieldCoefficient as C,
    )
    from flagquantum.compiler.registers.measured_unlookup import (
        measured_unlookup_operations,
    )
    from flagquantum.compiler.registers.recursive_lookup import (
        recursive_lookup_operations,
    )

    output = n
    spectator = n + 1
    work = tuple(range(n + 2, 2 * n + 2))
    table = tuple(
        (bool(row.bit_count() % 2),) if mode == 0 else (C(row % 2),)
        for row in range(1 << n)
    )
    phase = tuple(range(2, 2 + (1 << n)))
    forward = recursive_lookup_operations(
        table, tuple(range(n)), (output,), work, measurement_bit=phase[0]
    )
    inverse = measured_unlookup_operations(
        table,
        tuple(range(n)),
        (output,),
        work,
        phase,
        measurement_bit=phase[-1] + 1,
        reused_phase_bits=(phase[0],),
    )
    program = ReversiblePhaseProgram(
        2 * n + 2, phase[-1] + 2, (*forward, *inverse), input_bits=(0, 1)
    )
    setup = tuple(
        ClassicalXor((), slot, bool(value)).instruction()
        for slot, value in enumerate(classical)
    )
    ir = CircuitIR(
        program.n_qubits, (*setup, *bind_reversible_phase_execution(program))
    )
    generator = torch.Generator().manual_seed(589)
    psi = torch.complex(
        torch.randn((1 << n, 2), generator=generator, dtype=torch.float64),
        torch.randn((1 << n, 2), generator=generator, dtype=torch.float64),
    )
    psi /= psi.norm()
    initial = torch.zeros(1 << ir.n_wires, dtype=torch.complex128)
    for address in range(1 << n):
        for s in range(2):
            initial[
                (address << (ir.n_wires - n)) | (s << (ir.n_wires - 1 - spectator))
            ] = psi[address, s]
    return ir, initial


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("mode", [0, 1])
def test_unlookup_full_reverse_route_every_complex_branch(n: int, mode: int) -> None:
    ir, initial = _unlookup_core(n, mode, (1, 0))
    artifact = decode_kmx(encode_kmx(lower_reversible_phase(ir)))
    rebound = bind_reversible_phase_execution(artifact)

    def quantum(ops: tuple[Instruction, ...]) -> list[tuple[str, tuple[int, ...]]]:
        return [(gate.name, gate.wires) for gate in ops if gate.wires]

    assert quantum(ir.instructions) == quantum(rebound)
    original = expand_reset_branches(ir.instructions, first_slot=1000)
    restored = expand_reset_branches(rebound, first_slot=1000)
    events = sum(gate.name == "measure" for gate in original)
    assert events == sum(gate.name == "measure" for gate in restored)
    for outcomes in product((0, 1), repeat=events):
        actual = forced_branch(initial, restored, outcomes)
        expected = forced_branch(initial, original, outcomes)
        torch.testing.assert_close(actual, expected, rtol=4e-14, atol=4e-14)
        torch.testing.assert_close(
            actual, initial / 2 ** (events / 2), rtol=4e-14, atol=4e-14
        )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("classical", list(product((0, 1), repeat=2)))
def test_unlookup_reverse_route_native_exact_phases(
    n: int, mode: int, classical: tuple[int, int]
) -> None:
    from fractions import Fraction

    km = pytest.importorskip("kickmix")
    assert hasattr(km, "Simulator")
    ir, _ = _unlookup_core(n, mode, classical)
    artifact = lower_reversible_phase(ir)
    sim = km.Simulator(batch_size=64, seed=23)
    for wire in range(n):
        sim.write_across_shots(
            km.q(wire),
            sum(
                (((shot % (1 << n)) >> (n - 1 - wire)) & 1) << shot
                for shot in range(64)
            ),
        )
    sim.write_across_shots(km.q(n + 1), sum((shot % 2) << shot for shot in range(64)))
    sim.do(km.Circuit(encode_kmx(artifact)))
    for wire in range(n):
        assert sim.read_across_shots(km.q(wire), out=int) == sum(
            (((shot % (1 << n)) >> (n - 1 - wire)) & 1) << shot for shot in range(64)
        )
    for wire in (n, *range(n + 2, 2 * n + 2)):
        assert sim.read_across_shots(km.q(wire), out=int) == 0
    assert sim.read_across_shots(km.q(n + 1), out=int) == sum(
        (shot % 2) << shot for shot in range(64)
    )
    assert all(Fraction(sim.read_shot_phase(shot)) % 2 == 0 for shot in range(64))


@pytest.mark.parametrize("strategy", ["trajectory", "batched"])
def test_unlookup_reverse_route_preserves_original_classical_outputs(
    strategy: str,
) -> None:
    ir, initial = _unlookup_core(2, 1, (1, 0))
    artifact = lower_reversible_phase(ir)
    pairs = [
        DynamicCircuit(ir.n_wires, inputs=initial, dtype=torch.complex128).append(
            FieldOperation(ir.n_wires, gates)
        )
        for gates in (ir.instructions, bind_reversible_phase_execution(artifact))
    ]
    original, restored = [
        run_dynamic(circuit, shots=32, seed=71, strategy=strategy) for circuit in pairs
    ]
    torch.testing.assert_close(
        original.final_states, restored.final_states, rtol=0, atol=0
    )
    assert torch.equal(
        original.classical_bits,
        restored.classical_bits[:, : original.classical_bits.shape[1]],
    )
    assert (
        original.statistics["measurement_count"]
        == restored.statistics["measurement_count"]
    )
    assert original.statistics["reset_count"] == restored.statistics["reset_count"]


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("alias", [False, True])
def test_guarded_native_channels_preserve_seeded_exact_phase(
    enabled: bool, nested: bool, alias: bool
) -> None:
    from fractions import Fraction

    from flagquantum.compiler.artifacts.reversible_phase import (
        ReversiblePhaseOperation as Op,
    )

    km = pytest.importorskip("kickmix")
    assert hasattr(km, "Simulator")
    guards = (0, 1) if nested else (0,)
    source = ReversiblePhaseProgram(
        3,
        3,
        (
            *(Op("push_condition", (), condition_bit=bit) for bit in guards),
            Op("hmr", (0,), result_bit=0 if alias else 2),
            Op("r", (1,)),
            *(Op("pop_condition", ()) for _ in guards),
        ),
        input_bits=(0, 1, 2),
    )
    values = (enabled, True, True)
    setup = tuple(
        ClassicalXor((), bit, value).instruction() for bit, value in enumerate(values)
    )
    core = CircuitIR(3, (*setup, *bind_reversible_phase_execution(source)))
    restored = lower_reversible_phase(core)
    initialization = [f"BIT_STORE0 b{bit}" for bit in range(3)] + [
        f"BIT_INVERT b{bit}" for bit, value in enumerate(values) if value
    ]
    texts = (
        "\n".join((*initialization, *encode_kmx(source).splitlines())),
        encode_kmx(restored),
    )
    sims = [km.Simulator(batch_size=64, seed=73) for _ in range(2)]
    for sim, text in zip(sims, texts, strict=True):
        for wire in range(3):
            sim.write_across_shots(
                km.q(wire), sum(((shot >> wire) & 1) << shot for shot in range(64))
            )
        sim.do(km.Circuit(text))
    for wire in range(3):
        assert sims[0].read_across_shots(km.q(wire), out=int) == sims[
            1
        ].read_across_shots(km.q(wire), out=int)
    for bit in range(3):
        assert sims[0].read_across_shots(km.b(bit), out=int) == sims[
            1
        ].read_across_shots(km.b(bit), out=int)
    assert [Fraction(sims[0].read_shot_phase(shot)) % 2 for shot in range(64)] == [
        Fraction(sims[1].read_shot_phase(shot)) % 2 for shot in range(64)
    ]


@pytest.mark.parametrize("strategy", ["trajectory", "batched"])
def test_unowned_holes_preserve_enclosing_caller_values(strategy: str) -> None:
    ir = CircuitIR(1, (ClassicalXor((), 4, True).instruction(),))
    restored = bind_reversible_phase_execution(lower_reversible_phase(ir))
    caller = tuple(
        ClassicalXor((), bit, value).instruction()
        for bit, value in enumerate((False, True, True, False))
    )
    results = [
        run_dynamic(
            DynamicCircuit(1).append(FieldOperation(1, (*caller, *gates))),
            shots=4,
            seed=1,
            strategy=strategy,
        )
        for gates in (ir.instructions, restored)
    ]
    assert torch.equal(results[0].classical_bits, results[1].classical_bits)
    assert torch.all(results[1].classical_bits == torch.tensor([0, 1, 1, 0, 1]))
