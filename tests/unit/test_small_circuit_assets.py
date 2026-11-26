"""Validate frozen raw circuit assets, exact source branches and coherent maps."""

import hashlib
import json
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import product
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.formats.kmb import decode_kmb, encode_kmb
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.physical_basis_shots import (
    PackedBasisState,
    apply_packed_basis_records,
)
from tests.unit.circuit_asset_fixtures import ASSETS, asset, instructions
from tests.unit.conditional_erasure_certificate import certify_conditional_erasure_map
from tests.unit.measurement_checks import forced_branch

pytestmark = pytest.mark.unit


def test_complete_original_storage_and_packed_branch_replay() -> None:
    records: object = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("small_circuit_assets.json")
        .read_text()
    )
    assert isinstance(records, list) and len(records) == 6
    for row in records:
        raw = (ASSETS / row["name"]).read_bytes()
        assert (
            hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            == row["blob"]
        )
        circuit = asset(row["name"])
        assert circuit.to_kmx() == row["canonical"]
        assert encode_kmb(raw.decode()).hex() == row["kmb_hex"]
        assert decode_kmb(bytes.fromhex(row["kmb_hex"])) == circuit.to_kmx()
        assert Counter(record.opcode for record in circuit) == row["counts"]
        for case in row["cases"]:
            state = PackedBasisState(
                defaultdict(int, enumerate(case["qubits"])),
                defaultdict(int, enumerate(case["bits"])),
                [0] * 64,
            )
            draws = iter(case["draws"])
            apply_packed_basis_records(circuit, state, draws.__next__)
            assert next(draws, None) is None
            assert [state.qubits[q] for q in range(circuit.qubits)] == case[
                "final_qubits"
            ], row["name"]
            assert [state.bits[b] for b in range(circuit.bits)] == case[
                "final_bits"
            ], row["name"]
            assert [
                (phase + ((state.sign >> lane) & 1) * (1 << 127)) % (1 << 128)
                for lane, phase in enumerate(state.phases)
            ] == case["phases"], row["name"]


def coherent_state(width: int, clean: tuple[int, ...] = ()) -> torch.Tensor:
    generator = torch.Generator().manual_seed(119)
    state = torch.complex(
        torch.randn(1 << width, generator=generator, dtype=torch.float64),
        torch.randn(1 << width, generator=generator, dtype=torch.float64),
    )
    labels = torch.arange(1 << width)
    for wire in clean:
        state[(labels & (1 << (width - 1 - wire))) != 0] = 0
    state /= state.norm()
    return state


@pytest.mark.parametrize("name", ["cmul16_mod_21.kmx", "inc3.kmx"])
def test_unmeasured_assets_preserve_all_complex_columns(name: str) -> None:
    circuit = asset(name)
    width = circuit.qubits
    columns = torch.arange(1 << width)
    mapping = columns.clone()
    for record in circuit:
        assert record.opcode in {"X", "CX", "CCX"}
        active = torch.ones_like(mapping, dtype=torch.bool)
        for wire in record.qubits[:-1]:
            active &= (mapping & (1 << (width - 1 - wire))) != 0
        mapping ^= active.long() << (width - 1 - record.qubits[-1])
    assert len(set(mapping.tolist())) == 1 << width
    state = coherent_state(width)
    expected = torch.empty_like(state)
    expected[mapping] = state
    actual = (
        Circuit(width, inputs=state, dtype=torch.complex128)
        .append(FieldOperation(width, instructions(circuit)))
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    for before, after in enumerate(mapping.tolist()):
        column = torch.zeros_like(state)
        column[before] = torch.tensor(0.6 + 0.8j, dtype=torch.complex128)
        actual_column = (
            Circuit(width, inputs=column, dtype=torch.complex128)
            .append(FieldOperation(width, instructions(circuit)))
            .state()
            .flatten()
        )
        expected_column = torch.zeros_like(state)
        expected_column[after] = torch.tensor(0.6 + 0.8j, dtype=torch.complex128)
        torch.testing.assert_close(actual_column, expected_column, atol=1e-12, rtol=0)
    if name == "inc3.kmx":
        for before, after in enumerate(mapping.tolist()):
            little_before = sum(
                ((before >> (width - 1 - wire)) & 1) << wire for wire in range(3)
            )
            little_after = sum(
                ((after >> (width - 1 - wire)) & 1) << wire for wire in range(3)
            )
            assert little_after == (little_before + 1) % 8


def test_increment_eight_all_measurement_branches_and_clean_work() -> None:
    circuit = asset("inc8.kmx")
    width = circuit.qubits
    state = coherent_state(width, tuple(range(12, 17)))
    expected = torch.zeros_like(state)
    for before in range(1 << width):
        if state[before] == 0:
            continue
        value = sum(((before >> (width - 1 - wire)) & 1) << wire for wire in range(8))
        after = before
        toggles = value ^ ((value + 1) % 256)
        for wire in range(8):
            after ^= ((toggles >> wire) & 1) << (width - 1 - wire)
        expected[after] = state[before]
    gates = instructions(circuit)
    assert sum(g.name == "measure" for g in gates) == 5
    for outcomes in product((0, 1), repeat=5):
        actual = forced_branch(state.clone(), gates, outcomes)
        torch.testing.assert_close(actual, expected / 32**0.5, atol=1e-12, rtol=0)


def test_fourier_asset_all_complex_measurement_branches_clear_targets() -> None:
    circuit = asset("qft_measurement.kmx")
    width = circuit.qubits
    state = coherent_state(width)
    labels = torch.arange(32, dtype=torch.float64)
    matrix = torch.exp(2j * torch.pi * labels[:, None] * labels[None, :] / 32) / 32**0.5
    # Source q0 is the least significant Fourier input/output port.
    reversal = torch.tensor([int(f"{value:05b}"[::-1], 2) for value in range(32)])
    transformed = matrix @ state[reversal]
    gates = instructions(circuit)
    for outcomes in product((0, 1), repeat=5):
        actual = forced_branch(state.clone(), gates, outcomes)
        result = sum(bit << i for i, bit in enumerate(outcomes))
        expected = torch.zeros_like(state)
        expected[0] = transformed[result]
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("addend", [0, 1, 85, 127, 128, 255])
def test_classical_add_asset_all_erasure_branches_and_dirty_restore(
    addend: int,
) -> None:
    from dataclasses import replace

    from flagquantum.compiler.lowering.reversible_phase_execution import (
        bind_reversible_phase_execution,
    )
    from flagquantum.core.classical_dataflow import ClassicalXor
    from tests.unit.circuit_asset_fixtures import program

    circuit = asset("iadd8_qc_2clean_6dirty.kmx")
    # Each explicit R consumes an initially clean or immediately cleared lane.
    zero = {8, 9}
    resets = 0
    for record in circuit:
        code = record.opcode
        if code == "R":
            assert record.qubits[0] in zero
            resets += 1
        elif code == "HMR":
            zero.add(record.qubits[0])
        elif code in {"X", "CX", "CCX"}:
            zero.discard(record.qubits[-1])
        elif code == "SWAP":
            for wire in record.qubits:
                zero.discard(wire)
    assert resets == 6
    # On |0>, each reset's two X-basis Kraus maps is exactly I/sqrt(2).
    # All 64 reset histories therefore share each retained erasure branch.
    original = program(circuit)
    reduced = replace(
        original, operations=tuple(op for op in original.operations if op.opcode != "r")
    )
    setup = tuple(
        ClassicalXor((), bit, constant=bool((addend >> bit) & 1)).instruction()
        for bit in range(circuit.bits)
    )
    gates = (*setup, *bind_reversible_phase_execution(reduced))
    assert sum(g.name == "measure" for g in gates) == 6
    state = coherent_state(circuit.qubits, (8, 9))
    labels = torch.arange(1 << circuit.qubits)
    value = torch.zeros_like(labels)
    for wire in range(8):
        value |= ((labels >> (15 - wire)) & 1) << wire
    toggles = value ^ ((value + addend) % 256)
    destination = labels.clone()
    for wire in range(8):
        destination ^= ((toggles >> wire) & 1) << (15 - wire)
    expected = torch.empty_like(state)
    expected[destination] = state
    for outcomes in product((0, 1), repeat=6):
        actual = forced_branch(state.clone(), gates, outcomes)
        torch.testing.assert_close(actual, expected / 64**0.5, atol=1e-12, rtol=0)


def lane_mask(position: int, count: int) -> int:
    stride = 1 << position
    if stride < 8:
        byte = {1: 0xAA, 2: 0xCC, 4: 0xF0}[stride]
        raw = bytes([byte]) * (count // 8)
    else:
        raw = (b"\0" * (stride // 8) + b"\xff" * (stride // 8)) * (
            count // (2 * stride)
        )
    return int.from_bytes(raw, "little")


def test_classical_add_entire_basis_domain_all_erasure_histories() -> None:
    circuit = asset("iadd8_qc_2clean_6dirty.kmx")
    # Lanes enumerate every dirty word, target word and classical addend.
    count = 1 << 22
    mask = (1 << count) - 1
    initial_q = {wire: lane_mask(6 + wire, count) for wire in range(8)}
    initial_q.update({10 + wire: lane_mask(wire, count) for wire in range(6)})
    initial_b = {bit: lane_mask(14 + bit, count) for bit in range(8)}
    expected_q = {
        wire: int.from_bytes(
            b"".join(
                (b"\xff" if ((addend + value) >> wire) & 1 else b"\0") * 8
                for addend in range(256)
                for value in range(256)
            ),
            "little",
        )
        for wire in range(8)
    }
    for history in range(64):
        state = PackedBasisState(
            defaultdict(int, initial_q), defaultdict(int, initial_b), [0] * count
        )
        outcomes = iter((history >> bit) & 1 for bit in range(6))
        # The source has six initially-zero resets and six erasures, interleaved.
        draws = iter(
            0 if record.opcode == "R" else mask * next(outcomes)
            for record in circuit
            if record.opcode in {"R", "HMR"}
        )
        apply_packed_basis_records(circuit, state, draws.__next__)
        assert next(draws, None) is None
        assert all(
            state.qubits[wire] == value for wire, value in expected_q.items()
        ), history
        assert state.qubits[8] == state.qubits[9] == 0
        assert all(
            state.qubits[wire] == initial_q[wire] for wire in range(10, 16)
        ), history
        assert all(state.bits[bit] == initial_b[bit] for bit in range(8)), history
        assert state.sign == 0 and not any(state.phases), history


def test_controlled_asset_original_multiplier_name_mismatch() -> None:
    circuit = asset("cmul16_mod_21.kmx")
    count = 64
    initial = {wire: lane_mask(wire, count) for wire in range(6)}
    state = PackedBasisState(defaultdict(int, initial), defaultdict(int), [0] * count)
    apply_packed_basis_records(circuit, state, lambda: 0)
    assert state.qubits[5] == initial[5]
    for control, value in product(range(2), range(21)):
        lane = value + 32 * control
        actual = sum(((state.qubits[wire] >> lane) & 1) << wire for wire in range(5))
        assert actual == (4 * value) % 21 if control else actual == value
    assert sum(((state.qubits[wire] >> 33) & 1) << wire for wire in range(5)) == 4


def test_modular_asset_branch_family_clean_work_and_phase() -> None:
    from random import Random

    circuit = asset("mul16_mod_21.kmx")
    count = 64
    initial = {8 + wire: lane_mask(wire, count) for wire in range(5)}
    initial[13] = lane_mask(5, count)
    rng = Random(19)
    for _ in range(256):
        state = PackedBasisState(
            defaultdict(int, initial), defaultdict(int), [0] * count
        )
        apply_packed_basis_records(circuit, state, lambda: rng.getrandbits(count))
        valid = (1 << 21) - 1
        valid |= valid << 32
        assert all(state.qubits[wire] & valid == 0 for wire in range(8))
        assert state.qubits[13] == initial[13]
        assert state.sign & valid == 0
        for control, value in product(range(2), range(21)):
            lane = value + 32 * control
            actual = sum(
                ((state.qubits[8 + wire] >> lane) & 1) << wire for wire in range(5)
            )
            assert actual == (16 * value) % 21 if control else actual == value


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_modular_asset_exact_complex_branches(mode: int) -> None:
    from tests.unit.measurement_checks import expand_reset_branches

    circuit = asset("mul16_mod_21.kmx")
    width = circuit.qubits
    state = torch.zeros(1 << width, dtype=torch.complex128)
    coefficients = coherent_state(6)
    addresses = []
    for control, value in product(range(2), range(21)):
        label = control | sum(((value >> wire) & 1) << (5 - wire) for wire in range(5))
        addresses.append((control, value, label))
        state[label] = coefficients[label]
    state /= state.norm()
    gates = expand_reset_branches(instructions(circuit), first_slot=circuit.bits)
    events = [record for record in circuit if record.opcode in {"R", "HMR"}]
    assert len(events) == 92
    outcomes = tuple(
        0 if mode == 0 else 1 if mode == 1 else (index + mode) % 2
        for index in range(92)
    )
    count = 64
    mask = (1 << count) - 1
    packed = PackedBasisState(
        defaultdict(int, {8 + wire: lane_mask(wire, count) for wire in range(5)}),
        defaultdict(int),
        [0] * count,
    )
    packed.qubits[13] = lane_mask(5, count)
    branches = iter(outcomes)
    amplitudes = [1.0] * count
    for record in circuit:
        if record.opcode in {"R", "HMR"}:
            outcome = next(branches)
            active = (
                mask
                if record.condition_bit is None
                else packed.bits[record.condition_bit]
            )
            for lane in range(count):
                if (active >> lane) & 1:
                    amplitudes[lane] /= 2**0.5
                elif outcome:
                    amplitudes[lane] = 0.0
            draw = mask * outcome
        else:
            draw = 0
        apply_packed_basis_records((record,), packed, iter((draw,)).__next__)
    expected = torch.zeros_like(state)
    for control, value, label in addresses:
        result = (16 * value) % 21 if control else value
        destination = control | sum(
            ((result >> wire) & 1) << (5 - wire) for wire in range(5)
        )
        expected[destination] = state[label] * amplitudes[value + 32 * control]
    actual = forced_branch(state.clone(), gates, outcomes)
    torch.testing.assert_close(actual, expected, atol=1e-20, rtol=1e-11)


def test_modular_asset_entire_conditional_channel_all_valid_columns_and_histories() -> (
    None
):
    from tests.unit.measurement_checks import expand_reset_branches

    circuit = asset("mul16_mod_21.kmx")
    gates = expand_reset_branches(instructions(circuit), first_slot=circuit.bits)
    mapping = {}
    for control, value in product(range(2), range(21)):
        result = (16 * value) % 21 if control else value
        before = tuple((value >> wire) & 1 for wire in range(5)) + (control,)
        after = tuple((result >> wire) & 1 for wire in range(5)) + (control,)
        mapping[before] = after
    proof = certify_conditional_erasure_map(
        gates,
        n_wires=circuit.qubits,
        logical_wires=tuple(range(8, 14)),
        mapping=mapping,
    )
    assert proof["effects"] == 92 and proof["conditional_effects"] == 32
    assert proof["logical_columns"] == 42
    # Independently derive the history count and normalization from the raw
    # source effect guards, without using the Boolean graph's counting routine.
    effects = {"R", "HMR"}
    activity = Counter(
        record.condition_bit for record in circuit if record.opcode in effects
    )
    assert activity == {None: 60, 27: 8, 28: 16, 29: 8}
    inverted: dict[int, bool | None] = {27: None, 28: None, 29: None}
    for record in circuit:
        assert record.opcode not in {"PUSH_CONDITION", "POP_CONDITION"}
        slot = record.result_bit
        if slot is not None and slot in inverted:
            if record.opcode == "HMR":
                assert record.condition_bit is None and inverted[slot] is None
                inverted[slot] = False
            elif record.opcode == "BIT_INVERT":
                assert record.condition_bit is None and inverted[slot] is not None
                inverted[slot] = not inverted[slot]
            else:
                raise AssertionError("Unexpected source selector overwrite")
        condition = record.condition_bit
        if record.opcode in effects and condition is not None:
            assert inverted[condition] is False
    groups = tuple(activity[slot] for slot in inverted)
    histories = 1 << (activity[None] - len(groups))
    for size in groups:
        histories *= 1 + (1 << size)
    assert proof["allowed_histories"] == histories
    probability = Fraction(0)
    for choices in product((0, 1), repeat=len(groups)):
        selected = sum(size * bit for size, bit in zip(groups, choices, strict=True))
        probability += Fraction(
            1 << (activity[None] - len(groups) + selected),
            1 << (activity[None] + selected),
        )
    assert probability == 1
    # The proof must detect a genuinely wrong complex channel, rather than
    # deriving cleanup from the assumed arithmetic mapping.
    missing = next(index for index, gate in enumerate(gates) if gate.name == "cz")
    with pytest.raises(AssertionError, match="complex branch phase"):
        certify_conditional_erasure_map(
            (*gates[:missing], *gates[missing + 1 :]),
            n_wires=circuit.qubits,
            logical_wires=tuple(range(8, 14)),
            mapping=mapping,
        )


def test_complete_modular_asset_native_append_trajectory_has_the_same_complex_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    circuit = asset("mul16_mod_21.kmx")
    state = torch.zeros(1 << circuit.qubits, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    coefficients = coherent_state(6)
    for control, value in product(range(2), range(21)):
        before = control | sum(((value >> wire) & 1) << (5 - wire) for wire in range(5))
        result = 16 * value % 21 if control else value
        after = control | sum(((result >> wire) & 1) << (5 - wire) for wire in range(5))
        state[before] = coefficients[before]
        expected[after] = coefficients[before]
    scale = state.norm()
    state /= scale
    expected /= scale
    # The algebraic certificate supplies exhaustive history coverage. This
    # verifies the unchanged whole-program append/runtime path independently.
    with monkeypatch.context() as patch:
        for sdk in ("qualtran", "cirq", "qiskit", "kickmix"):
            patch.setitem(sys.modules, sdk, None)
        executable = DynamicCircuit(circuit.qubits, inputs=state, dtype=state.dtype)
        executable.append(FieldOperation(circuit.qubits, instructions(circuit)))
        execution = run_dynamic(executable, shots=3, seed=67, strategy="trajectory")
    torch.testing.assert_close(
        execution.final_states.reshape(3, -1),
        expected.expand(3, -1),
        atol=1e-11,
        rtol=0,
    )
    assert torch.count_nonzero(execution.final_states.reshape(3, -1)[:, 64:]) == 0


def test_source_records_and_bound_primitive_counts_are_separate() -> None:
    records: object = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("small_circuit_assets.json")
        .read_text()
    )
    assert isinstance(records, list)
    for row in records:
        circuit = asset(row["name"])
        expected: Counter[str] = Counter(classical_xor=circuit.bits)
        for record in circuit:
            code = record.opcode
            if code == "HMR":
                expected.update(("h", "measure", "x"))
            elif code == "R":
                expected.update(("h", "reset"))
            elif code == "NEG":
                expected.update(("x", "z", "x", "z"))
            elif code == "CCZ":
                expected.update(("h", "ccx", "h"))
            elif code == "Z_POW":
                expected.update(("phase",))
            elif code == "PUSH_CONDITION" or code.startswith("BIT_"):
                expected.update(("classical_xor",))
            elif code != "POP_CONDITION":
                expected.update((code.lower(),))
        assert Counter(gate.name for gate in instructions(circuit)) == expected, row[
            "name"
        ]
        assert Counter(record.opcode for record in circuit) == row["counts"]


def test_initially_zero_reset_each_kraus_factor() -> None:
    from flagquantum.compiler.registers.primitives.measurement import (
        reset_x_register_ir,
    )
    from tests.unit.measurement_checks import expand_reset_branches

    state = torch.tensor([0.6 + 0.8j, 0], dtype=torch.complex128)
    gates = expand_reset_branches(reset_x_register_ir((0,)), first_slot=0)
    for outcome in (0, 1):
        actual = forced_branch(state.clone(), gates, (outcome,))
        torch.testing.assert_close(actual, state / 2**0.5, atol=1e-12, rtol=0)
