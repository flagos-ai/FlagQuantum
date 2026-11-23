"""Compare persistent simulation ownership with unchanged physical source."""

import json
from copy import copy, deepcopy
from pathlib import Path

import pytest

from flagquantum.compiler.artifacts.phase_angle import PhaseAngle128
from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.core._fixed_width_integer import FixedWidthInteger
from flagquantum.simulation._basis_simulation import PackedBasisSimulation
from flagquantum.simulation.physical_basis_rng import _mt19937_64
from flagquantum.simulation.physical_basis_shots import BasisRecord

pytestmark = pytest.mark.unit
_DATA = json.loads(
    (Path(__file__).with_name("data") / "physical_basis_lifecycle.json").read_text()
)


def _word(value: int, width: int) -> str:
    return "".join("1" if (value >> bit) & 1 else "_" for bit in range(width))


def _phase(sim: PackedBasisSimulation) -> str:
    return str(PhaseAngle128._from_units(sim.read_shot_phase(0)))


def _count(sim: PackedBasisSimulation, name: str) -> int:
    return dict(sim.counters.snapshot(0)[0]).get(name, 0)


def _copies(width: int) -> list[str]:
    sim = PackedBasisSimulation(width, _mt19937_64(123), counting=True)
    circuit = read_kmx_circuit("X q0")
    sim.configure_for(circuit)
    sim.apply(circuit)
    cloned, assigned = copy(sim), deepcopy(sim)
    result = [
        f"copy:{width}:{int(next(sim.rng) == next(cloned.rng))}:{int(sim.next_measurement() == cloned.next_measurement())}:{int(assigned.state.qubits[0] == sim.state.qubits[0])}"
    ]
    cloned.apply(circuit)
    result.append(
        f"isolated:{int(cloned.state.qubits[0] != sim.state.qubits[0])}:{_count(sim, 'X')}:{_count(cloned, 'X')}"
    )

    def throwing(record: BasisRecord, value: int, active: int) -> None:
        raise RuntimeError("stop")

    sim.debug_handler = throwing
    with pytest.raises(RuntimeError, match="stop"):
        sim.apply(read_kmx_circuit("NEG\nDEBUG_PRINT\nX q0"))
    result.append(
        f"throw:{_phase(sim)}:{_count(sim, 'NEG')}:{_count(sim, 'DEBUG_PRINT_EMPTY')}:{_count(sim, 'X')}"
    )
    result.append(_condition_callback_trace(width))
    return result


def _lifecycle(width: int) -> list[str]:
    mask = (1 << width) - 1
    sim = PackedBasisSimulation(width, _mt19937_64(123), counting=True)
    circuit = read_kmx_circuit("PUSH_CONDITION if b0\nX q0")
    sim.configure_for(circuit)
    alternating = int("aaaaaaaa" * (width // 32), 16)
    sim.state.bits[0] = alternating
    sim.apply(circuit)
    sim.apply(read_kmx_circuit("X q0"))
    result = [
        f"width:{width}",
        f"retained:{_word(sim.state.qubits[0], width)}:{len(sim.state.conditions)}",
    ]
    sim.apply(read_kmx_circuit("POP_CONDITION\nX q0"))
    result.append(
        f"popped:{_word(sim.state.qubits[0], width)}:{len(sim.state.conditions)}"
    )
    sim.state.sign = mask ^ alternating
    sim.state.phases[0] = PhaseAngle128.from_half_turns_exact_double(0.125).units
    for qubits, bits, label in ((3, 0, "grown"), (0, 3, "grown2")):
        sim.ensure_big_enough_state_for(qubits, bits)
        result.append(
            f"{label}:{sim.qubits}:{sim.bits}:{_word(sim.state.sign, width)}:{_word(sim.state.qubits[0], width)}:{_word(sim.state.bits[0], width)}:{_phase(sim)}"
        )
    circuit = read_kmx_circuit(
        "DEBUG_PRINT q0 if b0\nNEG\nDEBUG_PRINT\nZ_POW q0 0.25\nDEBUG_PRINT"
    )
    sim.configure_for(circuit)
    sim.state.qubits[0] = mask
    calls = 0

    def debug(record: BasisRecord, value: int, active: int) -> None:
        nonlocal calls
        calls += 1
        target = f"q{record.qubits[0]}" if record.qubits else "False"
        predicate = (
            "true" if record.condition_bit is None else f"b{record.condition_bit}"
        )
        result.append(
            f"debug:{calls}:{target}:{_word(value, width)}:{predicate}:{_word(active, width)}:{_phase(sim)}"
        )

    sim.debug_handler = debug
    sim.apply(circuit)
    result.append(f"phase:{_phase(sim)}")
    sim.clear_for_shot()
    result.append(
        f"cleared:{_word(sim.state.sign, width)}:{_word(sim.state.qubits[0], width)}:{_phase(sim)}:{len(sim.state.conditions)}:{_count(sim, 'X')}"
    )
    sim.clear_operation_statistics()
    result.append(f"counts_clear:{_count(sim, 'X')}")
    return result


def test_original_four_width_lifecycle_and_copy_trace() -> None:
    lines = [line for width in (64, 128, 256, 512) for line in _copies(width)]
    lines.extend(line for width in (64, 128, 256, 512) for line in _lifecycle(width))
    assert "\n".join(lines) + "\n" == _DATA["observations"]


@pytest.mark.parametrize("width", (64, 128, 256, 512))
def test_buffer_lifecycle_single_register_and_snapshot_isolation(width: int) -> None:
    circuit = read_kmx_circuit(
        "APPEND_TO_REGISTER q0 r0\nAPPEND_TO_REGISTER b0 r0\nAPPEND_TO_REGISTER q1 r1\nX q1"
    )
    sim = PackedBasisSimulation(width, _mt19937_64(1), counting=True)
    sim.configure_for(circuit)
    for lane, row in enumerate(sim.register_buffer):
        row[0] = FixedWidthInteger(2, lane % 4)
        row[1] = FixedWidthInteger(1, lane % 2)
    before = deepcopy(sim.register_buffer)
    sim.copy_register_buffer_into_state(0)
    assert sim.state.qubits.get(1, 0) == 0
    sim.copy_register_buffer_into_state()
    cloned = copy(sim)
    sim.apply(circuit)
    sim.copy_state_into_register_buffer(1)
    assert [row[0] for row in sim.register_buffer] == [row[0] for row in before]
    assert [row[1].value for row in sim.register_buffer] == [
        1 - lane % 2 for lane in range(width)
    ]
    sim.copy_state_into_register_buffer()
    assert cloned.register_buffer == before
    assert cloned.state.qubits != sim.state.qubits
    sim.state.conditions[:] = [0]
    sim.clear_for_shot()
    assert sim.register_buffer != before and not sim.state.conditions
    assert _count(sim, "X") == 1
    sim.ensure_big_enough_state_for(3, 2)
    assert (sim.qubits, sim.bits) == (4, 2)
    for kind in ("q", "b"):
        assert sim.safe_read(kind, -1) == sim.safe_read(kind, 100) == 0
    assert sim.safe_read("bool", 1) == (1 << width) - 1
    assert sim.safe_read("bool", 0) == 0
    for lane in (-1, width):
        with pytest.raises(IndexError, match="lane"):
            sim.read_shot_phase(lane)
    for size in (-1, True):
        with pytest.raises(ValueError, match="dimensions"):
            sim.ensure_big_enough_state_for(size, 0)
    stream = next(sim.rng)
    sim.configure_for(circuit)
    assert all(cell.value == 0 for row in sim.register_buffer for cell in row)
    assert _count(sim, "X") == 1 and next(sim.rng) != stream


def test_simulator_rejects_unsupported_width() -> None:
    with pytest.raises(ValueError, match="width"):
        PackedBasisSimulation(32, _mt19937_64(0))


def test_debug_classical_target_inactive_mask_and_copied_callback_owner() -> None:
    sim = PackedBasisSimulation(64, _mt19937_64(0), counting=True)
    circuit = read_kmx_circuit("DEBUG_PRINT b0 if b1\nDEBUG_PRINT b0")
    sim.configure_for(circuit)
    sim.state.bits[0] = 5
    events: list[tuple[int, int, int | None, int]] = []

    def debug(record: BasisRecord, value: int, active: int) -> None:
        events.append((sim.read_shot_phase(0), value, record.condition_bit, active))

    sim.debug_handler = debug
    clone = copy(sim)
    clone.state.sign = (1 << 64) - 1
    clone.apply(circuit)
    assert events == [(0, 5, 1, 0), (0, 5, None, (1 << 64) - 1)]
    assert _count(clone, "DEBUG_PRINT_C_IF") == 0
    assert _count(clone, "DEBUG_PRINT_C") == 1
    assert _count(sim, "DEBUG_PRINT_C") == 0
    assert sim.safe_read("b", 0) == 5
    assert sim.safe_read("q", 0) == 0


def _condition_callback_trace(width: int) -> str:
    sim = PackedBasisSimulation(width, _mt19937_64(0))
    circuit = read_kmx_circuit("DEBUG_PRINT\nX q0")
    sim.configure_for(circuit)

    def suppress_next_scan(record: BasisRecord, value: int, active: int) -> None:
        sim.state.conditions[:] = [0]

    sim.debug_handler = suppress_next_scan
    sim.apply(circuit)
    values = [_word(sim.state.qubits[0], width)]
    sim.apply(read_kmx_circuit("X q0"))
    values.append(_word(sim.state.qubits[0], width))
    sim.apply(read_kmx_circuit("POP_CONDITION\nX q0"))
    values.append(_word(sim.state.qubits[0], width))
    return "cached:" + ":".join(values)
