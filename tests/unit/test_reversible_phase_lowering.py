"""Check the first reversible-phase lowering slice through Circuit.append."""

from dataclasses import dataclass
from fractions import Fraction
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType

import pytest

from flagquantum.circuit import Circuit
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.qrom import controlled_qrom_ir
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import CapabilityError, ValidationError

pytestmark = pytest.mark.unit


@pytest.fixture
def native_reference() -> ModuleType:
    km = pytest.importorskip("kickmix")
    try:
        installed = version("kickmix")
    except PackageNotFoundError:
        installed = None  # A reference extension built in place.
    if installed is not None and installed != "0.1.dev0":
        pytest.skip("Native comparison pins the 0.1.dev0 reference source")
    if installed is None and getattr(km, "__version__", None) != "0.1":
        pytest.skip("Native comparison needs the in-place 0.1 reference build")
    return km


@dataclass(frozen=True)
class _LookupAndErase:
    address_width: int
    words: tuple[int, ...]

    def append_to(self, circuit: Circuit) -> Circuit:
        for erase in (False, True):
            circuit.compose(
                controlled_qrom_ir(
                    self.address_width,
                    8,
                    self.words,
                    blocked=True,
                    inverse=erase,
                    measurement_bits=tuple(range(16)) if erase else (),
                )
            )
        return circuit


def _paired_circuit(width: int, words: tuple[int, ...]) -> CircuitIR:
    return DynamicCircuit(17 + 2 * width).append(_LookupAndErase(width, words)).to_ir()


def test_paired_lookup_and_erasure_lower_to_hmr_with_feedback() -> None:
    ir = _paired_circuit(2, (8, 20))
    lowered = lower_reversible_phase(ir)
    measured = [op for op in lowered.operations if op.opcode == "hmr"]
    assert lowered.n_wires == ir.n_wires == 21
    assert lowered.n_bits == 16
    assert [(op.wires, op.result_bit) for op in measured] == [
        ((wire,), bit) for bit, wire in enumerate(range(3, 19))
    ]
    assert len(ir.instructions) - len(lowered.operations) == 32
    text = encode_kmx(lowered)
    assert "HMR q3 b0\n" in text
    assert " if b" in text


def test_standalone_h_and_unwritten_condition_fail_closed() -> None:
    with pytest.raises(CapabilityError, match="instruction 0"):
        lower_reversible_phase(Circuit(1).h(0).to_ir())
    unwritten = Instruction("x", (0,), metadata={"conditions": ((3, 1),)})
    with pytest.raises(ValidationError, match="read before measurement"):
        lower_reversible_phase(CircuitIR(1, (unwritten,)))


def test_unrelated_feedback_is_not_fused_with_x_basis_measurement() -> None:
    ir = CircuitIR(
        2,
        (
            Instruction("h", (0,)),
            Instruction(
                "measure", (0,), metadata={"is_dynamic": True, "classical_bit": 0}
            ),
            Instruction("x", (1,), metadata={"conditions": ((0, 1),)}),
        ),
    )
    with pytest.raises(CapabilityError, match="instruction 0"):
        lower_reversible_phase(ir)


@pytest.mark.parametrize(
    "width,words",
    [
        (3, (8, 20, 0, 0, 0, 240)),
        (3, (248, 236, 0, 0, 0, 16)),
        (2, (8, 20)),
        (2, (248, 236)),
    ],
)
def test_paired_lookup_and_erasure_against_native_reference(
    width: int, words: tuple[int, ...], native_reference: ModuleType
) -> None:
    km = native_reference
    program = lower_reversible_phase(_paired_circuit(width, words))
    reference = km.Circuit(encode_kmx(program))
    n_input = width + 1
    common_measurement = None
    for prefix in range(1 << n_input):
        sim = km.Simulator(batch_size=64, seed=123)
        all_shots = (1 << 64) - 1
        expected = [
            all_shots if prefix & (1 << (n_input - wire - 1)) else 0
            for wire in range(n_input)
        ]
        for wire, value in enumerate(expected):
            sim.write_across_shots(km.q(wire), value)
        sim.do(reference)
        assert [
            sim.read_across_shots(km.q(wire), out=int)
            for wire in range(program.n_wires)
        ] == expected + [0] * (program.n_wires - n_input)
        assert all(sim.read_shot_phase(shot) == Fraction(0) for shot in range(64))
        measurements = tuple(
            sim.read_across_shots(km.b(bit), out=int) for bit in range(16)
        )
        if common_measurement is None:
            assert all(0 < sample < all_shots for sample in measurements)
            common_measurement = measurements
        assert measurements == common_measurement


@pytest.mark.parametrize(
    "turns", [Fraction(1, 128), Fraction(-1, 128), Fraction(1, 2**127)]
)
def test_exact_half_turns_round_trip(
    turns: Fraction, tmp_path: Path, native_reference: ModuleType
) -> None:
    km = native_reference
    b = km.CircuitBuilder()
    b.z_pow(km.q(0), turns)
    path = str(tmp_path / "phase.kmx")
    b.finish_circuit().to_file(path)
    sim = km.Simulator(batch_size=1, seed=1)
    sim.do(km.Circuit("X q0"))
    sim.do(km.Circuit.from_file(path))
    assert sim.read_shot_phase(0) == turns % 2


def test_reversible_gates_hmr_and_feedback(native_reference: ModuleType) -> None:
    km = native_reference
    circuit = km.Circuit("""
        X q0
        CX q0 q1
        CCX q0 q1 q2
        Z_POW q2 0.0078125
        HMR q2 b0
        Z q1 if b0
    """)
    sim = km.Simulator(batch_size=64, seed=123)
    sim.do(circuit)
    outcomes = sim.read_across_shots(km.b(0), out=int)
    assert 0 < outcomes < 2**64 - 1
    assert sim.read_across_shots(km.q(2), out=int) == 0
    assert sim.read_across_shots(km.q(0), out=int) == 2**64 - 1
    assert sim.read_across_shots(km.q(1), out=int) == 2**64 - 1
    assert all(sim.read_shot_phase(i) == Fraction(1, 128) for i in range(64))


def test_native_parser_rejects_standalone_h(native_reference: ModuleType) -> None:
    km = native_reference
    with pytest.raises(
        (ValueError, RuntimeError),
        match=r"(?i)(operation|instruction|opcode|unrecognized|unknown|invalid)",
    ):
        km.Circuit("H q0")


@pytest.mark.parametrize("turns", [Fraction(1, 3), Fraction(1, 2**128)])
def test_native_rounds_off_grid_phases(
    turns: Fraction, native_reference: ModuleType
) -> None:
    km = native_reference
    b = km.CircuitBuilder()
    b.z_pow(km.q(0), turns)
    sim = km.Simulator(batch_size=1, seed=1)
    sim.do(km.Circuit("X q0"))
    sim.do(b.finish_circuit())
    grid = 2**127
    observed = sim.read_shot_phase(0)
    assert observed == Fraction(round(turns * grid), grid)
    assert observed != turns
