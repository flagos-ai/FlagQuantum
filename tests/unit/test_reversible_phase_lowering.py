"""Check the first reversible-phase lowering slice through Circuit.append."""

from collections import Counter
from dataclasses import dataclass, replace
from fractions import Fraction
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType

import pytest
import torch

from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import summarize_components
from flagquantum.compiler._component_trace.trace import ComponentTrace
from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseProgram
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.qrom import controlled_qrom_ir
from flagquantum.core.classical_control import instruction_condition_clauses
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import CapabilityError, ValidationError
from flagquantum.runtime.dynamic.execution import run_dynamic

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
    trace: ComponentTrace | None = None

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
                    trace=self.trace,
                    trace_offset=(
                        len(circuit.to_ir().instructions)
                        if self.trace is not None
                        else 0
                    ),
                )
            )
        return circuit


def _paired_circuit(
    width: int, words: tuple[int, ...], trace: ComponentTrace | None = None
) -> CircuitIR:
    return (
        DynamicCircuit(17 + 2 * width)
        .append(_LookupAndErase(width, words, trace))
        .to_ir()
    )


def test_paired_lookup_and_erasure_lower_to_hmr_with_feedback() -> None:
    ir = _paired_circuit(2, (8, 20))
    lowered = lower_reversible_phase(ir)
    measured = [op for op in lowered.operations if op.opcode == "hmr"]
    assert lowered.n_qubits == ir.n_wires == 21
    assert lowered.n_bits == 16
    assert [(op.qubits, op.result_bit) for op in measured] == [
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


ANGLE_TABLES = [
    (3, (8, 20, 0, 0, 0, 240)),
    (3, (248, 236, 0, 0, 0, 16)),
    (2, (8, 20)),
    (2, (248, 236)),
]


@pytest.mark.parametrize("width,words", ANGLE_TABLES)
def test_trace_counts_and_kmx_round_trip(width: int, words: tuple[int, ...]) -> None:
    trace = ComponentTrace()
    ir = _paired_circuit(width, words, trace)
    assert ir.instructions == _paired_circuit(width, words).instructions
    assert all(root.parameters[2] == words for root in trace.calls)
    edges, attributed = summarize_components(ir, trace)
    assert edges == {
        ("Circuit", "PairedLookup"): 1,
        ("Circuit", "PairedErasure"): 1,
        ("PairedLookup", "AddressTraversal"): 1,
        ("PairedLookup", "BankExchange"): 1,
        ("PairedErasure", "BankMeasurement"): 1,
        ("PairedErasure", "AddressTraversal"): 1,
    }
    counts: Counter[tuple[str, bool]] = Counter()
    for (_, name, conditional), number in attributed.items():
        counts[name, conditional] += number
    assert counts == Counter(
        (op.name, bool(instruction_condition_clauses(op))) for op in ir
    )
    assert counts["measure", False] == counts["x", True] == 16
    program = lower_reversible_phase(ir)
    # The highest scratch slot can be idle, and the format must preserve it.
    enlarged = replace(
        program, n_qubits=program.n_qubits + 1, n_bits=program.n_bits + 2
    )
    assert decode_kmx(encode_kmx(enlarged)) == enlarged
    assert Counter(op.opcode for op in program.operations)["hmr"] == 16


@pytest.mark.parametrize("payload", ["H q0", "CX q0 q0", "HMR q0 b2", "Z q0 if b0"])
def test_kmx_decode_rejects_unsupported_or_invalid_stream(payload: str) -> None:
    header = 'INFO_TAG "@flagquantum:reversible-phase-v1 wires=2 bits=1"'
    with pytest.raises(ValueError):
        decode_kmx(f"{header}\n{payload}\n")


@pytest.mark.parametrize("width,words", ANGLE_TABLES)
def test_paired_lookup_and_erasure_against_native_reference(
    width: int, words: tuple[int, ...], native_reference: ModuleType, tmp_path: Path
) -> None:
    np = pytest.importorskip("numpy")
    km = native_reference
    program = lower_reversible_phase(_paired_circuit(width, words))
    path = tmp_path / "lookup.kmx"
    path.write_text(encode_kmx(program))
    reference = km.Circuit.from_file(str(path))
    reference.to_file(str(path))
    assert decode_kmx(path.read_text()) == program
    n_input = width + 1
    n_shots = 1 << 20
    all_shots = (1 << n_shots) - 1
    common_measurement = None
    for prefix in range(1 << n_input):
        sim = km.Simulator(batch_size=n_shots, seed=123)
        expected = [
            all_shots if prefix & (1 << (n_input - wire - 1)) else 0
            for wire in range(n_input)
        ]
        for wire, value in enumerate(expected):
            sim.write_across_shots(km.q(wire), value)
        sim.do(reference)
        assert [
            sim.read_across_shots(km.q(wire), out=int)
            for wire in range(program.n_qubits)
        ] == expected + [0] * (program.n_qubits - n_input)
        # The target subset has only integral half-turn phases.
        assert sim.read_phase_flipped_across_shots(out=int) == 0
        measurements = tuple(
            sim.read_across_shots(km.b(bit), out=int) for bit in range(16)
        )
        if common_measurement is None:
            codes = np.zeros(n_shots, dtype=np.uint16)
            for bit, packed in enumerate(measurements):
                row = np.unpackbits(
                    np.frombuffer(
                        packed.to_bytes(n_shots // 8, "little"), dtype=np.uint8
                    ),
                    bitorder="little",
                )
                codes |= row.astype(np.uint16) << bit
            np.testing.assert_array_equal(np.unique(codes), np.arange(1 << 16))
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
    payload = """
        X q0
        CX q0 q1
        CCX q0 q1 q2
        Z_POW q2 0.0078125
        HMR q2 b0
        Z q1 if b0
    """
    circuit = km.Circuit(payload)
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


def _expand_for_dynamic_reference(program: ReversiblePhaseProgram) -> CircuitIR:
    instructions = []
    for op in program.operations:
        if op.opcode == "hmr":
            instructions.extend(
                (
                    Instruction("h", op.qubits),
                    Instruction(
                        "measure",
                        op.qubits,
                        metadata={"is_dynamic": True, "classical_bit": op.result_bit},
                    ),
                    Instruction(
                        "x", op.qubits, metadata={"conditions": ((op.result_bit, 1),)}
                    ),
                )
            )
        else:
            metadata = (
                {"conditions": ((op.condition_bit, 1),)}
                if op.condition_bit is not None
                else {}
            )
            instructions.append(Instruction(op.opcode, op.qubits, metadata=metadata))
    return CircuitIR(program.n_qubits, tuple(instructions))


@pytest.mark.parametrize("width,words", ANGLE_TABLES)
def test_reconstructed_ir_preserves_all_complex_measurement_branches(
    width: int, words: tuple[int, ...]
) -> None:
    np = pytest.importorskip("numpy")
    ir = _paired_circuit(width, words)
    program = decode_kmx(encode_kmx(lower_reversible_phase(ir)))
    reference = _expand_for_dynamic_reference(program)
    assert reference.instructions == ir.instructions
    outcomes = np.arange(1 << 16, dtype=np.uint32)
    for prefix in range(1 << (width + 1)):
        state = [(prefix >> shift) & 1 for shift in range(width, -1, -1)]
        state += [0] * (program.n_qubits - len(state))
        # Different complex phases expose address-dependent phase errors.
        amplitude = (1j) ** prefix / (1 << (width + 1)) ** 0.5
        actual = np.full(outcomes.size, amplitude, dtype=np.complex128)
        for instruction in reference:
            name = instruction.name
            *controls, target = instruction.wires
            clauses = instruction_condition_clauses(instruction)
            if name in {"x", "cx", "ccx"} and not clauses:
                state[target] ^= all(state[q] for q in controls)
            elif name == "cswap":
                control, left, right = instruction.wires
                if state[control]:
                    state[left], state[right] = state[right], state[left]
            elif name == "h":
                # For a computational basis input, project H|x> using
                # <m|H|x> = (-1)**(x*m) / sqrt(2) at its following measure.
                continue
            elif name == "measure":
                bit = instruction.metadata["classical_bit"]
                if state[target]:
                    actual[(outcomes & (1 << bit)) != 0] *= -1
                actual *= 2**-0.5
            elif name == "x" and clauses:
                # The outcome-bit controlled X clears its measured wire.
                state[target] = 0
            else:
                assert name in {"z", "cz"}
                if all(state[q] for q in instruction.wires):
                    (((bit, expected),),) = clauses
                    actual[((outcomes >> bit) & 1) == expected] *= -1
        assert state == [(prefix >> shift) & 1 for shift in range(width, -1, -1)] + [
            0
        ] * (program.n_qubits - width - 1)
        np.testing.assert_allclose(actual, amplitude / 256, rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            abs(actual / amplitude) ** 2, 1 / (1 << 16), rtol=0, atol=1e-16
        )


def test_same_artifact_can_switch_between_dynamic_and_native_execution(
    native_reference: ModuleType,
) -> None:
    km = native_reference
    # One address bit and two one-bit banks fit the dense reference simulator.
    circuit = DynamicCircuit(5)
    for erase in (False, True):
        circuit.compose(
            controlled_qrom_ir(
                1,
                1,
                (1, 0),
                blocked=True,
                inverse=erase,
                measurement_bits=(0, 1) if erase else (),
            )
        )
    artifact = decode_kmx(encode_kmx(lower_reversible_phase(circuit.to_ir())))
    prefix = torch.zeros(1 << artifact.n_qubits, dtype=torch.complex128)
    prefix[::8] = torch.tensor([1, 1j, -1, -1j], dtype=torch.complex128) / 2
    dynamic = DynamicCircuit(artifact.n_qubits, dtype=torch.complex128, inputs=prefix)
    dynamic.compose(_expand_for_dynamic_reference(artifact))
    result = run_dynamic(dynamic, shots=64, seed=123, strategy="trajectory")
    assert len(set(map(tuple, result.classical_bits.tolist()))) == 4
    torch.testing.assert_close(
        result.final_states.reshape(64, -1), prefix.expand(64, -1), rtol=0, atol=1e-12
    )

    native = km.Circuit(encode_kmx(artifact))
    for value in range(4):
        sim = km.Simulator(batch_size=64, seed=123)
        for wire, shift in enumerate((1, 0)):
            sim.write_across_shots(km.q(wire), ((1 << 64) - 1) * ((value >> shift) & 1))
        sim.do(native)
        assert sim.read_phase_flipped_across_shots(out=int) == 0
        assert [sim.read_across_shots(km.q(wire), out=int) for wire in range(5)] == [
            ((1 << 64) - 1) * ((value >> 1) & 1),
            ((1 << 64) - 1) * (value & 1),
            0,
            0,
            0,
        ]


@pytest.mark.parametrize("width,words", ANGLE_TABLES)
def test_native_forward_banks_match_upstream_lookup(
    width: int, words: tuple[int, ...], native_reference: ModuleType
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("qualtran")
    if version("qualtran") != "0.7.0":
        pytest.skip("Lookup parity pins Qualtran 0.7.0")
    from qualtran.bloqs.data_loading import QROAMClean

    km = native_reference
    padded = (*words, *([0] * ((1 << width) - len(words))))
    upstream = QROAMClean.build_from_data(
        np.asarray(padded), target_bitsizes=(8,), num_controls=1, log_block_sizes=(1,)
    )
    upstream_decomposition = upstream.decompose_bloq()
    ir = controlled_qrom_ir(width, 8, words, blocked=True)
    artifact = decode_kmx(encode_kmx(lower_reversible_phase(ir)))
    circuit = km.Circuit(encode_kmx(artifact))
    for enable in (0, 1):
        for address in range(1 << width):
            sim = km.Simulator(batch_size=1, seed=123)
            sim.write_across_shots(km.q(0), enable)
            for wire in range(1, width + 1):
                sim.write_across_shots(km.q(wire), (address >> (width - wire)) & 1)
            sim.do(circuit)
            expected = upstream_decomposition.on_classical_vals(
                control=enable, selection=address
            )
            banks = (int(expected["target0_"]), int(expected["junk_target0_"][0]))
            payload = [
                (word >> shift) & 1 for word in banks for shift in range(7, -1, -1)
            ]
            assert [
                sim.read_across_shots(km.q(wire), out=int)
                for wire in range(width + 1, artifact.n_qubits)
            ] == payload + [0] * width


def test_trace_offsets_leave_surrounding_instructions_at_circuit_level() -> None:
    trace = ComponentTrace()
    circuit = DynamicCircuit(21)
    circuit.x(0).append(_LookupAndErase(2, (8, 20), trace)).z(0)
    ir = circuit.to_ir()
    assert trace.calls[0].start == 1
    assert trace.calls[-1].stop == len(ir.instructions) - 1
    _, counts = summarize_components(ir, trace)
    assert counts["Circuit", "x", False] == counts["Circuit", "z", False] == 1
