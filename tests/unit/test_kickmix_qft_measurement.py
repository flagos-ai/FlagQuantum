"""Verify every coherent Fourier-measurement branch and the locked source trace."""

from collections import Counter
from fractions import Fraction
from math import pi
from random import Random

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.kickmix_qft_measurement import (
    kickmix_qft_measurement_operations,
    kickmix_qft_measurement_program,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch

pytestmark = pytest.mark.unit


def _address(value: int, ports: tuple[int, ...], width: int) -> int:
    return sum(
        ((value >> bit) & 1) << (width - 1 - wire)
        for bit, wire in enumerate(reversed(ports))
    )


def _state(
    width: int, target: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    spectators = tuple(wire for wire in range(width) if wire not in target)
    generator = torch.Generator().manual_seed(119)
    shape = (1 << len(target), 1 << len(spectators))
    psi = torch.complex(
        torch.randn(shape, generator=generator, dtype=torch.float64),
        torch.randn(shape, generator=generator, dtype=torch.float64),
    )
    psi /= psi.norm()
    state = torch.zeros(1 << width, dtype=torch.complex128)
    for x in range(shape[0]):
        for s in range(shape[1]):
            state[_address(x, target, width) | _address(s, spectators, width)] = psi[
                x, s
            ]
    return state, psi, spectators


def _branches(psi: torch.Tensor, inverse: bool) -> torch.Tensor:
    dimension = psi.shape[0]
    labels = torch.arange(dimension, dtype=torch.float64)
    angles = (
        (-1 if inverse else 1) * 2 * pi / dimension * labels[:, None] * labels[None, :]
    )
    matrix = torch.exp(1j * angles) / dimension**0.5
    return matrix @ psi


@pytest.mark.parametrize("size", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("permuted", [False, True])
def test_every_unnormalized_complex_branch_matches_the_dense_fourier_projection(
    size: int, inverse: bool, permuted: bool
) -> None:
    if permuted and size:
        width = 2 * size
        target = tuple(range(0, width, 2))
        target = (*target[-1:], *target[:-1])
    else:
        width = size + 1
        target = tuple(range(size))
    outputs = tuple(3 + 2 * k for k in range(size))
    program = kickmix_qft_measurement_program(
        target, outputs, n_qubits=width, inverse_qft=inverse
    )
    assert decode_kmx(encode_kmx(program)) == program
    state, psi, spectators = _state(width, target)
    expected_branches = _branches(psi, inverse)
    gates = bind_reversible_phase_execution(program)
    total_probability = 0.0
    for y in range(1 << size):
        actual = forced_branch(
            state.clone(), gates, tuple((y >> k) & 1 for k in range(size))
        )
        expected = torch.zeros_like(state)
        for s, amplitude in enumerate(expected_branches[y]):
            expected[_address(s, spectators, width)] = amplitude
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
        total_probability += float(actual.abs().square().sum())
    assert total_probability == pytest.approx(1.0, abs=1e-12)
    assert all(wire in target for op in program.operations for wire in op.qubits)


@pytest.mark.parametrize("strategy", ["trajectory", "batched"])
@pytest.mark.parametrize("inverse", [False, True])
def test_local_execution_preserves_classical_holes_and_conditional_spectator_state(
    strategy: str, inverse: bool
) -> None:
    outputs = (2, 4, 6)
    program = kickmix_qft_measurement_program(
        (0, 1, 2), outputs, n_qubits=4, inverse_qft=inverse
    )
    assert program.input_bits == (0, 1, 3, 5)
    setup = tuple(
        ClassicalXor((), bit, constant=bool(bit % 2)).instruction()
        for bit in program.input_bits
    )
    state, psi, _ = _state(4, (0, 1, 2))
    branches = _branches(psi, inverse)
    probabilities = branches.abs().square().sum(dim=1)
    circuit = DynamicCircuit(4, inputs=state, dtype=torch.complex128).append(
        FieldOperation(4, (*setup, *bind_reversible_phase_execution(program)))
    )
    result = run_dynamic(circuit, shots=256, seed=41, strategy=strategy)
    rows = result.final_states.reshape(256, -1)
    words = sum(
        result.classical_bits[:, bit].long() << (2 - k) for k, bit in enumerate(outputs)
    )
    for y in range(8):
        selected = rows[words == y]
        if len(selected):
            expected = torch.zeros(16, dtype=torch.complex128)
            expected[:2] = branches[y] / probabilities[y].sqrt()
            torch.testing.assert_close(
                selected, expected.expand(len(selected), -1), atol=1e-12, rtol=0
            )
        assert abs(float((words == y).double().mean()) - float(probabilities[y])) < 0.12
    for bit in program.input_bits:
        assert torch.all(result.classical_bits[:, bit] == bit % 2)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 6, 8, 16, 64, 128])
@pytest.mark.parametrize("inverse", [False, True])
def test_native_basis_phase_is_the_exact_fourier_scalar(
    size: int, inverse: bool
) -> None:
    km = pytest.importorskip("kickmix")
    assert hasattr(km, "Simulator")
    program = kickmix_qft_measurement_program(
        tuple(reversed(range(size))),
        tuple(reversed(range(size))),
        n_qubits=size,
        inverse_qft=inverse,
    )
    sim = km.Simulator(batch_size=64, seed=41)
    rng = Random(119)
    inputs = [
        shot % (1 << size) if size <= 6 else rng.getrandbits(size) for shot in range(64)
    ]
    for q in range(size):
        sim.write_across_shots(
            km.q(q), sum(((x >> q) & 1) << shot for shot, x in enumerate(inputs))
        )
    sim.do(km.Circuit(encode_kmx(program)))
    columns = [sim.read_across_shots(km.b(bit), out=int) for bit in range(size)]
    assert all(sim.read_across_shots(km.q(q), out=int) == 0 for q in range(size))
    for shot, x in enumerate(inputs):
        y = sum(((column >> shot) & 1) << bit for bit, column in enumerate(columns))
        expected = Fraction((-1 if inverse else 1) * x * y, 1 << (size - 1)) % 2
        assert Fraction(sim.read_shot_phase(shot)) == expected


@pytest.mark.parametrize("size", [0, 1, 2, 4, 8, 128, 129, 130])
@pytest.mark.parametrize("inverse", [False, True])
def test_native_opcode_counts_and_fixed_precision_zero_nodes(
    size: int, inverse: bool
) -> None:
    operations = kickmix_qft_measurement_operations(
        tuple(range(size)), tuple(range(size)), inverse_qft=inverse
    )
    counts = Counter(op.opcode for op in operations)
    assert counts["swap"] == size // 2
    assert counts["hmr"] == size
    assert counts["z_pow"] == size * (size - 1) // 2
    zero_count = sum(op.opcode == "z_pow" and not op.angle for op in operations)
    assert zero_count == sum(size - distance for distance in range(128, size))


def test_invalid_fourier_ports_fail_before_emission() -> None:
    with pytest.raises(ValueError, match="matching"):
        kickmix_qft_measurement_operations((0,), ())
    with pytest.raises(ValueError, match="distinct nonnegative"):
        kickmix_qft_measurement_operations((0, 1), (2, 2))
    with pytest.raises(ValueError, match="distinct nonnegative"):
        kickmix_qft_measurement_operations((0,), (-1,))
    with pytest.raises(ValueError, match="boolean"):
        kickmix_qft_measurement_operations((0,), (0,), inverse_qft=1)  # type: ignore[arg-type]
    assert kickmix_qft_measurement_operations((), ()) == ()


def test_source_and_bound_resource_bases_are_separate() -> None:
    from flagquantum.compiler._component_trace.resource_comparison import (
        count_emitted_instructions,
    )
    from flagquantum.core.ir import CircuitIR

    program = kickmix_qft_measurement_program((0, 1, 2, 3), (0, 1, 2, 3), n_qubits=5)
    source = Counter(op.opcode for op in program.operations)
    assert source == {"swap": 2, "z_pow": 6, "hmr": 4}
    report = count_emitted_instructions(
        CircuitIR(5, bind_reversible_phase_execution(program))
    )
    assert report.unconditional == {"classical_xor": 4, "h": 4, "measure": 4, "swap": 2}
    assert report.conditional == {"phase": 6, "x": 4}
    assert report.instruction_qubits == 4


@pytest.mark.parametrize("inverse", [False, True])
def test_user_example_measures_the_prepared_frequency(inverse: bool) -> None:
    from examples.algorithms.fourier_measurement import make_circuit

    result = run_dynamic(
        make_circuit(inverse_qft=inverse), shots=16, seed=41, strategy="batched"
    )
    assert torch.all(result.classical_bits[:, :3] == torch.tensor([0, 1, 1]))
    assert torch.all(result.final_states.reshape(16, -1)[:, 1:].abs() < 1e-12)
