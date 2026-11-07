"""Check the locked clipped permutation, phases, work and approximation error."""

from collections import Counter
from math import ceil

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.finite_field import ClassicalFieldCoefficient as Bit
from flagquantum.compiler.registers.kickmix_near_power_doubling import (
    kickmix_near_power_doubling_ir,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_kickmix_bit_rotation import _certify

pytestmark = pytest.mark.unit


def _clipped_map(x: int, modulus: int, width: int, span: int, inverse: bool) -> int:
    mask = (1 << (span - 1)) - 1
    word_mask = (1 << width) - 1
    if not inverse:
        x = (x << 1 | x >> (width - 1)) & word_mask
    middle = ((x >> 1) + (1 if inverse else -1) * ((modulus >> 1) + 1) * (x & 1)) & mask
    x = (x & ~(mask << 1)) | (middle << 1)
    return (x >> 1 | (x & 1) << (width - 1)) if inverse else x


def _check_case(
    width: int,
    coefficients: tuple[int | bool | Bit, ...],
    *,
    tolerance: float,
    inverse: bool,
    classical_value: bool = False,
    quantum_count: int = 0,
    dirty_count: int = 1,
    minimize: bool = False,
    extra: int = 0,
) -> None:
    cutoff = len(coefficients)
    for value in coefficients:
        if value is not True:
            break
        cutoff -= 1
    span = cutoff + ceil(tolerance)
    data = width + quantum_count + dirty_count
    dirty = tuple(range(width + quantum_count, data))
    clean = tuple(range(data, data + span + extra))
    slots = 1 if any(isinstance(v, Bit) for v in coefficients) else 0
    gates = kickmix_near_power_doubling_ir(
        tuple(range(width)),
        coefficients,
        btol=tolerance,
        clean=clean,
        dirty=dirty,
        classical_bits=tuple(range(slots, slots + width + 3)),
        inverse=inverse,
        minimize_qubits=minimize,
    )
    prefix = (
        (ClassicalXor((), 0, constant=classical_value).instruction(),) if slots else ()
    )
    program = lower_reversible_phase(CircuitIR(data + len(clean), (*prefix, *gates)))
    assert decode_kmx(encode_kmx(program)) == program
    labels = torch.arange(1 << data)
    outputs = []
    for label in labels.tolist():
        modulus = sum(
            (
                int(value)
                if isinstance(value, bool)
                else (
                    int(classical_value)
                    if isinstance(value, Bit)
                    else (label >> (data - 1 - value) & 1)
                )
            )
            << bit
            for bit, value in enumerate(reversed(coefficients))
        )
        x = label >> (data - width)
        value = _clipped_map(x, modulus, width, span, inverse)
        outputs.append(label ^ ((x ^ value) << (data - width)))
    outputs = torch.tensor(outputs)
    _certify(program, data, outputs)
    amplitudes = torch.complex((labels + 1).double(), (3 * labels - 2).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << program.n_qubits, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    initial[labels << len(clean)] = amplitudes
    expected[outputs << len(clean)] = amplitudes
    circuit = DynamicCircuit(
        program.n_qubits, inputs=initial, dtype=torch.complex128
    ).append(FieldOperation(program.n_qubits, (*prefix, *gates)))
    result = run_dynamic(circuit, shots=8, seed=41, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), expected.expand(8, -1), atol=1e-12, rtol=0
    )
    if slots:
        assert torch.all(result.classical_bits[:, 0] == classical_value)
    backward = kickmix_near_power_doubling_ir(
        tuple(range(width)),
        coefficients,
        btol=tolerance,
        clean=clean,
        dirty=dirty,
        classical_bits=tuple(range(slots, slots + width + 3)),
        inverse=not inverse,
        minimize_qubits=minimize,
    )
    restored = run_dynamic(
        circuit.append(FieldOperation(program.n_qubits, backward)),
        shots=8,
        seed=41,
        strategy="batched",
    )
    torch.testing.assert_close(
        restored.final_states.reshape(8, -1), initial.expand(8, -1), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    "width,tolerance", [(1, 1), (2, 1), (3, 1), (3, 2.5), (4, 2), (5, 2.5)]
)
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("extra", [0, 2])
def test_all_ones_word_keeps_the_source_cyclic_permutation(
    width: int, tolerance: float, inverse: bool, extra: int
) -> None:
    _check_case(
        width, (True,) * width, tolerance=tolerance, inverse=inverse, extra=extra
    )


@pytest.mark.parametrize("mode", ["constant", "classical", "quantum"])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("tolerance", [0, 0.5, 1])
def test_mixed_modulus_clipped_map(mode: str, inverse: bool, tolerance: float) -> None:
    coefficient = True if mode == "constant" else Bit(0) if mode == "classical" else 4
    coefficients = (True, coefficient, False, True)
    for value in (False, True) if mode == "classical" else (False,):
        _check_case(
            4,
            coefficients,
            tolerance=tolerance,
            inverse=inverse,
            classical_value=value,
            quantum_count=int(mode == "quantum"),
        )


@pytest.mark.parametrize("inverse", [False, True])
def test_clipped_window_with_a_wider_offset_and_entangled_vented_work(
    inverse: bool,
) -> None:
    _check_case(
        8,
        (True, True, True, True, True, False, True, True),
        tolerance=2,
        inverse=inverse,
        dirty_count=2,
        minimize=True,
    )


def test_source_adjacent_swap_order_and_span_checks() -> None:
    forward = kickmix_near_power_doubling_ir(
        (0, 1, 2, 3), (True,) * 4, btol=1, clean=(4,), classical_bits=(0,)
    )
    assert [(op.name, op.wires) for op in forward] == [
        ("swap", (0, 1)),
        ("swap", (1, 2)),
        ("swap", (2, 3)),
    ]
    reverse = kickmix_near_power_doubling_ir(
        (0, 1, 2, 3), (True,) * 4, btol=1, clean=(4,), classical_bits=(0,), inverse=True
    )
    assert [op.wires for op in reverse] == [(2, 3), (1, 2), (0, 1)]
    for tolerance in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite tolerance"):
            kickmix_near_power_doubling_ir((0,), (True,), btol=tolerance, clean=(1,))
    with pytest.raises(ValueError, match="nonempty clipped span"):
        kickmix_near_power_doubling_ir((0,), (True,), btol=0, clean=(1,))
    with pytest.raises(ValueError, match="cutoff plus tolerance"):
        kickmix_near_power_doubling_ir((0, 1), (False, True), btol=1, clean=(2, 3))
    with pytest.raises(ValueError, match="bottom_span clean"):
        kickmix_near_power_doubling_ir((0, 1, 2), (True,) * 3, btol=2, clean=(3,))


def test_near_power_resource_counts_remain_primitive() -> None:
    gates = kickmix_near_power_doubling_ir(
        (0, 1, 2, 3), (True,) * 4, btol=2.5, clean=(4, 5, 6), classical_bits=(0,)
    )
    counts = Counter(
        op.opcode for op in lower_reversible_phase(CircuitIR(7, gates)).operations
    )
    assert counts["swap"] == 3
    assert counts["hmr"] == 1


@pytest.mark.ftqc_reference
def test_low_precision_matches_the_unmodified_native_fixture_and_error() -> None:
    km = pytest.importorskip("kickmix")
    assert hasattr(km, "Simulator")
    native = "SWAP q3 q2\nSWAP q2 q1\nSWAP q1 q0\nX q1\nCX q0 q1\nX q1"
    sim = km.Simulator(batch_size=1, seed=41)
    for q in (0, 1, 2):
        sim.write_across_shots(km.q(q), 1)
    sim.do(km.Circuit(native))
    assert sum(sim.read_across_shots(km.q(q), out=int) << q for q in range(4)) == 14
    assert sim.read_shot_phase(0) == 0
    gates = kickmix_near_power_doubling_ir(
        (3, 2, 1, 0),
        (True, True, False, True),
        btol=0,
        clean=(4, 5),
        classical_bits=(0,),
    )
    program = lower_reversible_phase(CircuitIR(6, gates))
    assert encode_kmx(program).splitlines()[1:] == native.splitlines()
    initial = torch.zeros(64, dtype=torch.complex128)
    initial[14 << 2] = 1  # Source qubit 0 is the low bit of input seven.
    expected = torch.zeros_like(initial)
    expected[7 << 2] = 1
    exact = torch.zeros_like(initial)
    exact[8 << 2] = 1  # Exact modular doubling would return one.
    circuit = DynamicCircuit(6, inputs=initial, dtype=torch.complex128).append(
        FieldOperation(6, gates)
    )
    result = run_dynamic(circuit, shots=1, seed=41, strategy="batched")
    actual = result.final_states.reshape(-1)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert (actual - exact).norm().item() == pytest.approx(2**0.5)


@pytest.mark.parametrize("inverse", [False, True])
def test_native_vented_primitive_counts(inverse: bool) -> None:
    gates = kickmix_near_power_doubling_ir(
        tuple(reversed(range(8))),
        (True, True, True, True, True, False, True, True),
        btol=2,
        clean=(8, 9, 10, 11, 12),
        dirty=(13, 14),
        classical_bits=tuple(range(11)),
        minimize_qubits=True,
        inverse=inverse,
    )
    counts = Counter(
        op.opcode for op in lower_reversible_phase(CircuitIR(15, gates)).operations
    )
    assert counts == {
        "swap": 7,
        "x": 8 if inverse else 16,
        "cx": 19,
        "ccx": 8,
        "hmr": 2,
        "cz": 1,
        "z": 3,
        "bit_store0": 1,
        "bit_invert": 1,
    }
