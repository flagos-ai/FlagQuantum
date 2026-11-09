"""Check zero repair, measured phases and modulus input preservation."""

from collections import Counter
from itertools import product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.compiler.registers.modular.mixed_modular_negation import (
    mixed_modular_negation_program,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_kickmix_modular_addition import _certify_all_histories

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("width,modulus", [(1, 1), (2, 3), (3, 5), (4, 11)])
@pytest.mark.parametrize("classical", [False, True])
def test_modular_negation_preserves_complex_amplitudes(
    width: int, modulus: int, classical: bool
) -> None:
    low_bits = tuple(bool((modulus >> i) & 1) for i in reversed(range(width - 1)))
    coefficients = (
        True,
        *(
            tuple(ClassicalFieldCoefficient(i) for i in range(width - 1))
            if classical
            else low_bits
        ),
    )
    clean = tuple(range(width, 2 * width + 2))
    program = mixed_modular_negation_program(
        tuple(range(width)), coefficients, clean=clean
    )
    assert decode_kmx(encode_kmx(program)) == program
    initial = torch.zeros(1 << program.n_qubits, dtype=torch.complex128)
    amplitudes = torch.complex(
        torch.arange(1, modulus + 1).double(), torch.arange(modulus).double()
    )
    amplitudes /= amplitudes.norm()
    stride = 1 << len(clean)
    expected = torch.zeros_like(initial)
    for x, amplitude in enumerate(amplitudes):
        initial[x * stride] = amplitude
        expected[(-x % modulus) * stride] = amplitude
    setup = tuple(
        ClassicalXor((), i, constant=value).instruction()
        for i, value in enumerate(low_bits)
        if classical
    )
    gates = bind_reversible_phase_execution(program)
    circuit = DynamicCircuit(
        program.n_qubits, inputs=initial, dtype=torch.complex128
    ).append(FieldOperation(program.n_qubits, (*setup, *gates)))
    result = run_dynamic(circuit, shots=64, seed=41, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(64, -1), expected.expand(64, -1), atol=1e-12, rtol=0
    )
    if classical:
        assert (
            result.classical_bits[:, : width - 1].tolist()
            == [list(map(int, low_bits))] * 64
        )


@pytest.mark.parametrize("width,modulus", [(2, 3), (3, 5)])
def test_modular_negation_all_measurement_and_reset_histories(
    width: int, modulus: int
) -> None:
    np = pytest.importorskip("numpy")
    idle = 2 * width + 1
    program = mixed_modular_negation_program(
        tuple(reversed(range(width))),
        tuple(bool((modulus >> i) & 1) for i in reversed(range(width))),
        clean=tuple(range(idle, idle + width + 1)),
    )
    actual = np.asarray(_certify_all_histories(program, width=width, modulus=modulus))
    expected = np.repeat([(-x) % modulus for x in range(modulus)], 2 * modulus)
    np.testing.assert_array_equal(
        actual.reshape(-1, len(expected)),
        np.broadcast_to(expected, (len(actual) // len(expected), len(expected))),
    )
    counts = Counter(op.opcode for op in program.operations)
    assert counts["hmr"] == 3 * width


def test_modular_negation_checks_full_capacity_and_zero_repair_work() -> None:
    with pytest.raises(ValueError, match="full-capacity"):
        mixed_modular_negation_program((0, 1), (False, True), clean=(2, 3, 4))
    with pytest.raises(ValueError, match="clean zero-repair"):
        mixed_modular_negation_program((0,), (True,), clean=())
    with pytest.raises(ValueError, match="one clean wire per input"):
        mixed_modular_negation_program((0, 1), (True, True), clean=(2, 3))


@pytest.mark.parametrize("width", [2, 3])
def test_quantum_modulus_and_entangled_dirty_work_are_preserved(width: int) -> None:
    low_modulus = tuple(range(width, 2 * width - 1))
    clean = tuple(range(2 * width - 1, 3 * width))
    dirty = (3 * width, 3 * width + 1)
    program = mixed_modular_negation_program(
        tuple(range(width)), (True, *low_modulus), clean=clean, dirty=dirty
    )
    initial = torch.zeros(1 << program.n_qubits, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for modulus in range(1 << (width - 1), 1 << width):
        for x, borrowed in product(range(modulus), range(4)):
            suffix = ((modulus - (1 << (width - 1))) << (len(clean) + 2)) | borrowed
            shift = (width - 1) + len(clean) + 2
            amplitude = complex(x + 1, modulus + 2 * borrowed)
            initial[(x << shift) | suffix] = amplitude
            expected[((-x % modulus) << shift) | suffix] = amplitude
    scale = initial.norm()
    circuit = DynamicCircuit(
        program.n_qubits, inputs=initial / scale, dtype=torch.complex128
    ).append(FieldOperation(program.n_qubits, bind_reversible_phase_execution(program)))
    result = run_dynamic(circuit, shots=32, seed=41, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(32, -1),
        (expected / scale).expand(32, -1),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "width,modulus,work,source_counts",
    [
        (2, 3, 3, {"x": 18, "cx": 5, "ccx": 3, "cz": 3, "z": 3, "hmr": 6}),
        (
            3,
            5,
            4,
            {"x": 23, "cx": 6, "ccx": 6, "cz": 5, "z": 3, "neg": 1, "hmr": 9},
        ),
        (
            4,
            11,
            6,
            {"x": 34, "cx": 10, "ccx": 9, "cz": 9, "z": 2, "neg": 2, "hmr": 12},
        ),
    ],
)
def test_resources_match_locked_native_primitive_counts(
    width: int, modulus: int, work: int, source_counts: dict[str, int]
) -> None:
    program = mixed_modular_negation_program(
        tuple(reversed(range(width))),
        tuple(bool(modulus >> i & 1) for i in reversed(range(width))),
        clean=tuple(range(width, width + work)),
    )
    counts = Counter(op.opcode for op in program.operations)
    # Native leaf R operations initialize the already-clean caller workspace.
    assert counts.pop("r") == 1
    assert counts.pop("push_condition") == counts.pop("pop_condition") == 1
    assert counts == source_counts
    emitted = Counter(op.name for op in bind_reversible_phase_execution(program))
    assert emitted["ccx"] == source_counts["ccx"]
    assert emitted["measure"] == source_counts["hmr"]
    assert program.n_qubits == width + work
