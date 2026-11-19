"""Close the complete frozen distillation channel utilities through native IR."""

from hashlib import sha1
from itertools import product
from pathlib import Path

import cirq
import numpy as np
import pytest
import torch
from qualtran.surface_code.t_factory_utils import (
    NoisyPauliRotation,
    _rotation_unitary,
)
from qualtran.surface_code.t_factory_utils import storage_error as source_storage

from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.qec.resources.distillation_channels import (
    noisy_pauli_rotation,
    storage_error,
)
from flagquantum.simulation.density_matrix import density_matrix_from_ir

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_frozen_complete_source_and_original_test_assets() -> None:
    root = (
        Path(__file__).resolve().parents[3] / "ftqc-reference-env/qualtran/surface_code"
    )
    for filename, digest in (
        ("t_factory_utils.py", "934703ab8dc9e7cde7df42b50ba4892a265dd605"),
        ("t_factory_utils_test.py", "f40e12d12434db519cb200ae9bbbf5572ffa8846"),
    ):
        raw = (root / filename).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


@pytest.mark.parametrize(
    "errors", ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.1, 0.2, 0.3))
)
def test_all_small_pauli_words_and_ordered_complex_branches(
    errors: tuple[float, float, float],
) -> None:
    for width in (1, 2, 3):
        for letters in product("IXYZ", repeat=width):
            word = "".join(letters)
            wires = tuple(reversed(range(width)))
            native = noisy_pauli_rotation(word, *errors, wires)
            original = NoisyPauliRotation(word, *errors)
            assert native.matrix is not None
            assert native.wires == wires
            assert native.metadata["is_channel"] is True
            assert cirq.num_qubits(original) == len(native.wires)
            assert len(native.matrix) == len(cirq.kraus(original)) == 4
            np.testing.assert_allclose(
                torch.stack(native.matrix).numpy(),
                cirq.kraus(original),
                rtol=0,
                atol=1e-15,
            )
            if errors == (0, 0, 0):
                np.testing.assert_allclose(
                    native.matrix[0].numpy(),
                    _rotation_unitary(word, np.pi / 8),
                    rtol=0,
                    atol=1e-15,
                )


@pytest.mark.parametrize("kind", ("X", "Z"))
@pytest.mark.parametrize("probabilities", ((0.0, 0.0), (1.0, 1.0), (0.2, 0.7)))
def test_noncontiguous_storage_and_rotation_execute_with_full_coherence(
    kind: str,
    probabilities: tuple[float, float],
) -> None:
    wires = (3, 0)
    prep = (Instruction("h", (0,)), Instruction("s", (0,)), Instruction("h", (3,)))
    native = (
        *prep,
        noisy_pauli_rotation("YX", 0.1, 0.2, 0.3, wires),
        *storage_error(kind, probabilities, wires),
    )
    qubits = tuple(cirq.LineQubit(i) for i in range(4))
    reference = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.S(qubits[0]),
        cirq.H(qubits[3]),
        NoisyPauliRotation("YX", 0.1, 0.2, 0.3).on(qubits[3], qubits[0]),
        source_storage(kind, probabilities, (qubits[3], qubits[0])),
    )
    expected = (
        cirq.DensityMatrixSimulator(dtype=np.complex128)
        .simulate(
            reference,
            qubit_order=qubits,
        )
        .final_density_matrix
    )
    actual = density_matrix_from_ir(
        CircuitIR(4, native).validate(),
        dtype=torch.complex128,
    )[0].numpy()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)
    assert np.trace(actual) == pytest.approx(1, rel=0, abs=2e-15)
    # Untouched wires 1 and 2 remain in |00>, including every off-diagonal entry.
    for i in range(16):
        if i & 0b0110:
            assert not np.any(actual[i, :])
            assert not np.any(actual[:, i])
    assert len(native) == 6
    assert [op.wires for op in native[-2:]] == [(3,), (0,)]


def test_storage_empty_and_mismatched_operands() -> None:
    assert storage_error("X", (), ()) == ()
    assert source_storage("X", (), ()) == []
    with pytest.raises(ValueError):
        storage_error("Z", (0.1,), (0, 1))
    with pytest.raises(ValueError):
        source_storage("Z", (0.1,), cirq.LineQubit.range(2))
    with pytest.raises(ValueError):
        storage_error("Y", (), ())
    with pytest.raises(ValueError):
        source_storage("Y", (), ())
