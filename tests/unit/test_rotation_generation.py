#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Adapted from Qualtran v0.7.0 rotation_synthesis/matrix/generation_test.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/rotation_synthesis/matrix/generation_test.py

import sys

import cirq
import numpy as np
import pytest
from numpy.typing import NDArray

from flagquantum.synth._rotation_generation import (
    generate_cliffords,
    generate_rotations,
    generate_rotations_iter,
)
from flagquantum.synth._su2_ct import SU2CliffordT


def _are_close_up_to_global_phase(
    u: NDArray[np.complex128], v: NDArray[np.complex128]
) -> bool:
    i, j = np.unravel_index(  # pylint: disable=unbalanced-tuple-unpacking
        np.argmax(np.abs(u)), u.shape
    )
    return bool(np.allclose(u * v[i, j] / u[i, j], v))


def test_generated_rotations_determinant() -> None:
    all_rotations = generate_rotations(5, with_progress_bar=False)
    for n in range(len(all_rotations)):
        assert np.allclose(
            [abs(np.linalg.det(r.matrix.astype(complex))) for r in all_rotations[n]],
            2 * (2 + np.sqrt(2)) ** n,
        )


_SQRT2 = np.sqrt(2)
_X = np.array([[0, 1], [1, 0]])
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.array([[1, 0], [0, -1]])
_TX_numpy = (np.eye(2) + (np.eye(2) - 1j * _X) / _SQRT2) / np.sqrt(2 + _SQRT2)
_TY_numpy = (np.eye(2) + (np.eye(2) - 1j * _Y) / _SQRT2) / np.sqrt(2 + _SQRT2)
_TZ_numpy = (np.eye(2) + (np.eye(2) - 1j * _Z) / _SQRT2) / np.sqrt(2 + _SQRT2)

_numpy_matrix_for_symbol = {
    "H": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    "S": np.array([[1, 0], [0, 1j]]),
    "Tx": _TX_numpy,
    "Ty": _TY_numpy,
    "Tz": _TZ_numpy,
}


def test_generated_rotations_unitary() -> None:
    all_rotations = generate_rotations(5, with_progress_bar=False)
    for row in all_rotations:
        for r in row:
            u = np.sqrt(2) * np.eye(2, dtype=np.complex128)
            gates = r.gates
            assert gates is not None
            for p in gates:
                u = _numpy_matrix_for_symbol[p] @ u
            assert _are_close_up_to_global_phase(u, r.numpy())


def test_generate_cliffords() -> None:
    cliffords = generate_cliffords()
    cirq_cliffords = [
        cirq.unitary(c) for c in cirq.SingleQubitCliffordGate.all_single_qubit_cliffords
    ]
    assert np.allclose(np.abs([np.linalg.det(c.numpy()) for c in cliffords]), 1)
    for c in cliffords:
        u = c.numpy()
        assert np.any([_are_close_up_to_global_phase(u, c) for c in cirq_cliffords])


def test_exact_source_order_and_labels() -> None:
    from qualtran.rotation_synthesis.matrix import _generation as source

    cliffords = generate_cliffords()
    assert cliffords is generate_cliffords()
    assert tuple(cliffords.values()) == tuple(source.generate_cliffords().values())
    native = generate_rotations_iter(5, with_progress_bar=False)
    locked = source.generate_rotations_iter(5, with_progress_bar=False)
    for level, (row, reference) in enumerate(zip(native, locked, strict=True)):
        assert len(row) == len(reference)
        assert [g.gates for g in row] == [g.gates for g in reference]
        for got, want in zip(row, reference, strict=True):
            assert [v.coords for v in got.matrix.flat] == [
                v.coords for v in want.matrix.flat
            ]
            assert got.num_t_gates() == want.num_t_gates() == level
            assert got.gates is not None
            assert SU2CliffordT.from_sequence(got.gates) == got
            np.testing.assert_array_equal(got.numpy(), want.numpy())


def test_progress_switches(capsys: pytest.CaptureFixture[str]) -> None:
    generate_rotations(1, with_progress_bar=False)
    assert capsys.readouterr() == ("", "")
    generate_rotations(1, with_progress_bar=True, verbose=True)
    output, progress = capsys.readouterr()
    assert output == "1 72\n"
    assert "generate level 1" in progress


def test_unavailable_optional_progress(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "tqdm", None)
    assert generate_rotations(-1) == [tuple(generate_cliffords())]
    assert generate_rotations(1) == generate_rotations(1, with_progress_bar=False)
    assert capsys.readouterr() == ("", "")
