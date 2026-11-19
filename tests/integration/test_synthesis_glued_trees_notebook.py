"""Enforce uncropped source Hamiltonians, original sweeps and absolute states."""

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq

pytest.importorskip("numpy")

from examples.synthesis.algorithms.quantum_walks.glued_trees.overview import (
    quantum,
    workflow,
)
from examples.synthesis.pauli_evolution import (
    load_terms,
    reference_circuit,
)

pytestmark = pytest.mark.integration


def test_full_source_inputs() -> None:
    assert quantum.EDGES_6 == 44
    assert quantum.MATRIX_6.shape == (64, 64)
    assert len(quantum.TERMS_6) == 1088 and quantum.EXPANSION_ERROR < 1e-12
    root = Path(quantum.__file__).parent
    provenance = json.loads((root / "provenance.json").read_text())
    data = root / "glued_trees_20_terms.json"
    assert (
        hashlib.sha256(data.read_bytes()).hexdigest()
        == provenance["source_data"]["converted_ordered_cache"]
    )
    assert (
        hashlib.sha256((root / "LICENSE-data.txt").read_bytes()).hexdigest()
        == provenance["source_data"]["license"]
    )
    width, terms = load_terms(data)
    assert width == 20 and len(terms) == 200
    assert workflow.time_points(6) == tuple(float(t) for t in range(-4, 21, 2))
    assert workflow.time_points(20) == tuple(float(t) for t in range(24, 49, 2))
    assert (workflow.exit_word(6), workflow.exit_word(20)) == (29, 524285)
    assert workflow.SHOTS == 8192


@pytest.mark.parametrize("time", [-2.0, 2.0])
def test_six_qubit_composition_preserves_absolute_state(time: float) -> None:
    state = quantum.evolve(6, time)
    terms = quantum.TERMS_6
    order = tuple((i, time * coef / 10) for i, (coef, _) in enumerate(terms)) * 10
    expected = fq.run(reference_circuit(6, terms, order)).statevector().flatten()
    torch.testing.assert_close(state.amplitudes, expected, rtol=0, atol=5e-12)
    assert state.work_qubits == 0 and state.max_physical_width == 6


def test_wide_full_state_at_original_first_time() -> None:
    np = pytest.importorskip("numpy")
    words = np.arange(1 << 20, dtype=np.uint32)
    result = np.zeros(words.size, dtype=np.complex128)
    result[0] = 1
    _, terms = load_terms(Path(quantum.__file__).with_name("glued_trees_20_terms.json"))
    time = 24.0
    for _ in range(10):
        for coef, support in terms:
            flips = 0
            signs = 0
            num_y = 0
            for bit, axis in support:
                if axis in ("X", "Y"):
                    flips |= 1 << bit
                if axis in ("Y", "Z"):
                    signs |= 1 << bit
                num_y += int(axis == "Y")
            source = words ^ np.uint32(flips)
            folded = source & np.uint32(signs)
            for shift in (16, 8, 4, 2, 1):
                folded ^= folded >> shift
            odd = folded & 1
            theta = time * coef / 10
            phase = (1j**num_y) * (1 - 2 * odd.astype(np.int8))
            result = (
                math.cos(theta) * result - 1j * math.sin(theta) * phase * result[source]
            )
    actual = quantum.evolve(20, time)
    torch.testing.assert_close(
        actual.amplitudes, torch.from_numpy(result), rtol=0, atol=1e-10
    )
    assert actual.amplitudes.numel() == 1 << 20 and actual.work_qubits == 0
    word = workflow.exit_word(20)
    assert float(actual.amplitudes[word].abs().square()) > 0


def test_unsupported_width_and_nonfinite_time() -> None:
    for width, time in [(5, 2.0), (6, float("nan")), (20, float("inf"))]:
        with pytest.raises(ValueError, match="finite times"):
            quantum.evolve(width, time)


def test_unexecuted_notebook_has_all_original_stages() -> None:
    root = Path(quantum.__file__).parent
    book = json.loads((root / "walkthrough.ipynb").read_text())
    proof = json.loads((root / "provenance.json").read_text())
    cells = [c for c in book["cells"] if c["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 8
    assert all(not c["outputs"] and c["execution_count"] is None for c in cells)
