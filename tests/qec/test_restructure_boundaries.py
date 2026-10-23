"""Exercise the approved matrix/noise split without changing the 001 baseline."""

from __future__ import annotations

import dataclasses
import io
import json
import pickle
import subprocess
import sys
from pathlib import Path
from types import FunctionType

import pytest
import torch

import flagquantum.qec as qec
from flagquantum.qec import dem_construction, dem_matrices, noise
from flagquantum.qec.codes import matrices
from flagquantum.qec.execution.noise_sweep import run_repetition_memory_noise_sweep

_FIXTURES = Path(__file__).with_name("fixtures")


@pytest.mark.unit
def test_matrix_and_sweep_aliases_preserve_historical_identity() -> None:
    assert matrices.CssCodeMatrices is dem_matrices.CssCodeMatrices
    assert matrices.CssCodeMatrices is dem_construction.CssCodeMatrices
    assert matrices.CssCodeMatrices is qec.CssCodeMatrices
    assert matrices.css_code_matrices is dem_matrices.css_code_matrices
    assert matrices.css_code_matrices is dem_construction.css_code_matrices
    assert matrices.css_code_matrices is qec.css_code_matrices
    assert qec.CssCodeMatrices.__module__ == "flagquantum.qec.dem_matrices"
    for member in vars(qec.CssCodeMatrices).values():
        if isinstance(member, property):
            member = member.fget
        if isinstance(member, FunctionType):
            assert member.__module__ == "flagquantum.qec.dem_matrices"
    assert qec.css_code_matrices.__module__ == "flagquantum.qec.dem_matrices"
    assert run_repetition_memory_noise_sweep is noise.run_repetition_memory_noise_sweep
    assert run_repetition_memory_noise_sweep is qec.run_repetition_memory_noise_sweep
    assert run_repetition_memory_noise_sweep.__module__ == "flagquantum.qec.noise"
    assert (
        noise.run_repetition_memory_experiment is qec.run_repetition_memory_experiment
    )
    assert "run_repetition_memory_noise_sweep" in dir(noise)
    with pytest.raises(AttributeError, match="no attribute"):
        getattr(noise, "missing_noise_record")


@pytest.mark.integration
def test_pre_move_tensor_pickle_loads_and_new_checkpoints_keep_the_old_path() -> None:
    # This repository-owned protocol-4 fixture was written by the unchanged 001 code.
    saved = pickle.loads((_FIXTURES / "restructure_002_matrices.pkl").read_bytes())
    assert type(saved["matrices"]) is qec.CssCodeMatrices
    for name, method in saved["methods"].items():
        assert method is getattr(qec.CssCodeMatrices, name)
        assert pickle.loads(pickle.dumps(method, protocol=4)) is method
    assert saved["extract"] is qec.css_code_matrices
    assert saved["sweep"] is qec.run_repetition_memory_noise_sweep
    expected = qec.css_code_matrices(qec.SteaneCode())
    for field in ("hz", "hx", "lz", "lx"):
        assert torch.equal(getattr(saved["matrices"], field), getattr(expected, field))
    stream = io.BytesIO()
    torch.save(saved, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    records = io.BytesIO()
    torch.save((saved["matrices"], saved["noise"]), records)
    records.seek(0)
    with torch.serialization.safe_globals(
        [qec.CssCodeMatrices, qec.RepetitionNoiseProfile]
    ):
        allowed = torch.load(records, weights_only=True)
    assert type(allowed[0]) is qec.CssCodeMatrices
    assert type(restored["matrices"]) is qec.CssCodeMatrices
    assert restored["extract"] is qec.css_code_matrices
    assert restored["sweep"] is qec.run_repetition_memory_noise_sweep
    assert pickle.dumps(restored["extract"], protocol=4) == pickle.dumps(
        saved["extract"], protocol=4
    )
    for field in ("hz", "hx", "lz", "lx"):
        assert torch.equal(
            getattr(restored["matrices"], field), getattr(expected, field)
        )
    assert (
        restored["noise"].to_noise_model().identity
        == json.loads((_FIXTURES / "restructure_002_sweep.json").read_text())[
            "noise_identity"
        ]
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "mode", ("compiled_lookup", "offline_pauli_frame", "runtime_decoder")
)
def test_seeded_sweeps_match_the_pre_move_results(mode: str) -> None:
    expected = json.loads((_FIXTURES / "restructure_002_sweep.json").read_text())
    points = noise.run_repetition_memory_noise_sweep(
        (0.0, 0.125),
        rounds=3,
        shots=32,
        seed=53,
        strategy="trajectory",
        feedback_mode=mode,
    )
    assert [dataclasses.asdict(point) for point in points] == expected["sweeps"][mode]


@pytest.mark.integration
def test_data_modules_construct_codes_without_importing_execution() -> None:
    # The existing root facade eagerly imports QEC. After bootstrapping it, evict
    # the leaf modules and infrastructure so their own dependencies are measured.
    script = """
import importlib.abc
import sys
import flagquantum.qec
blocked = ("flagquantum.compiler", "flagquantum.runtime",
           "flagquantum.qec.dem_construction", "flagquantum.qec.dem_matrices",
           "flagquantum.qec.repetition", "flagquantum.qec.execution")
for name in list(sys.modules):
    if name.startswith(blocked) or name in (
        "flagquantum.qec.codes.matrices", "flagquantum.qec.css_code", "flagquantum.qec.noise"
    ):
        del sys.modules[name]
class NoExecution(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if fullname.startswith(blocked):
            raise AssertionError("Data module imported execution: " + fullname)
sys.meta_path.insert(0, NoExecution())
from flagquantum.qec.codes import SteaneCode
from flagquantum.qec.codes.matrices import css_code_matrices
from flagquantum.qec.css_code import CssCode
from flagquantum.qec.noise import PhenomenologicalNoise, RepetitionNoiseProfile
code = CssCode(matrices=css_code_matrices(SteaneCode()), distance=3)
assert code.num_data_qubits == 7
assert PhenomenologicalNoise(data_flip=0.125).data_flip_rates(num_qubits=7) == (0.125,) * 7
assert RepetitionNoiseProfile(0.125).to_noise_model().identity
assert not any(name.startswith(blocked) for name in sys.modules)
print("Data-only construction passed")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout.strip() == "Data-only construction passed"
