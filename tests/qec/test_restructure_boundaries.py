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
from _retirement import (
    assert_retired_payload,
)

import flagquantum.qec as qec
from flagquantum.qec import noise
from flagquantum.qec.codes import matrices
from flagquantum.qec.execution.noise_sweep import run_repetition_memory_noise_sweep

_FIXTURES = Path(__file__).with_name("fixtures")


@pytest.mark.unit
def test_matrix_and_sweep_owners_preserve_root_identity() -> None:
    from flagquantum.qec import noise

    assert matrices.CssCodeMatrices is qec.CssCodeMatrices
    assert matrices.css_code_matrices is qec.css_code_matrices
    assert qec.CssCodeMatrices.__module__ == "flagquantum.qec.codes.matrices"
    for member in vars(qec.CssCodeMatrices).values():
        if isinstance(member, property):
            member = member.fget
        if isinstance(member, FunctionType):
            assert member.__module__ == "flagquantum.qec.codes.matrices"
    assert qec.css_code_matrices.__module__ == "flagquantum.qec.codes.matrices"
    assert (
        run_repetition_memory_noise_sweep
        is noise.run_repetition_memory_noise_sweep
        is qec.run_repetition_memory_noise_sweep
    )
    assert run_repetition_memory_noise_sweep.__module__ == "flagquantum.qec.noise"
    assert (
        noise.run_repetition_memory_experiment is qec.run_repetition_memory_experiment
    )
    assert "run_repetition_memory_noise_sweep" in dir(noise)
    with pytest.raises(AttributeError, match="no attribute"):
        getattr(noise, "missing_noise_record")


@pytest.mark.integration
def test_retired_matrix_pickle_and_canonical_tensor_checkpoints() -> None:
    assert_retired_payload("restructure_002_matrices.pkl")
    expected = json.loads(
        (
            Path(__file__).with_name("retirement_fixtures")
            / "pre_retirement_matrices.json"
        ).read_text()
    )
    matrices = qec.css_code_matrices(qec.SteaneCode())
    for field, record in expected.items():
        value = getattr(matrices, field)
        assert str(value.dtype) == record["dtype"]
        assert value.tolist() == record["values"]
    survivors = pickle.loads(
        (
            Path(__file__).with_name("retirement_fixtures") / "surviving_instances.pkl"
        ).read_bytes()
    )
    values = (matrices, survivors["restructure_002_matrices.pkl"]["noise"])
    records = io.BytesIO()
    torch.save(values, records)
    records.seek(0)
    with torch.serialization.safe_globals(
        [qec.CssCodeMatrices, qec.RepetitionNoiseProfile]
    ):
        restored = torch.load(records, weights_only=True)
    assert type(restored[0]) is qec.CssCodeMatrices
    for field in expected:
        assert torch.equal(getattr(restored[0], field), getattr(matrices, field))
    assert (
        restored[1].to_noise_model().identity
        == json.loads((_FIXTURES / "restructure_002_sweep.json").read_text())[
            "noise_identity"
        ]
    )
    for value in (
        qec.CssCodeMatrices,
        qec.css_code_matrices,
        qec.run_repetition_memory_noise_sweep,
    ):
        assert pickle.loads(pickle.dumps(value, protocol=4)) is value


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
        "flagquantum.qec.codes.matrices", "flagquantum.qec.codes.css",
        "flagquantum.qec.css_code", "flagquantum.qec.noise"
    ):
        del sys.modules[name]
class NoExecution(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if fullname.startswith(blocked):
            raise AssertionError("Data module imported execution: " + fullname)
sys.meta_path.insert(0, NoExecution())
from flagquantum.qec.codes import SteaneCode
from flagquantum.qec.codes.matrices import css_code_matrices
from flagquantum.qec.codes.css import CssCode
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
