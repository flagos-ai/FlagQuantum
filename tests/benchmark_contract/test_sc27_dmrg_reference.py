from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / "paper" / "sc27" / "build_dmrg_reference.py"
SPEC = importlib.util.spec_from_file_location("sc27_dmrg_reference", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def raw_reference() -> dict:
    return {
        "solver": "ITensor",
        "solver_version": "test",
        "command": "julia dmrg.jl",
        "n_sites": 256,
        "anisotropy": 1.0,
        "field": 0.0,
        "boundary_condition": "open",
        "energy": -440.0,
        "energy_variance": 1e-10,
        "max_bond": 512,
        "sweeps": 20,
        "discarded_weight": 1e-12,
        "converged": True,
    }


def test_normalizes_independent_reference() -> None:
    result = MODULE.normalize(raw_reference(), raw_sha256="a" * 64)
    assert result["result"]["energy"] == -440.0
    assert result["provenance"]["independent_of_flagquantum"] is True


def test_rejects_flagquantum_or_workload_drift() -> None:
    raw = raw_reference()
    raw["command"] = "python flagquantum_reference.py"
    with pytest.raises(ValueError, match="independent"):
        MODULE.normalize(raw, raw_sha256="a" * 64)
    raw = raw_reference()
    raw["n_sites"] = 128
    with pytest.raises(ValueError, match="256"):
        MODULE.normalize(raw, raw_sha256="a" * 64)
