"""Importing the object adapter keeps native execution independent of upstream."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_molecular_object_adapter_without_external_science_packages() -> None:
    script = """
        import importlib.abc
        import sys
        blocked = {"openfermion", "cirq", "numpy", "h5py", "scipy", "pyscf"}
        class NoScience(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in blocked:
                    raise ModuleNotFoundError(fullname)
        sys.meta_path.insert(0, NoScience())
        import flagquantum as fq
        from flagquantum.ecosystem.chemistry.molecular_data import from_openfermion_molecular_data
        from flagquantum.observables.fermion import jordan_wigner, number
        try:
            from_openfermion_molecular_data(None)
        except ImportError as error:
            assert "separately installed" in str(error)
        else:
            raise AssertionError("missing upstream dependency accepted")
        value = fq.run(fq.Circuit(2).x(0), outputs=[fq.expectation(jordan_wigner(number(0), n_modes=2))])
        assert float(value.expectation().reshape(-1)[0]) == 1
        assert not blocked.intersection(name.split(".")[0] for name in sys.modules)
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
