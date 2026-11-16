"""The file-to-measurement path requires no external science packages."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_fcidump_workflows_with_external_science_imports_blocked() -> None:
    script = """
        import importlib.abc
        import runpy
        import sys
        blocked = {"openfermion", "openfermionpyscf", "cirq", "pyscf", "numpy", "scipy", "h5py"}
        class NoExternalScience(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in blocked:
                    raise ModuleNotFoundError("blocked science dependency: " + fullname)
                return None
        sys.meta_path.insert(0, NoExternalScience())
        runpy.run_module("examples.algorithms.fermion_fcidump", run_name="__main__")
        assert not (blocked & {name.split(".")[0] for name in sys.modules})
        # PyTorch optimizer initialization may import NumPy.
        blocked.remove("numpy")
        from examples.algorithms.fermion_fcidump_training import main as train
        train(encoding="bravyi_kitaev")
        from examples.algorithms.fermion_h2_tapered_uccsd import main as train_tapered
        for encoding in ("jordan_wigner", "parity", "bravyi_kitaev"):
            train_tapered(encoding=encoding)
        assert not (blocked & {name.split(".")[0] for name in sys.modules})
    """
    script = textwrap.dedent(script)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("reference energy -0.9000000000 Ha") == 3
    assert "Encoding: bravyi_kitaev" in result.stdout
    assert "Particles: 2.00000000" in result.stdout
    assert result.stdout.count("Register: 4 -> 2 qubits") == 3
