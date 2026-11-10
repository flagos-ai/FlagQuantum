"""Optional HDF5 imports stop at the file boundary."""

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def test_reader_import_and_native_workflow_without_hdf5() -> None:
    script = """
        import importlib.abc
        import sys
        blocked={"numpy","h5py","scipy","pyscf","openfermion","openfermionpyscf","cirq"}
        class Blocked(importlib.abc.MetaPathFinder):
            def find_spec(self,fullname,path=None,target=None):
                if fullname.split(".")[0] in blocked:
                    raise ModuleNotFoundError(fullname)
        sys.meta_path.insert(0,Blocked())
        from flagquantum.ecosystem.chemistry.molecular_hdf5 import read_molecular_hdf5
        from examples.algorithms.fermion_chemistry import main
        try:
            read_molecular_hdf5("not-opened")
        except ImportError as error:
            assert "chemistry-hdf5" in str(error)
        else:
            raise AssertionError("missing optional dependency accepted")
        main()
        assert not blocked.intersection(name.split(".")[0] for name in sys.modules)
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("h5py") is None, reason="requires chemistry-hdf5"
)
def test_file_measurement_without_circuit_or_chemistry_sdks() -> None:
    script = """
        import importlib.abc
        import sys
        blocked={"scipy","pyscf","openfermion","openfermionpyscf","cirq"}
        class Blocked(importlib.abc.MetaPathFinder):
            def find_spec(self,fullname,path=None,target=None):
                if fullname.split(".")[0] in blocked:
                    raise ModuleNotFoundError(fullname)
        sys.meta_path.insert(0,Blocked())
        from examples.algorithms.fermion_hdf5 import main
        main()
        assert not blocked.intersection(name.split(".")[0] for name in sys.modules)
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("reference energy -0.9000000000 Ha") == 3
