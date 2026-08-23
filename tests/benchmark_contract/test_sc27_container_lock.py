from __future__ import annotations

import importlib.util
from pathlib import Path

PATH = Path(__file__).parents[2] / "paper" / "sc27" / "containers" / "verify_lock.py"
MPI_LAUNCHER = PATH.parent / "pennylane" / "sc27-mpirun"
PENNYLANE_DOCKERFILE = PATH.parent / "pennylane" / "Dockerfile"
SPEC = importlib.util.spec_from_file_location("sc27_container_lock", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_accepts_installed_exact_pin(tmp_path: Path) -> None:
    lock = tmp_path / "lock.txt"
    lock.write_text("packaging==26.2\n", encoding="utf-8")
    assert MODULE.errors(lock) == []


def test_rejects_non_exact_missing_and_wrong_version(tmp_path: Path) -> None:
    lock = tmp_path / "lock.txt"
    lock.write_text(
        "packaging>=1\nmissing-sc27-package==1.0\npackaging==0.0\n",
        encoding="utf-8",
    )
    failures = MODULE.errors(lock)
    assert any("not_exact_pin" in item for item in failures)
    assert any("missing:missing-sc27-package" in item for item in failures)
    assert any("version_mismatch:packaging" in item for item in failures)


def test_pennylane_launcher_supports_default_root_container_user() -> None:
    launcher = MPI_LAUNCHER.read_text(encoding="utf-8")
    assert "--allow-run-as-root" in launcher
    assert "--mca plm isolated" in launcher


def test_pennylane_image_requires_cuda_aware_openmpi() -> None:
    dockerfile = PENNYLANE_DOCKERFILE.read_text(encoding="utf-8")
    launcher = MPI_LAUNCHER.read_text(encoding="utf-8")
    assert "OPENMPI_CUDA_SHA256=" in dockerfile
    assert "opal_built_with_cuda_support:value:true" in dockerfile
    assert "/tmp/sc27-openmpi-cuda-tcp/install/lib/libmpi.so.40" in dockerfile
    assert "rm -f /opt/python/lib/libmpi.so /opt/python/lib/libmpi.so.40" in dockerfile
    assert "OMPI_MCA_plm=isolated python -c" in dockerfile
    assert 'mpi_prefix=/tmp/sc27-openmpi-cuda-tcp/install' in launcher
