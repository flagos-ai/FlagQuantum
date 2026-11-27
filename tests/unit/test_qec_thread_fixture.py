"""QEC's scientific thread context must not leak into other CI selections."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_qec_fixture_restores_threads_after_success_and_failure(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    qec = tmp_path / "qec"
    qec.mkdir()
    shutil.copyfile(root / "tests/qec/conftest.py", qec / "conftest.py")
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (qec / "test_context.py").write_text(
        "import torch\n"
        "assert torch.get_num_threads() == 3\n"
        "def test_success():\n"
        "    assert torch.get_num_threads() == 1\n"
        "    torch.set_num_threads(2)\n"
        "def test_failure():\n"
        "    assert torch.get_num_threads() == 1\n"
        "    torch.set_num_threads(2)\n"
        "    raise AssertionError('deliberate thread-context failure')\n"
    )
    (tmp_path / "test_after.py").write_text(
        "import torch\n"
        "def test_original_allocation_is_restored():\n"
        "    assert torch.get_num_threads() == 3\n"
    )
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="3",
        MKL_NUM_THREADS="3",
        OPENBLAS_NUM_THREADS="3",
        PYTEST_ADDOPTS="",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "qec", "test_after.py"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "deliberate thread-context failure" in output
    assert "1 failed, 2 passed" in output
