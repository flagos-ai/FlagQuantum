import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_scaling_adapters_support_direct_help_entrypoint():
    for name in (
        "statevector_weak_scaling.py",
        "statevector_strong_scaling.py",
        "statevector_training_scaling.py",
    ):
        result = subprocess.run(
            [sys.executable, f"benchmarks/runners/{name}", "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "--json-output" in result.stdout
