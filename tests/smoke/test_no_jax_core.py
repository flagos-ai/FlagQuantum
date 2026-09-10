"""Core FlagQuantum import and autograd must not import optional JAX."""

import json
import subprocess
import sys

import pytest


@pytest.mark.smoke
def test_core_import_and_autograd_do_not_load_jax() -> None:
    # Other tests may already have loaded JAX; inspect a fresh interpreter.
    script = """
import json
import sys

sys.path[:] = json.loads(sys.argv[1])
assert not any(name == "jax" or name.startswith("jax.") for name in sys.modules)

import torch
import flagquantum as fq

theta = torch.tensor(0.3, requires_grad=True)
circuit = fq.Circuit(2).h(0).cx(0, 1).rx(0, theta=theta)
value = circuit.expectation_z(0).sum()
value.backward()

assert theta.grad is not None
assert not any(name == "jax" or name.startswith("jax.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, json.dumps(sys.path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
