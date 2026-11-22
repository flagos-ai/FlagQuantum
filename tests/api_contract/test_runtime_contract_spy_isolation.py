"""Contract spies must not escape into subsequently imported runtime modules."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("option", ["allow_approximate", "allow_backend_fallback"])
@pytest.mark.parametrize("without_numpy", [False, True])
def test_cold_rejection_preserves_later_mps_execution(
    option: str, without_numpy: bool
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

if sys.argv[2] == "True":
    sys.modules["numpy"] = None

import pytest
import torch

assert "flagquantum.runtime.execution" not in sys.modules
from tests.api_contract import test_execution_runtime_inputs as contract

with pytest.MonkeyPatch.context() as patch:
    contract.test_unsupported_routes_fail_before_planner(
        contract.fq.ExecutionOptions(**{sys.argv[1]: True}), patch
    )

from flagquantum.runtime import execution, planner

assert execution.build_plan is planner.plan_advanced, "contract spy escaped into runtime planning"
import flagquantum.runtime as runtime

theta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
circuit = contract.fq.Circuit(2, dtype=torch.complex128).ry(0, theta).cx(0, 1)
state = runtime.run_native(circuit, mode="mps", max_bond=4, cutoff=0.0)
value = state.expectation_ps(z=(1,)).sum()
gradient, = torch.autograd.grad(value, theta)
torch.testing.assert_close(value, theta.cos(), atol=1e-10, rtol=0)
torch.testing.assert_close(gradient, -theta.sin(), atol=1e-10, rtol=0)
""",
            option,
            str(without_numpy),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
