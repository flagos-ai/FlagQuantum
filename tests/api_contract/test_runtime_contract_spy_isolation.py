"""Contract spies must not escape into subsequently imported runtime modules."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("option", ["allow_approximate", "allow_backend_fallback"])
def test_cold_rejection_preserves_later_mps_execution(option: str) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import pytest

assert "flagquantum.runtime.execution" not in sys.modules
from tests.api_contract import test_execution_runtime_inputs as contract

with pytest.MonkeyPatch.context() as patch:
    contract.test_unsupported_routes_fail_before_planner(
        contract.fq.ExecutionOptions(**{sys.argv[1]: True}), patch
    )

from flagquantum.runtime import execution, planner

assert execution.build_plan is planner.plan_advanced, "contract spy escaped into runtime planning"
from tests.benchmark_contract import test_mps_training_recovery as recovery

recovery.test_truncated_derivative_error_is_separate_from_sharding()
""",
            option,
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
