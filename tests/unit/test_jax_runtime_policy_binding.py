"""Keep lazy JAX planning independent of expired policy replacements."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.distributed_cpu]
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("backward_backend", ("pmap", "shard_map"))
def test_cold_jax_planning_restores_the_canonical_policy(
    backward_backend: str,
) -> None:
    # A subprocess preserves the cold import even when other tests preload JAX.
    code = """
import importlib
import sys
from dataclasses import replace
from unittest.mock import patch

import flagquantum as fq
from flagquantum.runtime.distributed import backend_policy

name = "flagquantum.runtime.executors.jax.runtime_environment"
assert name not in sys.modules
production = backend_policy.resolve_distributed_backend_policy(profile="production")
temporary = replace(production, profile="development", local_world_size=2)
with patch.object(
    backend_policy, "resolve_distributed_backend_policy", return_value=temporary
) as resolver:
    runtime = importlib.import_module(name)
    assert runtime._resolve_policy(distributed_profile="production") is temporary
    resolver.assert_called_once_with(profile="production")

assert runtime._resolve_policy(distributed_profile="production") == production
with patch.object(
    backend_policy,
    "resolve_distributed_backend_policy",
    side_effect=AssertionError("An explicit policy must bypass resolution"),
):
    assert runtime._resolve_policy(distributed_backend_policy=production) is production
    overridden = runtime._resolve_policy(
        distributed_backend_policy=production,
        jax_backend="pmap",
        torch_backend="gloo",
    )
assert overridden == replace(
    production,
    jax_backend="pmap",
    torch_backend="gloo",
    source={
        **production.source,
        "runtime_jax_backend": "pmap",
        "runtime_torch_backend": "gloo",
    },
)
assert "runtime_jax_backend" not in production.source
assert "runtime_torch_backend" not in production.source

backward = sys.argv[1]
circuit = fq.Circuit(4).ry(0, theta=0.2).rxx(2, 3, theta=0.1).cx(2, 3)
selection = circuit.runtime_plan(
    world_size=4,
    local_world_size=2 if backward == "pmap" else 4,
    prefer_jax=True,
    require_gradients=True,
    distributed_profile="production",
    jax_backward_backend=backward,
    assume_jax_devices_ready=True,
)
candidates = {candidate.mode: candidate.summary() for candidate in selection.candidates}
candidate = candidates["jax_sharded_statevector"]
assert candidate["available"] is False
assert candidate["scalability_claim_allowed"] is False
assert selection.recommended_mode != "jax_sharded_statevector"
assert "statevector_training_not_claimable" in candidate["blockers"]
assert "optimizer_update_semantics_not_measured" in candidate["blockers"]
assert candidate["execution_plan"]["runtime_tier"] == f"planned_{backward}_execution"
assert candidate["claimable_production_training"] is False
assert candidate["statevector_training_claimability_status"] == "preflight_only"
if backward == "shard_map":
    assert (
        "shard_map_statevector_multi_sharded_wire_transport_pending:all_to_all"
        in candidate["blockers"]
    )
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, backward_backend],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
