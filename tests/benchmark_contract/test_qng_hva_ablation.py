"""Independent QNG geometry, trajectory and failure-accounting acceptance tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

np = pytest.importorskip("numpy")

from benchmarks.qng_hva_ablation import protocol_path, run_case, verify  # noqa: E402
from benchmarks.qng_hva_geometry import (  # noqa: E402
    Geometry,
    UnsupportedRouteError,
    Workload,
    independent_reference,
)

pytestmark = [pytest.mark.benchmark_contract, pytest.mark.integration]


@pytest.mark.parametrize("qubits,depth", [(4, 1), (4, 2), (6, 1)])
@pytest.mark.parametrize(
    "method", ["reverse_jacobian", "unfused_forward", "pennylane_metric"]
)
def test_geometry_and_update_match_independent_dense_reference(
    qubits: int, depth: int, method: str
) -> None:
    if method == "pennylane_metric" and importlib.util.find_spec("pennylane") is None:
        pytest.skip("optional PennyLane is not installed")
    workload = Workload(qubits, depth)
    parameters = (
        0.07
        * torch.randn(
            workload.count,
            generator=torch.Generator().manual_seed(29),
            dtype=torch.float64,
        )
    ).requires_grad_()
    result = verify(Geometry(method, workload), parameters, 0.001, 2e-6)
    assert result["passed"], result


def test_independent_oracle_has_analytic_dimer_and_ground_energies() -> None:
    workload = Workload(4, 2)
    state, _, metric, gradient = independent_reference(
        workload, np.zeros(workload.count)
    )
    assert np.vdot(state, workload.hamiltonian() @ state).real == pytest.approx(-6)
    assert np.linalg.eigvalsh(workload.hamiltonian())[0] == pytest.approx(
        -3 - 2 * np.sqrt(3)
    )
    assert np.allclose(metric, metric.T)
    assert np.linalg.eigvalsh(metric).min() > -1e-10
    assert np.isfinite(gradient).all()


def test_fused_cpu_is_unsupported_without_silent_fallback() -> None:
    with pytest.raises(UnsupportedRouteError, match="CUDA complex64"):
        Geometry("fused_forward", Workload(4, 1))


def test_wrong_tangent_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    workload = Workload(4, 1)
    parameters = torch.full(
        (workload.count,), 0.07, dtype=torch.float64, requires_grad=True
    )
    geometry = Geometry("unfused_forward", workload)
    original = geometry.tangent
    monkeypatch.setattr(
        geometry, "tangent", lambda p: (original(p)[0], original(p)[1] * 1.05)
    )
    assert not verify(geometry, parameters, 0.001, 2e-6)["passed"]


def test_complete_trajectories_match_and_total_includes_first_call() -> None:
    manifest = json.loads(protocol_path("cpu").read_text())
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        reverse = run_case(manifest, Workload(4, 1), "reverse_jacobian", 11, smoke=True)
        forward = run_case(manifest, Workload(4, 1), "unfused_forward", 11, smoke=True)
    finally:
        torch.set_num_threads(previous)
    for left, right in zip(reverse["trace"], forward["trace"], strict=True):
        assert left["parameters"] == pytest.approx(right["parameters"], abs=1e-9)
        assert left["direction"] == pytest.approx(right["direction"], abs=1e-8)
        assert left["energy"] == pytest.approx(right["energy"], abs=1e-10)
    for result in (reverse, forward):
        expected = sum(
            result[key]
            for key in (
                "backend_setup_seconds",
                "first_call_seconds",
                "warmup_seconds",
                "initial_assessment_seconds",
                "training_seconds",
            )
        )
        assert result["total_measured_seconds"] == pytest.approx(expected)
        assert result["first_call_seconds"] > 0
        assert len(result["trace"]) == 3
        assert result["process_peak_rss_bytes"] > 0
        if result["time_to_target_training_seconds"] is None:
            assert result["time_to_target_total_seconds"] is None


def test_timeout_failure_and_unsupported_are_retained_without_ratios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks import qng_hva_ablation as producer

    monkeypatch.setattr(
        producer,
        "identity",
        lambda path: {"code_version": "test", "producer_sha256": {}},
    )
    observed = []

    def worker(command, **kwargs):
        import subprocess
        from types import SimpleNamespace

        observed.append(command)
        path = Path(command[command.index("--output") + 1])
        if len(observed) == 1:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if len(observed) == 2:
            return SimpleNamespace(returncode=1)
        manifest = json.loads(producer.protocol_path("cpu").read_text())
        path.write_text(
            json.dumps(
                {
                    "source": producer.identity(None),
                    "protocol": manifest,
                    "status": "unsupported",
                    "error": "test route unavailable",
                }
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(producer, "runtime_metadata", lambda **kwargs: kwargs)
    monkeypatch.setattr(producer, "run_worker", worker)
    output = tmp_path / "campaign"
    assert producer.campaign(output, device="cpu", smoke=True) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert [c["status"] for c in summary["cases"]] == [
        "timeout",
        "failed",
        "unsupported",
        "unsupported",
    ]
    report = (output / "REPORT.md").read_text()
    assert "unavailable" in report
    assert "inf" not in report.lower()
    assert all(command[-1] == "--smoke" for command in observed)
    summary["cases"].pop()
    (output / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="all seeds"):
        producer.render_report(output)


def test_source_identity_binds_geometry_protocol_and_optional_versions() -> None:
    from benchmarks.qng_hva_ablation import identity

    source = identity(protocol_path("cpu"))
    assert "triton" in source["versions"]
    assert "benchmarks/qng_hva_geometry.py" in source["producer_sha256"]
    assert len(source["protocol_sha256"]) == 64


def test_worker_timeout_terminates_compiler_descendants(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    from benchmarks.qng_hva_ablation import run_worker

    if os.name != "posix":
        pytest.skip("POSIX process-group cleanup check")
    import fcntl
    import time

    lock_path, ready_path = tmp_path / "compiler.lock", tmp_path / "compiler.ready"
    child = (
        "import fcntl,sys,time; "
        "lock=open(sys.argv[1],'w'); fcntl.flock(lock,fcntl.LOCK_EX); "
        "open(sys.argv[2],'w').write('ready'); time.sleep(60)"
    )
    script = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],sys.argv[3]]); "
        "time.sleep(60)"
    )
    with (
        (tmp_path / "worker.log").open("w") as log,
        pytest.raises(subprocess.TimeoutExpired),
    ):
        run_worker(
            [sys.executable, "-c", script, child, str(lock_path), str(ready_path)],
            env=dict(os.environ),
            log=log,
            timeout=2,
        )
    assert ready_path.exists(), "descendant did not acquire its resource before timeout"
    deadline = time.monotonic() + 2
    with lock_path.open("w") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                assert (
                    time.monotonic() < deadline
                ), "compiler descendant survived timeout"
                time.sleep(0.01)


def test_report_rejects_truncated_completed_trajectory(tmp_path: Path) -> None:
    from dataclasses import asdict

    from benchmarks.qng_hva_ablation import METHODS, render_report

    protocol = json.loads(protocol_path("cpu").read_text())
    metadata = {
        "schema": "flagquantum.qng_hva_ablation.v1",
        "source": {"code_version": "test"},
        "protocol": protocol,
        "smoke": True,
    }
    cases = []
    for method in METHODS:
        name = f"{method}.json"
        result = {
            **metadata,
            "status": "unsupported",
            "error": "test unsupported route",
        }
        if method == "reverse_jacobian":
            result.update(
                status="completed",
                workload=asdict(Workload(4, 1)),
                completed_steps=3,
                parity={"passed": True},
                trace=[{"step": 1}],
            )
        (tmp_path / name).write_text(json.dumps(result))
        cases.append(
            {
                "qubits": 4,
                "depth": 1,
                "method": method,
                "seed": 11,
                "result": name,
                "case": method,
                "status": result["status"],
            }
        )
    (tmp_path / "freeze.json").write_text(json.dumps(metadata))
    (tmp_path / "summary.json").write_text(json.dumps({**metadata, "cases": cases}))
    with pytest.raises(ValueError, match="incomplete optimizer trajectory"):
        render_report(tmp_path)
