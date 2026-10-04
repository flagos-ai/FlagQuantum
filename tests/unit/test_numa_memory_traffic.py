from pathlib import Path

import pytest

from flagquantum.benchmarking import numa_memory_traffic as numa

pytestmark = pytest.mark.unit


def test_event_config_decodes_intel_cas_event(tmp_path: Path) -> None:
    event = tmp_path / "cas_count_read"
    event.write_text("event=0x04,umask=0x03\n")
    assert numa._event_config(event) == 0x0304


def test_uncore_sources_keep_only_numbered_imcs_with_cas_events(
    tmp_path: Path,
) -> None:
    for name in ("uncore_imc_2", "uncore_imc_0", "uncore_imc_free_running_0"):
        events = tmp_path / name / "events"
        events.mkdir(parents=True)
        (events / "cas_count_read").write_text("event=0x04,umask=0x03\n")
        (events / "cas_count_write").write_text("event=0x04,umask=0x0c\n")
    incomplete = tmp_path / "uncore_imc_1" / "events"
    incomplete.mkdir(parents=True)
    (incomplete / "cas_count_read").write_text("event=0x04,umask=0x03\n")

    assert [path.name for path in numa._uncore_sources(tmp_path)] == [
        "uncore_imc_0",
        "uncore_imc_2",
    ]


def test_summary_records_samples_and_relative_mad() -> None:
    result = numa._summary((1.0, 2.0, 3.0))
    assert result["median_seconds"] == 2.0
    assert result["relative_median_absolute_deviation"] == 0.5
    with pytest.raises(ValueError, match="positive"):
        numa._summary(())


def test_annotate_cases_separates_native_and_external_tolerances() -> None:
    def case(engine: str, policy: str, value: float) -> dict[str, object]:
        return {
            "workload": "hardware_efficient_vqe",
            "engine": engine,
            "policy": policy,
            "value": value,
            "gradient": [value],
            "ready": {"native_cpu_adjoint": True},
            "timing": {
                "total": {
                    "median_seconds": 1.0,
                    "relative_median_absolute_deviation": 0.01,
                }
            },
            "traffic": {"bytes_per_call": 1024.0},
        }

    cases = [
        case("flagquantum_adjoint", "default", 1.0),
        case("flagquantum_adjoint", "interleave", 1.0 + 5e-10),
        case("pennylane_lightning_adjoint", "default", 1.0 + 5e-7),
    ]
    assessment = numa._annotate_cases(cases)
    assert assessment["passed"] is True
    assert cases[1]["correctness"]["tolerance"] == 1e-9
    assert cases[2]["correctness"]["tolerance"] == 1e-6


def test_run_benchmark_applies_external_engine_only_to_default_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(numa, "_require_linux_x86_64", lambda: None)

    def fake_run_case(**kwargs: object) -> dict[str, object]:
        calls.append((str(kwargs["engine"]), str(kwargs["policy"])))
        return {
            **kwargs,
            "value": 1.0,
            "gradient": [2.0],
            "ready": {"native_cpu_adjoint": True},
            "timing": {
                "total": {
                    "median_seconds": 1.0,
                    "relative_median_absolute_deviation": 0.01,
                }
            },
            "traffic": {"bytes_per_call": 1024.0},
        }

    monkeypatch.setattr(numa, "_run_case", fake_run_case)
    monkeypatch.setattr(
        numa,
        "runtime_metadata",
        lambda **kwargs: {**kwargs, "python": "test", "hostname": "private"},
    )
    payload = numa.run_benchmark(
        workloads=("hardware_efficient_vqe",),
        engines=("flagquantum_adjoint", "pennylane_lightning_adjoint"),
        policies=("default", "bind", "interleave"),
        nodes=(0, 1),
        cpus=(0, 1),
        socket_cpus=(0, 1),
        n_qubits=2,
        layers=1,
        threads=2,
        warmup=0,
        calls=3,
    )

    assert calls == [
        ("flagquantum_adjoint", "default"),
        ("flagquantum_adjoint", "bind"),
        ("flagquantum_adjoint", "interleave"),
        ("pennylane_lightning_adjoint", "default"),
    ]
    assert payload["schema"] == numa.SCHEMA
    assert payload["hostname"] == "redacted"
    assert payload["assessment"]["passed"] is True


def test_render_markdown_names_external_framework_and_reproduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(numa, "_require_linux_x86_64", lambda: None)

    def fake_run_case(**kwargs: object) -> dict[str, object]:
        return {
            **kwargs,
            "value": 1.0,
            "gradient": [2.0],
            "ready": {"native_cpu_adjoint": True},
            "timing": {
                "total": {
                    "median_seconds": 1.0,
                    "relative_median_absolute_deviation": 0.01,
                }
            },
            "traffic": {"bytes_per_call": 2**30},
        }

    monkeypatch.setattr(numa, "_run_case", fake_run_case)
    payload = numa.run_benchmark(
        workloads=("hardware_efficient_vqe",),
        engines=("flagquantum_adjoint", "pennylane_lightning_adjoint"),
        policies=("default",),
        nodes=(0,),
        cpus=(0,),
        socket_cpus=(0,),
        n_qubits=2,
        layers=1,
        threads=1,
        warmup=0,
        calls=3,
        source_revision="abc123",
    )
    report = numa.render_markdown(payload, artifact_name="result.json")
    assert "PennyLane Lightning adjoint" in report
    assert "1.000 ms" not in report
    assert "1000.000 ms" in report
    assert "--source-revision abc123" in report
    assert "result.json" in report


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("workloads", (), "workloads"),
        ("engines", (), "engines"),
        ("policies", (), "policies"),
    ),
)
def test_run_benchmark_rejects_empty_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: tuple[()],
    message: str,
) -> None:
    monkeypatch.setattr(numa, "_require_linux_x86_64", lambda: None)
    arguments = {
        "workloads": ("hardware_efficient_vqe",),
        "engines": ("flagquantum_adjoint",),
        "policies": ("default",),
        "nodes": (0,),
        "cpus": (0,),
        "socket_cpus": (0,),
        "n_qubits": 2,
        "layers": 1,
        "threads": 1,
        "warmup": 0,
        "calls": 3,
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        numa.run_benchmark(**arguments)
