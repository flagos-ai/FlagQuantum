from pathlib import Path

ROOT = Path(__file__).parents[2]
RUNNERS = ROOT / "benchmarks" / "runners"


def test_runner_namespace_has_no_research_dependency():
    forbidden = ("benchmarks.research", "plot_", "/research/")
    for path in RUNNERS.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path


def test_research_boundary_is_documented():
    assert (ROOT / "benchmarks" / "research" / "README.md").is_file()
    assert (RUNNERS / "README.md").is_file()


def test_runner_migration_log_is_present():
    log = RUNNERS / "MIGRATION_LOG.md"
    assert log.is_file()
    text = log.read_text(encoding="utf-8")
    for name in (
        "environment_probe",
        "statevector_strong_scaling",
        "statevector_weak_scaling",
        "statevector_training_scaling",
    ):
        assert name in text
