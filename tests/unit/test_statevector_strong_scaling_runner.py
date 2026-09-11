"""The registered strong-scaling entry point dispatches without eager execution."""

import pytest

from flagquantum.benchmarking import statevector_strong_scaling
from flagquantum.benchmarking.registry import resolve

pytestmark = pytest.mark.unit


def test_strong_scaling_runner_dispatches_and_preserves_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def main() -> int:
        calls.append("strong_scaling")
        return 3

    monkeypatch.setattr(statevector_strong_scaling, "main", main)
    runner = resolve("statevector_strong_scaling")
    assert calls == []
    assert runner() == 3
    assert calls == ["strong_scaling"]
