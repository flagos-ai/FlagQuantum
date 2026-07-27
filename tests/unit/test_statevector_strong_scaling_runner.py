from benchmarks.runners.registry import resolve


def test_strong_scaling_runner_is_registered_and_callable():
    runner = resolve("statevector_strong_scaling")
    assert callable(runner)
    assert runner.__module__ == "benchmarks.runners.statevector_strong_scaling"
