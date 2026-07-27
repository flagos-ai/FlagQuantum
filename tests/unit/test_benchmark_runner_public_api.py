import benchmarks.runners as runners


def test_runner_package_exposes_stable_registry_api():
    assert "statevector_training_scaling" in runners.names()
    assert callable(runners.resolve("environment_probe"))
