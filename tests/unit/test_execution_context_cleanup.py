from contextlib import contextmanager

import pytest
import torch

import flagquantum as fq
from flagquantum.compute import flaggems
from flagquantum.core.runtime_config import get_runtime_config
from flagquantum.runtime.execution import run_native

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mode", ["statevector", "unsupported"])
def test_execution_restores_contexts_after_success_or_failure(monkeypatch, mode):
    original = get_runtime_config()
    events = []

    @contextmanager
    def backend(name, **options):
        assert name == "test_backend"
        assert options["include"] == ("mm",)
        assert get_runtime_config().complex_dtype == "complex128"
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    monkeypatch.setattr(flaggems, "operator_backend", backend)
    circuit = fq.Circuit(1).h(0)
    kwargs = dict(
        mode=mode,
        dtype=torch.complex128,
        operator_backend="test_backend",
        operator_backend_include=("mm",),
    )
    if mode == "unsupported":
        with pytest.raises(ValueError):
            run_native(circuit, **kwargs)
    else:
        run_native(circuit, **kwargs)
    assert events == ["enter", "exit"]
    assert get_runtime_config() == original
