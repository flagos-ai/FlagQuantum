"""Empty compiled bindings are valid and remain isolated between contexts."""

from contextvars import copy_context

import pytest
import torch

from flagquantum.circuit import Circuit
from flagquantum.runtime.builder_compilation import BuilderBindings
from flagquantum.runtime.module import Module
from flagquantum.runtime.options import ExecutionOptions
from flagquantum.runtime.policy import RuntimePolicy

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mode", ["statevector", "mps", "tensor_network"])
def test_constant_builder_executes_and_reuses_its_compiled_circuit(mode: str) -> None:
    def fixed_circuit(parameters: torch.Tensor) -> Circuit:
        return Circuit(1).gate("h", 0)

    module = Module(
        fixed_circuit,
        1,
        init=torch.tensor([0.2]),
        policy=RuntimePolicy(execution_options=ExecutionOptions(mode=mode)),
    )
    first = module.execute()
    second = module.execute()
    assert isinstance(first.value, torch.Tensor)
    assert isinstance(second.value, torch.Tensor)
    torch.testing.assert_close(
        first.value, torch.zeros_like(first.value), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(second.value, first.value)
    assert second.runtime["builder_compile_count"] == 1
    assert second.runtime["builder_cache_hits"] == 1
    with pytest.raises(RuntimeError, match="slots are not bound"):
        module._builder_bindings.values()


def test_empty_binding_and_clear_do_not_modify_the_parent_context() -> None:
    bindings = BuilderBindings()
    with pytest.raises(RuntimeError, match="slots are not bound"):
        bindings.values()
    tensor = torch.tensor(0.2)
    bindings.bind((tensor,))
    child = copy_context()
    child.run(bindings.bind, ())
    assert child.run(bindings.values) == ()
    assert bindings.value(0) is tensor
    child.run(bindings.clear)
    with pytest.raises(RuntimeError, match="slots are not bound"):
        child.run(bindings.values)
    assert bindings.value(0) is tensor
    bindings.clear()
    with pytest.raises(RuntimeError, match="slots are not bound"):
        bindings.value(0)
