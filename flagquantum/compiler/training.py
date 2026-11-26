"""Compatibility imports for :mod:`flagquantum.compiler._torch.training`."""

from flagquantum.compiler.frontends.torch._training import (
    TorchBackend as TorchBackend,
)
from flagquantum.compiler.frontends.torch._training import (
    compile_training_callable as compile_training_callable,
)
