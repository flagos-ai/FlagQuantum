# SPDX-License-Identifier: Apache-2.0

"""Explicit per-function numerical suppliers for selected component graphs."""

from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Concatenate, Generic, ParamSpec

from ...core.ir.program import FunctionRecord
from .models import TensorNetworkNode

ProviderArgs = ParamSpec("ProviderArgs")
TensorProvider = Callable[
    Concatenate[FunctionRecord, ProviderArgs],
    Iterable[TensorNetworkNode | int],
]


class ComponentTensorProviders(Generic[ProviderArgs]):
    """Bind exact Core function names to explicitly chosen tensor suppliers.

    The same immutable mapping answers the Compiler's partial-expansion leaf
    predicate and dispatches selected Simulation calls. This corresponds to
    source per-object ``my_tensors`` dispatch without inspecting Python method
    names or introducing a process-global registry. Missing providers fail
    explicitly; callback results and exceptions are not transformed.
    """

    def __init__(self, providers: Mapping[str, TensorProvider[ProviderArgs]]) -> None:
        copied = dict(providers)
        if any(not isinstance(name, str) or not name for name in copied):
            raise ValueError("tensor provider keys must be non-empty function names")
        if any(not callable(provider) for provider in copied.values()):
            raise TypeError("tensor providers must be callable")
        self._providers = MappingProxyType(copied)

    def has_custom_tensor(self, function_name: str) -> bool:
        """Return whether this exact typed function has a registered supplier."""
        return function_name in self._providers

    def supply(
        self,
        function: FunctionRecord,
        *args: ProviderArgs.args,
        **kwargs: ProviderArgs.kwargs,
    ) -> Iterable[TensorNetworkNode | int]:
        """Dispatch one selected function to its supplier without fallback."""
        try:
            provider = self._providers[function.name]
        except KeyError as exc:
            raise ValueError(
                f"no tensor provider registered for function {function.name!r}"
            ) from exc
        return provider(function, *args, **kwargs)
