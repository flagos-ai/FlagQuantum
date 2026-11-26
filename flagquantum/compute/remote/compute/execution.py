"""Adapt the stable run journey to a resident Jiuding workspace."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from flagquantum._import_compat import alias_module as _alias_module

from .jiuding import JiudingClient

if TYPE_CHECKING:
    from flagquantum.core.measurement import OutputRequest
    from flagquantum.core.noise import NoiseModel
    from flagquantum.runtime.options import ExecutionOptions
    from flagquantum.runtime.results import ExecutionResult

_CLIENTS: dict[str, JiudingClient] = {}


def execute_jiuding(
    program: Any,
    *,
    target: str,
    outputs: OutputRequest | Sequence[OutputRequest] | None = None,
    shots: int | None = None,
    options: ExecutionOptions | None = None,
    noise_model: NoiseModel | None = None,
    compiler: str | None = None,
    target_qubits: Sequence[int] | None = None,
    name: str | None = None,
) -> ExecutionResult:
    """Validate the public controls and execute through a resident workspace."""

    if compiler is not None:
        raise TypeError("Jiuding workspace execution does not accept compiler")
    if target_qubits is not None:
        raise TypeError("Jiuding workspace execution does not accept target_qubits")
    if options is not None or noise_model is not None:
        raise TypeError(
            "options and noise_model are not yet supported by Jiuding workspace execution"
        )
    if name is not None:
        raise TypeError("Jiuding workspace execution does not accept name")
    from flagquantum.runtime.execution_plan import ExecutionPlan

    if isinstance(program, ExecutionPlan):
        raise TypeError(
            "Jiuding workspace execution requires a Circuit or CircuitIR, "
            "not an ExecutionPlan"
        )

    workspace = os.environ.get("JIUDING_WORKSPACE", "").strip()
    client = _CLIENTS.get(workspace)
    if client is None:
        client = JiudingClient(workspace=workspace or None)
        _CLIENTS[workspace] = client
    return client.run(program, target=target, outputs=outputs, shots=shots)


__all__: tuple[str, ...] = ()


# Preserve historical import identities for pickles and public introspection.
for _identity_export in (execute_jiuding,):
    _identity_export.__module__ = "flagquantum.remote.compute.execution"
del _identity_export

_alias_module("flagquantum.remote.compute.execution", __name__)
