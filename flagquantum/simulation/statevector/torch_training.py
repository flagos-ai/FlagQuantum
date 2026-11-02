"""Reusable Torch numerical graphs derived from canonical local Core IR."""

from __future__ import annotations

import operator
from collections.abc import Mapping

import torch
from torch.fx import GraphModule

from ...core.ir import CircuitIR
from ...errors import CapabilityError


def capture_training_value(
    program: CircuitIR,
    bindings: Mapping[str, torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    observable_qubits: tuple[int, ...],
    sum_observables: bool,
) -> GraphModule:
    """Capture existing statevector numerics, leaving lifecycle checks outside.

    Args:
        program: Canonical symbolic IR validated by its Runtime caller.
        bindings: Current dynamic angle examples in stable name order.
        batch_size: Existing explicitly declared Circuit batch.
        device: Resolved local CPU device; no device selection or transfer.
        dtype: Existing explicit complex precision.
        observable_qubits: Existing Z readout qubits.
        sum_observables: Whether the caller requests their scalar sum.

    Returns:
        An ephemeral Torch numerical graph with one positional input per binding.
        It is a backend implementation, not another quantum IR.

    Raises:
        CapabilityError: This initial CPU Z-readout capture is not applicable.
    """
    if device.type != "cpu":
        raise CapabilityError("Torch training value capture is CPU-only")
    if (
        program.observables
        or program.measurements
        or any(instruction.matrix is not None for instruction in program.instructions)
    ):
        raise CapabilityError("Torch training value capture requires plain gate IR")
    if not bindings:
        raise CapabilityError("Torch training value capture needs dynamic bindings")
    if any(
        value.device != device or not value.is_floating_point()
        for value in bindings.values()
    ):
        raise CapabilityError(
            "Torch training angle bindings must be real on the requested device"
        )

    from torch.fx.experimental.proxy_tensor import make_fx

    from ...circuit import Circuit

    prototype = Circuit.from_ir(program, bsz=batch_size, device=device, dtype=dtype)
    names = tuple(bindings)
    if set(names) != set(prototype.parameter_names):
        raise CapabilityError("Torch training capture needs every canonical binding")

    def numerical(*angles: torch.Tensor) -> torch.Tensor:
        circuit = prototype.bind_parameters(dict(zip(names, angles, strict=True)))
        values = circuit.expectation_z(observable_qubits)
        if sum_observables:
            return values.sum(dim=-1)
        return values if len(observable_qubits) > 1 else values[..., 0]

    # Frozen features still change between calls. Differentiable private examples
    # select functional tensor numerics rather than inference-only matrix caches.
    # Actual calls retain their real grad mode and requires_grad flags.
    with torch.enable_grad():
        samples = tuple(
            value.detach().clone().requires_grad_(True) for value in bindings.values()
        )
        graph = make_fx(numerical, decomposition_table={})(*samples)
    for node in graph.graph.nodes:
        if node.op in {"placeholder", "output", "get_attr"}:
            continue
        if node.op != "call_function" or (
            node.target is not operator.getitem
            and (
                not isinstance(node.target, torch._ops.OpOverload)
                or node.target.namespace != "aten"
            )
        ):
            raise CapabilityError(
                f"Torch training capture cannot preserve {node.target}"
            )
    return graph
