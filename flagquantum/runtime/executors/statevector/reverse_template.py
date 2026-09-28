"""Reusable parameter-free IR templates for statevector adjoint execution."""

from __future__ import annotations

import os
import weakref
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from threading import Lock
from typing import Any

import torch

from ....core.ir import CircuitIR
from ....core.parameters import ParameterExpression
from ...builder_compilation import detached_ir_snapshot

_TEMPLATE_CACHE_CAPACITY = 128


@dataclass(frozen=True)
class _TemplateCacheEntry:
    source: weakref.ReferenceType[CircuitIR]
    slots: tuple[tuple[int, str, int], ...]
    template: CircuitIR


_TEMPLATE_CACHE: OrderedDict[int, _TemplateCacheEntry] = OrderedDict()
_TEMPLATE_CACHE_LOCK = Lock()


def _contains_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, ParameterExpression):
        return any(_contains_tensor(item) for item in value.args)
    if isinstance(value, Mapping):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def _cache_eligible(ir: CircuitIR, slots: Sequence[tuple[int, str, int]]) -> bool:
    """Cache only IRs whose tensors are exactly the direct trainable slots."""

    slot_keys = {(instruction_index, name) for instruction_index, name, _ in slots}
    for instruction_index, instruction in enumerate(ir.instructions):
        for name, value in instruction.params.items():
            if (instruction_index, str(name)) not in slot_keys and _contains_tensor(
                value
            ):
                return False
        if _contains_tensor(instruction.matrix) or _contains_tensor(
            instruction.metadata
        ):
            return False
    if _contains_tensor(ir.metadata):
        return False
    if any(
        _contains_tensor(observable.coefficient)
        or _contains_tensor(observable.metadata)
        for observable in ir.observables
    ):
        return False
    return not any(
        _contains_tensor(measurement.metadata) for measurement in ir.measurements
    )


def _parameter_free_snapshot(
    ir: CircuitIR, slots: Sequence[tuple[int, str, int]]
) -> CircuitIR:
    snapshot = detached_ir_snapshot(ir)
    instructions = list(snapshot.instructions)
    params_by_instruction: dict[int, dict[str, Any]] = {}
    for instruction_index, name, _ in slots:
        params = params_by_instruction.setdefault(
            instruction_index, dict(instructions[instruction_index].params)
        )
        params[name] = 0.0
    for instruction_index, params in params_by_instruction.items():
        instructions[instruction_index] = replace(
            instructions[instruction_index], params=params
        )
    return replace(snapshot, instructions=tuple(instructions))


def prepare_adjoint_ir_template(
    ir: CircuitIR, slots: Sequence[tuple[int, str, int]]
) -> CircuitIR:
    """Return a detached parameter-free template, reusing safe circuit identities."""

    normalized_slots = tuple(slots)
    cache_enabled = os.getenv(
        "FQ_STATEVECTOR_ADJOINT_TEMPLATE_CACHE", "1"
    ).strip().lower() not in {"0", "false", "off", "no"}
    eligible = cache_enabled and _cache_eligible(ir, normalized_slots)
    cache_key = id(ir)
    if eligible:
        with _TEMPLATE_CACHE_LOCK:
            cached = _TEMPLATE_CACHE.get(cache_key)
            if (
                cached is not None
                and cached.source() is ir
                and cached.slots == normalized_slots
            ):
                _TEMPLATE_CACHE.move_to_end(cache_key)
                return cached.template

    template = _parameter_free_snapshot(ir, normalized_slots)
    if eligible:
        with _TEMPLATE_CACHE_LOCK:
            _TEMPLATE_CACHE[cache_key] = _TemplateCacheEntry(
                source=weakref.ref(ir),
                slots=normalized_slots,
                template=template,
            )
            _TEMPLATE_CACHE.move_to_end(cache_key)
            while len(_TEMPLATE_CACHE) > _TEMPLATE_CACHE_CAPACITY:
                _TEMPLATE_CACHE.popitem(last=False)
    return template


def _clear_adjoint_ir_template_cache() -> None:
    """Clear process-local templates for deterministic tests."""

    with _TEMPLATE_CACHE_LOCK:
        _TEMPLATE_CACHE.clear()


__all__ = ("prepare_adjoint_ir_template",)
