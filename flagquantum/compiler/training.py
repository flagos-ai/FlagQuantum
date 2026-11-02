"""Lazy generic Torch compilation at the approved training callable boundary."""

from __future__ import annotations

import dis
import inspect
import time
from collections.abc import Callable
from types import FunctionType
from typing import Any, ParamSpec, TypeVar, cast

import torch
from torch.fx import GraphModule

_P = ParamSpec("_P")
_R = TypeVar("_R")
TorchBackend = Callable[[GraphModule, list[Any]], Callable[..., Any]]


def _functional_tensor_operation(name: str) -> bool:
    # Admit Tensor arithmetic, not arbitrary Torch functions (file/storage
    # factories and device transfers are not pure scalar-objective work).
    if (
        not inspect.ismethoddescriptor(getattr(torch.Tensor, name, None))
        or name.startswith(("new_", "resize"))
        or name in {"to", "cpu", "cuda", "pin_memory"}
    ):
        return False
    public = getattr(torch, name, None)
    if callable(public) and not inspect.isbuiltin(public):
        return False
    packet = getattr(torch.ops.aten, name, None)
    if not isinstance(packet, torch._ops.OpOverloadPacket):
        return False
    overload_names = cast(Callable[[], list[str]], packet.overloads)()
    operations = [getattr(packet, overload) for overload in overload_names]
    functional = [
        operation for operation in operations if not operation._schema.is_mutable
    ]
    return bool(functional) and not any(
        tag in operation.tags
        for operation in functional
        for tag in (
            torch.Tag.nondeterministic_seeded,
            torch.Tag.nondeterministic_bitwise,
        )
    )


def _safe_objective_capture(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return (
            type(value) in {torch.Tensor, torch.nn.Parameter}
            and not value.requires_grad
        )
    if type(value) is tuple:
        return all(_safe_objective_capture(item) for item in value)
    return (
        value is torch
        or value is None
        or isinstance(
            value, (bool, int, float, complex, str, torch.dtype, torch.device)
        )
    )


def _objective_exclusion(objective: Callable[..., Any]) -> str:
    if inspect.isbuiltin(objective) and getattr(objective, "__module__", "") == "torch":
        return (
            ""
            if _functional_tensor_operation(objective.__name__)
            else "objective_operation"
        )
    if not isinstance(objective, FunctionType):
        return "opaque_objective"
    if objective.__code__.co_flags & (inspect.CO_GENERATOR | inspect.CO_COROUTINE):
        return "objective_control_flow"
    for instruction in dis.get_instructions(objective):
        if instruction.opname.startswith(
            ("DELETE_", "IMPORT_", "JUMP", "POP_JUMP", "INPLACE_")
        ) or instruction.opname in {
            "STORE_GLOBAL",
            "STORE_DEREF",
            "STORE_ATTR",
            "STORE_SUBSCR",
            "FOR_ITER",
            "MAKE_FUNCTION",
            "YIELD_VALUE",
            "GET_AWAITABLE",
            "CALL_FUNCTION_EX",
            "BUILD_MAP",
            "BUILD_CONST_KEY_MAP",
            "DICT_MERGE",
            "DICT_UPDATE",
        }:
            return "objective_side_effect_or_control_flow"
        if instruction.opname == "BINARY_OP" and instruction.argrepr.endswith("="):
            return "objective_inplace_operation"
    captured = inspect.getclosurevars(objective)
    if any(
        not _safe_objective_capture(value)
        for value in (*captured.globals.values(), *captured.nonlocals.values())
    ):
        return "objective_state_not_certified"
    if any(
        not _safe_objective_capture(value)
        for value in (
            *(objective.__defaults__ or ()),
            *(objective.__kwdefaults__ or {}).values(),
        )
    ):
        return "objective_defaults_not_certified"
    if set(captured.builtins) - {"abs", "len"}:
        return "objective_python_call_not_certified"
    if any(
        isinstance(value, tuple) and "out" in value
        for value in objective.__code__.co_consts
    ):
        return "objective_out_mutation"
    known = set(captured.globals) | set(captured.nonlocals) | set(captured.builtins)
    for name in set(objective.__code__.co_names) - known:
        if name in {"shape", "ndim", "dtype", "device"} or isinstance(
            getattr(torch, name, None), torch.dtype
        ):
            continue
        if name in {
            "item",
            "tolist",
            "numpy",
            "data",
        } or not _functional_tensor_operation(name):
            return f"objective_operation_not_certified:{name}"
    return ""


def compile_training_callable(
    step: Callable[_P, _R],
    *,
    backend: str | TorchBackend,
    diagnostics: dict[str, Any],
    objective: Callable[..., Any] | None = None,
) -> Callable[_P, _R]:
    """Return lazy Torch compilation without owning the training lifecycle.

    Args:
        step: Runtime-owned ordinary Torch training callable.
        backend: Existing Torch compiler backend selected by the caller.
        diagnostics: Caller-owned free runtime diagnostics, updated on capture.
        objective: Optional user objective whose functional Torch syntax must be
            certified before tracing. Opaque or stateful callables remain eager.

    Returns:
        A callable with the same arguments/result. Unavailable compilation uses
        the original callable with a reason. A backend declining one captured
        graph executes that graph eagerly at its original position; it never
        retries a whole step or replays an optimizer update.
    """
    diagnostics.update(
        status="lazy",
        backend=backend if isinstance(backend, str) else "provided_torch_compiler",
        captured_graph_count=0,
        compiled_graph_count=0,
        declined_graph_count=0,
        restarted_graph_count=0,
        backend_build_seconds=0.0,
        graph_break_reasons=[],
        whole_step_fused=False,
        reason="",
    )
    if objective is not None:
        reason = _objective_exclusion(objective)
        if reason:
            diagnostics.update(status="eager", reason=reason)
            return step
    compile_function = getattr(torch, "compile", None)
    if not callable(compile_function):
        diagnostics.update(status="eager", reason="torch_compile_unavailable")
        return step
    try:
        from torch._dynamo.backends.registry import lookup_backend
        from torch._dynamo.exc import RestartAnalysis, SkipFrame, Unsupported

        compiler = lookup_backend(backend)
    except Exception as error:
        diagnostics.update(
            status="eager", reason=f"compiler_unavailable:{type(error).__name__}"
        )
        return step

    def compile_graph(graph: GraphModule, examples: list[Any]) -> Callable[..., Any]:
        diagnostics["captured_graph_count"] += 1
        reason = getattr(graph, "compile_subgraph_reason", None)
        if reason is not None and getattr(reason, "graph_break", False):
            message = str(getattr(reason, "reason", "unknown"))
            if message not in diagnostics["graph_break_reasons"]:
                diagnostics["graph_break_reasons"].append(message)
        started = time.perf_counter()
        try:
            result = compiler(graph, examples)
        except RestartAnalysis:
            # Scalar specialization and related analysis restarts belong to
            # Dynamo; treating them as backend failures prevents optimization.
            diagnostics["restarted_graph_count"] += 1
            raise
        except (SkipFrame, Unsupported) as error:
            diagnostics["declined_graph_count"] += 1
            diagnostics["status"] = "eager_frame_declined"
            diagnostics["reason"] = f"torch_frame_declined:{type(error).__name__}"
            raise
        except Exception as error:
            diagnostics["declined_graph_count"] += 1
            diagnostics["reason"] = f"graph_compiler_declined:{type(error).__name__}"
            result = graph.forward
        else:
            diagnostics["compiled_graph_count"] += 1
        finally:
            diagnostics["backend_build_seconds"] += time.perf_counter() - started
        diagnostics["status"] = (
            "mixed_regions"
            if diagnostics["compiled_graph_count"]
            and diagnostics["declined_graph_count"]
            else (
                "compiled_regions"
                if diagnostics["compiled_graph_count"]
                else "eager_fx_regions"
            )
        )
        return result

    # Runtime supplies changing tensors and owns all eligibility/invalidation.
    # No quantum kernel, objective or optimizer executes while this is created.
    try:
        return cast(
            Callable[_P, _R],
            compile_function(
                step, backend=compile_graph, fullgraph=False, dynamic=False
            ),
        )
    except Exception as error:
        diagnostics.update(
            status="eager", reason=f"compiler_creation_failed:{type(error).__name__}"
        )
        return step
