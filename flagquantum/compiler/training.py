"""Lazy generic Torch compilation at the approved training callable boundary."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, cast

import torch
from torch.fx import GraphModule

_P = ParamSpec("_P")
_R = TypeVar("_R")
TorchBackend = Callable[[GraphModule, list[Any]], Callable[..., Any]]


def compile_training_callable(
    step: Callable[_P, _R],
    *,
    backend: str | TorchBackend,
    diagnostics: dict[str, Any],
) -> Callable[_P, _R]:
    """Return lazy Torch compilation without owning the training lifecycle.

    Args:
        step: Runtime-owned ordinary Torch training callable.
        backend: Existing Torch compiler backend selected by the caller.
        diagnostics: Caller-owned free runtime diagnostics, updated on capture.

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
        backend_build_seconds=0.0,
        graph_break_reasons=[],
        whole_step_fused=False,
        reason="",
    )
    compile_function = getattr(torch, "compile", None)
    if not callable(compile_function):
        diagnostics.update(status="eager", reason="torch_compile_unavailable")
        return step
    try:
        from torch._dynamo.backends.registry import lookup_backend

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
