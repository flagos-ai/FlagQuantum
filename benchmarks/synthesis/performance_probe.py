"""Benchmark-only nested Python spans and externally observed process RSS."""

from __future__ import annotations

import inspect
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from types import FrameType
from typing import Literal

Stage = Literal["C1", "C2", "C3", "C4", "C5", "C6"]


def stage_of(module: str, function: str) -> Stage | None:
    if module.startswith("flagquantum.compiler._construction."):
        leaf = module.rsplit(".", 1)[-1]
        if leaf in {"arithmetic_rewrites", "ranges", "dependencies", "lifetimes"}:
            return "C2"
        if leaf in {"lowering", "register_lowering"}:
            return "C3"
        if leaf == "target":
            return "C4"
        if leaf == "selection":
            return "C5"
        if leaf.startswith("frontend") or leaf in {
            "bindings",
            "register_profile",
            "profile",
            "register_expansion",
        }:
            return "C1"
    if module.startswith(
        ("flagquantum.ecosystem._qmod", "flagquantum.compiler._hybrid.verifier")
    ):
        return "C1"
    if module == "flagquantum.compiler.resource_estimation":
        return "C5"
    if module.startswith(
        (
            "flagquantum.services",
            "flagquantum.runtime.execution_plan",
            "flagquantum.runtime.planner",
        )
    ):
        return "C6"
    if module == "flagquantum.core.ir" and function in {
        "__init__",
        "__post_init__",
        "to_dict",
        "from_dict",
        "to_json",
        "from_json",
        "content_hash",
        "validate",
        "_encode_value",
        "_decode_value",
    }:
        return "C6"
    return None


@dataclass
class Span:
    id: int
    parent: int | None
    function: str
    stage: Stage
    context: tuple[tuple[str, str], ...]
    calls: int = 0
    inclusive_ns: int = 0
    exclusive_ns: int = 0
    argument_items: int = 0
    argument_text_bytes: int = 0
    returned_text_bytes: int = 0


@dataclass
class Active:
    frame: FrameType
    span: Span
    start: int
    child_ns: int = 0


class Probe:
    """Aggregate call-tree edges, without retaining individual instruction events.

    This deliberately intrusive diagnostic is never the primary E0 timer. Generator
    resumptions are excluded; their work remains charged to the enclosing span.
    """

    def __init__(self, emit_stage: Callable[[str], None] | None = None) -> None:
        self.spans: dict[tuple[int | None, str, tuple[tuple[str, str], ...]], Span] = {}
        self.stack: list[Active] = []
        self.emit_stage = emit_stage
        self.last_emit_ns = 0

    def __enter__(self) -> Probe:
        if sys.getprofile() is not None:
            raise RuntimeError("PF-00 requires a worker without another profiler")
        sys.setprofile(self.event)
        return self

    def __exit__(self, *args: object) -> None:
        sys.setprofile(None)
        if self.stack:
            raise RuntimeError("incomplete diagnostic call stack")

    def event(self, frame: FrameType, event: str, arg: object) -> None:
        if event not in {"call", "return"}:
            return
        if event == "return":
            if not self.stack or self.stack[-1].frame is not frame:
                return
            active = self.stack.pop()
            elapsed = time.perf_counter_ns() - active.start
            active.span.inclusive_ns += elapsed
            active.span.exclusive_ns += elapsed - active.child_ns
            if isinstance(arg, (str, bytes)):
                active.span.returned_text_bytes += len(
                    arg.encode() if isinstance(arg, str) else arg
                )
            if self.stack:
                self.stack[-1].child_ns += elapsed
            return
        if frame.f_code.co_flags & (inspect.CO_GENERATOR | inspect.CO_COROUTINE):
            return
        module = str(frame.f_globals.get("__name__", ""))
        stage = stage_of(module, frame.f_code.co_name)
        if stage is None:
            return
        function = (
            module
            + "."
            + str(getattr(frame.f_code, "co_qualname", frame.f_code.co_name))
        )
        # Generated dataclass constructors otherwise share the unhelpful name __init__.
        owner = frame.f_locals.get("self")
        if frame.f_code.co_name == "__init__" and owner is not None:
            function = module + "." + type(owner).__name__ + ".__init__"
        parent = self.stack[-1].span.id if self.stack else None
        context = tuple(
            (name, str(frame.f_locals[name]))
            for name in ("strategy", "scheduling", "implementation", "output_mode")
            if name in frame.f_locals and isinstance(frame.f_locals[name], (str, bool))
        )
        key = (parent, function, context)
        if key not in self.spans:
            if len(self.spans) >= 20000:
                raise RuntimeError("diagnostic exceeds 20000 aggregated call paths")
            self.spans[key] = Span(len(self.spans), parent, function, stage, context)
        span = self.spans[key]
        span.calls += 1
        for name, value in frame.f_locals.items():
            if name == "self" and type(value).__module__ == "flagquantum.core.ir":
                instructions = getattr(value, "instructions", ())
                span.argument_items += len(instructions)
            elif isinstance(value, (tuple, list, dict)):
                span.argument_items += len(value)
            elif isinstance(value, (str, bytes)):
                span.argument_text_bytes += len(
                    value.encode() if isinstance(value, str) else value
                )
        now = time.perf_counter_ns()
        if self.emit_stage is not None and now - self.last_emit_ns >= 10_000_000:
            self.emit_stage(stage)
            self.last_emit_ns = now
        self.stack.append(Active(frame, span, time.perf_counter_ns()))

    def report(self, elapsed_ns: int) -> dict[str, object]:
        exclusive = sum(span.exclusive_ns for span in self.spans.values())
        if exclusive > elapsed_ns or any(
            s.exclusive_ns < 0 for s in self.spans.values()
        ):
            raise ValueError("invalid nested span accounting")
        return {
            "clock": "perf_counter_ns",
            "scope": "separate intrusive diagnostic; not primary E0",
            "spans": [asdict(span) for span in self.spans.values()],
            "unattributed_ns": elapsed_ns - exclusive,
            "exclusive_sum_ns": exclusive,
            "scale_scope": "direct argument lengths and string/bytes lengths (including JSON when present), not unique nodes, allocated or serialized nested-object bytes; text uses UTF-8",
            "generator_scope": "generator resumptions excluded; time remains in enclosing spans",
        }


def process_rss(pid: int) -> dict[str, object]:
    """Sample simultaneous parent/tree RSS; ps reports KiB on Linux and macOS.

    No sum of independent process high-water marks is used. Shared pages may be
    counted twice. Failure is explicit rather than replaced with zero.
    """
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss="],
        capture_output=True,
        text=True,
        check=True,
        timeout=2,
    )
    rows = [
        tuple(map(int, line.split()))
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    tree = {pid}
    for _ in rows:
        expanded = tree | {child for child, parent, _ in rows if parent in tree}
        if expanded == tree:
            break
        tree = expanded
    parent_rss = next((rss for child, _, rss in rows if child == pid), None)
    if parent_rss is None:
        raise ProcessLookupError(pid)
    return {
        "parent_rss_bytes": parent_rss * 1024,
        "process_tree_rss_sum_bytes": sum(
            rss * 1024 for child, _, rss in rows if child in tree
        ),
        "process_count": sum(child in tree for child, _, _ in rows),
        "raw_parent_rss": parent_rss,
        "raw_unit": "KiB",
    }
