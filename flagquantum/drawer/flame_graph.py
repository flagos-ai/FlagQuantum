# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 flame_graph._populate_flame_graph_data.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/flame_graph.py

"""Stream declared T-cost paths in the folded-stack FlameGraph format."""

import shutil
import subprocess
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

from ..compiler._component_trace.call_graph import Node, _build_call_graph
from ..compiler._component_trace.resource_generalization import (
    ConjunctionCall,
    ControlledZPowerCall,
    DeclaredResourceCalls,
    FixedTCall,
    PhaseCall,
    ResourceCall,
)
from ..compiler._component_trace.resource_model import ComponentCost
from ..compiler._component_trace.symbolic_math import SymbolicInt
from ..compiler._component_trace.terminal_t_cost import (
    _terminal_resource_callees,
    query_terminal_t_cost,
)
from .call_graph import _call_order


def render_flame_graph_svg(
    rows: Iterable[str], *, executable: str, timeout: float = 30.0
) -> bytes:
    """Render folded rows with a caller-provided FlameGraph executable.

    The executable must support --countname TCounts and a folded-data filename.
    Rows are streamed to a UTF-8 temporary file, closed on every exit path.
    The result occupies memory proportional to the SVG output. The subprocess
    has a bounded timeout. Missing executables raise ImportError, process
    failures raise RuntimeError with a bounded stderr diagnostic.
    """
    renderer = shutil.which(executable)
    if renderer is None:
        raise ImportError(
            f"Flame graph rendering requires an executable {executable!r}"
        )
    with TemporaryDirectory(prefix="flagquantum-flame-") as directory:
        path = Path(directory) / "folded.txt"
        with path.open("w", encoding="utf-8") as data:
            separator = ""
            for row in rows:
                data.write(separator)
                data.write(row)
                separator = "\n"
        try:
            return subprocess.run(
                [renderer, "--countname", "TCounts", str(path)],
                capture_output=True,
                check=True,
                timeout=timeout,
            ).stdout
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Flame graph rendering timed out after {timeout:g}s"
            ) from exc
        except subprocess.CalledProcessError as exc:
            diagnostic = exc.stderr[-4000:].decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Flame graph SVG rendering failed: {diagnostic}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Cannot run flame graph executable {renderer}: {exc}"
            ) from exc


def _keep_small_resource(call: ResourceCall) -> bool:
    return isinstance(call, (FixedTCall, ControlledZPowerCall)) or (
        isinstance(call, ConjunctionCall)
        and isinstance(call.controls, tuple)
        and len(call.controls) == 2
    )


def _resource_title(call: ResourceCall) -> str:
    return call.name if isinstance(call, ComponentCost) else str(call)


def _concrete_count(value: SymbolicInt) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Flame graph counts must be nonnegative concrete integers")
    return value


def iter_resource_flame_graph_data(
    *roots: ResourceCall,
    declared_calls: DeclaredResourceCalls | None = None,
    title: Callable[[ResourceCall], str] = _resource_title,
    keep: Callable[[ResourceCall], bool] | None = _keep_small_resource,
    max_depth: int | None = None,
) -> Generator[str, None, None]:
    """Compose native declarations, full precision costs and visible DFS paths.

    Roots are emitted in argument order. Each root's complete costs are
    computed independently of keep and max_depth. The default keep predicate
    retains binary conjunctions, Toffoli, Fredkin and controlled Z powers.
    Known compound keys expand using the shared Compiler recipes. Explicit
    declarations override those recipes. Component names provide native titles,
    other keys use str unless
    title is supplied. Counts must be concrete nonnegative Python integers.
    Files and output buffering belong to the caller.
    """
    costs: dict[ResourceCall, SymbolicInt] = {}
    for root in roots:
        query_terminal_t_cost(root, declared_calls=declared_calls, cache=costs)
        visible = _build_call_graph(
            root,
            lambda node: _terminal_resource_callees(node, declared_calls),
            None,
            keep,
            max_depth,
        )
        calls = {edge: _concrete_count(n) for edge, n in visible.direct_calls.items()}
        totals = {node: _concrete_count(n) for node, n in costs.items()}
        yield from iter_flame_graph_data(
            root,
            calls,
            totals,
            title=title,
            is_t_gate=lambda node: isinstance(node, PhaseCall) and node.opcode == "t",
        )


def iter_flame_graph_data(
    root: Node,
    direct_calls: Mapping[tuple[Node, Node], int],
    t_counts: Mapping[Node, int],
    *,
    title: Callable[[Node], str] = str,
    is_t_gate: Callable[[Node], bool] | None = None,
) -> Generator[str, None, None]:
    """Yield DFS paths, repeating complete subtrees by call multiplicity.

    Counts describe full subtrees even when direct_calls is depth-limited.
    A single positive-cost T or T-adjoint child is represented by its parent
    frame when is_t_gate identifies it. Zero-cost children are omitted.
    The iterator keeps graph and path memory, independent of emitted row count.
    Validation runs before the first row. Cycles, missing or nonconcrete costs,
    negative multiplicities and frame separators in titles raise ValueError.

    Examples:
        >>> list(iter_flame_graph_data(
        ...     "Root", {("Root", "And"): 2}, {"Root": 8, "And": 4}
        ... ))
        ['Root(T:8);And(T:4)\\t4', 'Root(T:8);And(T:4)\\t4']
    """
    for count in direct_calls.values():
        if type(count) is not int or count < 0:
            raise ValueError("Flame graph multiplicities must be nonnegative integers")
    order, successors = _call_order(root, direct_calls)
    frames = {}
    for node in order:
        cost = t_counts.get(node)
        if type(cost) is not int or cost < 0:
            raise ValueError("Flame graph T costs must be nonnegative integers")
        label = title(node)
        if any(separator in label for separator in (";", "\t", "\n", "\r")):
            raise ValueError("Flame graph titles cannot contain frame separators")
        frames[node] = f"{label}(T:{cost})"
    path: list[str] = []

    def visit(node: Node) -> Iterator[str]:
        children = [child for child in successors.get(node, ()) if t_counts[child] > 0]
        path.append(frames[node])
        if not children or (
            len(children) == 1 and is_t_gate is not None and is_t_gate(children[0])
        ):
            yield ";".join(path) + "\t" + str(t_counts[node])
        else:
            for child in children:
                for _ in range(direct_calls[node, child]):
                    yield from visit(child)
        path.pop()

    yield from visit(root)
