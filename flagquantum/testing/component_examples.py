# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _infra/bloq_example.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/bloq_example.py

"""Immutable example factories and documentation metadata for native components.

These records describe examples. They do not build circuits until called,
interpret generalizers, or participate in production execution decisions.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, TypeVar, overload

Component = TypeVar("Component")


def _identity(component: Component) -> Component:
    return component


@dataclass(frozen=True)
class ComponentExample(Generic[Component]):
    """Store a repeatable factory with names and optional resource generalizers.

    The factory remains lazy and is called afresh by both make and __call__.
    Generalizer sequences are frozen as tuples. A factory's equality contributes
    to record equality, while its hash and representation are excluded.
    """

    factory: Callable[[], Component] = field(repr=False, hash=False)
    name: str
    component_type: type[Component]
    generalizer: (
        Callable[[Component], Component | None]
        | Sequence[Callable[[Component], Component | None]]
    ) = _identity
    description: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.generalizer, Sequence):
            object.__setattr__(self, "generalizer", tuple(self.generalizer))

    def make(self) -> Component:
        return self.factory()

    def __call__(self) -> Component:
        return self.make()


@overload
def component_example(
    factory: Callable[[], Component],
    *,
    generalizer: (
        Callable[[Component], Component | None]
        | Sequence[Callable[[Component], Component | None]]
    ) = _identity,
) -> ComponentExample[Component]: ...


@overload
def component_example(
    factory: None = None,
    *,
    generalizer: (
        Callable[[Component], Component | None]
        | Sequence[Callable[[Component], Component | None]]
    ) = _identity,
) -> Callable[[Callable[[], Component]], ComponentExample[Component]]: ...


def component_example(
    factory: Callable[[], Component] | None = None,
    *,
    generalizer: (
        Callable[[Component], Component | None]
        | Sequence[Callable[[Component], Component | None]]
    ) = _identity,
) -> (
    ComponentExample[Component]
    | Callable[[Callable[[], Component]], ComponentExample[Component]]
):
    """Infer a factory's name, concrete return class and original docstring.

    Names strip all leading underscores. Annotations must be actual classes;
    forward references are deliberately not evaluated. Native component types
    have no required common base class. Factory exceptions propagate on make.
    """

    def decorate(function: Callable[[], Component]) -> ComponentExample[Component]:
        if "return" not in function.__annotations__:
            raise ValueError(f"{function} must have a return type annotation.")
        component_type = function.__annotations__["return"]
        if not isinstance(component_type, type):
            raise TypeError("component factory return annotation must be a class")
        return ComponentExample(
            factory=function,
            name=function.__name__.lstrip("_"),
            component_type=component_type,
            generalizer=generalizer,
            description=function.__doc__,
        )

    return decorate if factory is None else decorate(factory)


class _Default(Enum):
    FIRST_EXAMPLE = "first_example"


@dataclass(frozen=True, init=False)
class ComponentDocumentation(Generic[Component]):
    """Group factory metadata without executing examples or rendering output.

    Examples are captured from the supplied iterable once. By default the first
    example supplies the call graph. An explicit None suppresses that graph.
    Import inference follows the source parent-package rule, so callers whose
    types live outside package modules should supply an explicit import line.
    """

    component_type: type[Component]
    examples: tuple[ComponentExample[Component], ...]
    import_line: str
    call_graph_example: ComponentExample[Component] | None

    def __init__(
        self,
        *,
        component_type: type[Component],
        examples: Iterable[ComponentExample[Component]] = (),
        import_line: str | None = None,
        call_graph_example: ComponentExample[Component] | None | _Default = (
            _Default.FIRST_EXAMPLE
        ),
    ) -> None:
        stored = tuple(examples)
        if import_line is None:
            package = ".".join(component_type.__module__.split(".")[:-1])
            import_line = f"from {package} import {component_type.__name__}"
        if call_graph_example is _Default.FIRST_EXAMPLE:
            call_graph_example = stored[0] if stored else None
        object.__setattr__(self, "component_type", component_type)
        object.__setattr__(self, "examples", stored)
        object.__setattr__(self, "import_line", import_line)
        object.__setattr__(self, "call_graph_example", call_graph_example)
