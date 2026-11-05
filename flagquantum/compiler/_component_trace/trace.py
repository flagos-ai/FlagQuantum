"""Call instances bound to half-open instruction ranges in emitted IR."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ComponentCall:
    """One invocation and its ordered child calls."""

    name: str
    start: int
    stop: int
    parameters: tuple[object, ...] = ()
    children: tuple["ComponentCall", ...] = ()


@dataclass
class ComponentTrace:
    """A compiler-local sink owned by the caller assembling the final IR."""

    calls: list[ComponentCall] = field(default_factory=list)
