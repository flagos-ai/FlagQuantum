"""Immutable, explicitly declared classical-layer presentation data."""

from __future__ import annotations

import operator
from dataclasses import dataclass


def _positive_count(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    count = operator.index(value)
    if count <= 0:
        raise ValueError(f"{name} must be positive")
    return count


@dataclass(frozen=True)
class ClassicalLayer:
    """Describe one classical layer without executing or tracing it.

    Args:
        label: Plain-text operation label, such as Conv2d or Flatten.
        input_shape: Positive dimensions excluding the batch dimension.
        output_shape: Positive dimensions excluding the batch dimension.
        trainable: Whether this layer owns parameters to update.

    Shapes are declarations, not a proof of an operation's shape formula.
    draw_hybrid checks continuity; compare with actual model outputs separately.

    Examples:
        >>> from flagquantum.drawer import ClassicalLayer
        >>> layer = ClassicalLayer("Linear", (3,), (2,), trainable=True)
        >>> layer.output_shape
        (2,)
    """

    label: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    trainable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("layer label must be a nonempty string")
        if not isinstance(self.trainable, bool):
            raise TypeError("layer trainable must be a bool")
        for name, dimensions in (
            ("input_shape", self.input_shape),
            ("output_shape", self.output_shape),
        ):
            shape = tuple(_positive_count(n, name) for n in dimensions)
            if not shape:
                raise ValueError(f"{name} must contain at least one dimension")
            object.__setattr__(self, name, shape)
