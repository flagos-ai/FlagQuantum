"""Private bounded decoder-feedback records for dynamic trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class DynamicFeedbackPoint:
    """One decision boundary after a classical bit has been recorded."""

    name: str
    trigger_classical_bit: int
    classical_bits: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("feedback point name cannot be empty")
        if self.trigger_classical_bit < 0:
            raise ValueError("feedback trigger bit must be non-negative")
        if not self.classical_bits:
            raise ValueError("feedback point requires at least one classical bit")
        if any(bit < 0 for bit in self.classical_bits):
            raise ValueError("feedback classical bits must be non-negative")
        if len(set(self.classical_bits)) != len(self.classical_bits):
            raise ValueError("feedback classical bits must be unique")
        if self.trigger_classical_bit not in self.classical_bits:
            raise ValueError("feedback trigger must belong to the observed bits")


@dataclass(frozen=True)
class DynamicFeedbackObservation:
    """True/observed measurement record presented to one controller."""

    point_name: str
    decision_index: int
    classical_bits: tuple[int, ...]
    true_bits: tuple[int, ...]
    observed_bits: tuple[int, ...]
    frame_x_wires_before: tuple[int, ...]


@dataclass(frozen=True)
class DynamicFeedbackAction:
    """A bounded physical-X, Pauli-frame-X, or no-op decision."""

    mode: str = "none"
    wire: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"none", "physical_x", "frame_x"}:
            raise ValueError("unsupported dynamic feedback action mode")
        if self.mode == "none" and self.wire is not None:
            raise ValueError("a no-op feedback action cannot name a wire")
        if self.mode != "none" and (type(self.wire) is not int or self.wire < 0):
            raise ValueError("an X feedback action requires a non-negative wire")


@dataclass(frozen=True)
class DynamicFeedbackDecision:
    """One auditable observation, action, and resulting frame state."""

    observation: DynamicFeedbackObservation
    action: DynamicFeedbackAction
    frame_x_wires_after: tuple[int, ...]


@dataclass(frozen=True)
class DynamicFeedbackTrace:
    """Ordered decisions for one independent trajectory."""

    decisions: tuple[DynamicFeedbackDecision, ...]


class DynamicFeedbackController(Protocol):
    """Private stateless controller contract over one trajectory history."""

    def decide(
        self, history: Sequence[DynamicFeedbackObservation]
    ) -> DynamicFeedbackAction: ...


@dataclass(frozen=True)
class DynamicFeedbackPlan:
    """Validated feedback boundaries and their bounded controller."""

    points: tuple[DynamicFeedbackPoint, ...]
    controller: DynamicFeedbackController
    allowed_wires: tuple[int, ...]
    allowed_action_modes: tuple[str, ...] = ("physical_x",)

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("dynamic feedback plan requires at least one point")
        names = tuple(point.name for point in self.points)
        triggers = tuple(point.trigger_classical_bit for point in self.points)
        if len(set(names)) != len(names) or len(set(triggers)) != len(triggers):
            raise ValueError("feedback point names and triggers must be unique")
        if tuple(sorted(triggers)) != triggers:
            raise ValueError("feedback points must be ordered by trigger bit")
        if tuple(sorted(set(self.allowed_wires))) != self.allowed_wires:
            raise ValueError("feedback allowed wires must be unique and ordered")
        if not self.allowed_wires or any(wire < 0 for wire in self.allowed_wires):
            raise ValueError("feedback plan requires non-negative allowed wires")
        if not self.allowed_action_modes or any(
            mode not in {"physical_x", "frame_x"} for mode in self.allowed_action_modes
        ):
            raise ValueError("feedback plan action modes are invalid")
        if len(set(self.allowed_action_modes)) != len(self.allowed_action_modes):
            raise ValueError("feedback plan action modes must be unique")
        if not callable(getattr(self.controller, "decide", None)):
            raise TypeError("feedback controller must define decide(history)")


__all__ = (
    "DynamicFeedbackAction",
    "DynamicFeedbackController",
    "DynamicFeedbackDecision",
    "DynamicFeedbackObservation",
    "DynamicFeedbackPlan",
    "DynamicFeedbackPoint",
    "DynamicFeedbackTrace",
)
