"""Reusable quantum primitives shared by the algorithm modules.

Primitives build :class:`~flagquantum.circuit.Circuit` objects from other circuits and from
classical data. They do not define circuit or operator semantics, select a runtime, or
introduce an execution path of their own.

A primitive is admitted here only when at least two algorithm modules need it; this package
is not a general-purpose quantum toolkit.
"""

from __future__ import annotations

from .phase_estimation import PhaseEstimationSpec as PhaseEstimationSpec
from .phase_estimation import append_phase_estimation as append_phase_estimation
from .phase_estimation import phase_estimation_circuit as phase_estimation_circuit
from .qft import append_qft as append_qft
from .qft import qft as qft
from .types import ControlledUnitary as ControlledUnitary
from .types import Predicate as Predicate
from .types import StatePreparationOperator as StatePreparationOperator

__all__ = [
    "ControlledUnitary",
    "PhaseEstimationSpec",
    "Predicate",
    "StatePreparationOperator",
    "append_phase_estimation",
    "append_qft",
    "phase_estimation_circuit",
    "qft",
]
