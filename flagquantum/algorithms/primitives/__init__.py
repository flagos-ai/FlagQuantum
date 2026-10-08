"""Reusable quantum primitives shared by the algorithm modules.

Primitives build :class:`~flagquantum.circuit.Circuit` objects from other circuits and from
classical data. They do not define circuit or operator semantics, select a runtime, or
introduce an execution path of their own.

A primitive is admitted here on three dimensions, and the package README states each of them at
length. Its **consumer** fixes the ground: more than one algorithm module needs it, or one does
and the expectation of a second is later confirmed, or it is a public unit callers use
directly. No consumer anywhere is not a fourth ground, it is a defect: an export that nothing
in this repository imports is deleted rather than grandfathered. It claims no **distribution
semantics** beyond the default single-device fast path unless the contract records otherwise,
because a subroutine that names a semantic is making a scalability claim it cannot support.
And it records its **differentiability**: a signature that takes a tensor and returns no
gradient through it is reading classical data, and the contract says so rather than leaving a
trainable-looking argument to be misread.

This package is not a general-purpose quantum toolkit.
"""

from __future__ import annotations

from .oracle import append_bit_oracle as append_bit_oracle
from .oracle import append_comparator as append_comparator
from .oracle import append_multi_controlled_x as append_multi_controlled_x
from .oracle import append_phase_oracle as append_phase_oracle
from .oracle import bit_oracle as bit_oracle
from .oracle import marked_states as marked_states
from .oracle import phase_oracle as phase_oracle
from .phase_estimation import PhaseEstimationSpec as PhaseEstimationSpec
from .phase_estimation import append_phase_estimation as append_phase_estimation
from .phase_estimation import phase_estimation_circuit as phase_estimation_circuit
from .qft import append_qft as append_qft
from .qft import qft as qft
from .state_preparation import append_arbitrary_state as append_arbitrary_state
from .state_preparation import arbitrary_state as arbitrary_state
from .state_preparation import uniform_state as uniform_state
from .types import AmplitudeOperator as AmplitudeOperator
from .types import ControlledUnitary as ControlledUnitary
from .types import Predicate as Predicate

__all__ = [
    "AmplitudeOperator",
    "ControlledUnitary",
    "PhaseEstimationSpec",
    "Predicate",
    "append_arbitrary_state",
    "append_bit_oracle",
    "append_comparator",
    "append_multi_controlled_x",
    "append_phase_estimation",
    "append_phase_oracle",
    "append_qft",
    "arbitrary_state",
    "bit_oracle",
    "marked_states",
    "phase_estimation_circuit",
    "phase_oracle",
    "qft",
    "uniform_state",
]
