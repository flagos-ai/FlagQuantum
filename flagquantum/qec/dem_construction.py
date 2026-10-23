"""Historical imports for :mod:`flagquantum.qec.dem.construction`; Core-owned compatibility boundary."""

from __future__ import annotations

import math as math
from collections.abc import Mapping as Mapping
from dataclasses import dataclass as dataclass
from dataclasses import replace as replace
from numbers import Integral as Integral
from types import MappingProxyType as MappingProxyType
from typing import Literal as Literal

import torch as torch

from flagquantum.compiler._hybrid.capture import capture_source as capture_source
from flagquantum.compiler._hybrid.dynamic_lowering import (
    lower_dynamic_program as lower_dynamic_program,
)
from flagquantum.compiler._hybrid.model import INDEX as INDEX
from flagquantum.core.ir import CircuitIR as CircuitIR
from flagquantum.core.ir import Instruction as Instruction
from flagquantum.errors import CapabilityError as CapabilityError
from flagquantum.errors import ValidationError as ValidationError
from flagquantum.noise.channels import KrausChannel as KrausChannel
from flagquantum.noise.model import NoiseModel as NoiseModel
from flagquantum.qec.circuits.memory import MeasurementRef as MeasurementRef
from flagquantum.qec.circuits.memory import MemoryCircuit as MemoryCircuit
from flagquantum.qec.codes.checks import CodeCheck as CodeCheck
from flagquantum.qec.codes.checks import StabilizerCode as StabilizerCode
from flagquantum.qec.codes.matrices import CssCodeMatrices as CssCodeMatrices
from flagquantum.qec.codes.matrices import _binary_matrix as _binary_matrix
from flagquantum.qec.codes.matrices import css_code_matrices as css_code_matrices
from flagquantum.qec.dem.construction import _COMPOSITE_KIND as _COMPOSITE_KIND
from flagquantum.qec.dem.construction import _COMPOSITE_PARTS as _COMPOSITE_PARTS
from flagquantum.qec.dem.construction import (
    _GATE_BOUND_FAMILIES as _GATE_BOUND_FAMILIES,
)
from flagquantum.qec.dem.construction import _GATE_BOUND_GATES as _GATE_BOUND_GATES
from flagquantum.qec.dem.construction import _PAULI_MATRICES as _PAULI_MATRICES
from flagquantum.qec.dem.construction import (
    _UNPLACEABLE_RULE_GATES as _UNPLACEABLE_RULE_GATES,
)
from flagquantum.qec.dem.construction import __all__ as __all__
from flagquantum.qec.dem.construction import __annotations__ as __annotations__
from flagquantum.qec.dem.construction import _check_rate_order as _check_rate_order
from flagquantum.qec.dem.construction import _data_family_rates as _data_family_rates
from flagquantum.qec.dem.construction import (
    _flipped_measurements as _flipped_measurements,
)
from flagquantum.qec.dem.construction import (
    _gate_bound_mechanisms as _gate_bound_mechanisms,
)
from flagquantum.qec.dem.construction import _GateBoundMechanism as _GateBoundMechanism
from flagquantum.qec.dem.construction import _layout_signature as _layout_signature
from flagquantum.qec.dem.construction import _Mechanism as _Mechanism
from flagquantum.qec.dem.construction import _mechanisms as _mechanisms
from flagquantum.qec.dem.construction import (
    _memory_circuit_entries as _memory_circuit_entries,
)
from flagquantum.qec.dem.construction import _pauli_fault as _pauli_fault
from flagquantum.qec.dem.construction import (
    _require_owned_syndrome_references as _require_owned_syndrome_references,
)
from flagquantum.qec.dem.construction import (
    _require_source_anchors as _require_source_anchors,
)
from flagquantum.qec.dem.construction import _spliced_fault as _spliced_fault
from flagquantum.qec.dem.matrices import Entry as Entry
from flagquantum.qec.dem.matrices import _code_matrix_entries as _code_matrix_entries
from flagquantum.qec.dem.matrices import _column_supports as _column_supports
from flagquantum.qec.execution.signatures import _FAULT_GATES as _FAULT_GATES
from flagquantum.qec.execution.signatures import (
    _FAULT_KIND_BY_LETTER as _FAULT_KIND_BY_LETTER,
)
from flagquantum.qec.execution.signatures import (
    _FAULT_LETTER_BY_KIND as _FAULT_LETTER_BY_KIND,
)
from flagquantum.qec.execution.signatures import _FLIP_INDENT as _FLIP_INDENT
from flagquantum.qec.execution.signatures import (
    _ROUND_LOOP_ANCHOR as _ROUND_LOOP_ANCHOR,
)
from flagquantum.qec.execution.signatures import _checked_round as _checked_round
from flagquantum.qec.execution.signatures import _checked_wire as _checked_wire
from flagquantum.qec.execution.signatures import _DataFaultFamily as _DataFaultFamily
from flagquantum.qec.execution.signatures import (
    _executed_flip_set as _executed_flip_set,
)
from flagquantum.qec.execution.signatures import _forced_flip as _forced_flip
from flagquantum.qec.execution.signatures import _forced_signature as _forced_signature
from flagquantum.qec.execution.signatures import _inject_data_flip as _inject_data_flip
from flagquantum.qec.execution.signatures import (
    _inject_measurement_flip as _inject_measurement_flip,
)
from flagquantum.qec.execution.signatures import _lowered_program as _lowered_program
from flagquantum.qec.execution.signatures import _lowered_source as _lowered_source
from flagquantum.qec.execution.signatures import _MechanismKind as _MechanismKind
from flagquantum.qec.execution.signatures import _single_anchor as _single_anchor
from flagquantum.qec.noise import PhenomenologicalNoise as PhenomenologicalNoise
from flagquantum.runtime.dynamic.hybrid_session import (
    execute_hybrid_dynamic_session as execute_hybrid_dynamic_session,
)
