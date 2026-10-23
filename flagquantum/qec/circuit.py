"""Historical circuit imports; Core owns these compatibility aliases."""

from dataclasses import dataclass as dataclass
from numbers import Integral as Integral
from typing import Literal as Literal

from flagquantum.qec.codes import StabilizerCode as StabilizerCode
from flagquantum.qec.codes import ancilla_band_counts as ancilla_band_counts
from flagquantum.qec.logical import certify_logical_product as certify_logical_product
from flagquantum.qec.pauli import Pauli as Pauli

from .circuits.memory import _FUNCTION_NAME as _FUNCTION_NAME
from .circuits.memory import _READOUT_BASES as _READOUT_BASES
from .circuits.memory import Detector as Detector
from .circuits.memory import DetectorLayout as DetectorLayout
from .circuits.memory import LogicalObservable as LogicalObservable
from .circuits.memory import MeasurementRef as MeasurementRef
from .circuits.memory import MemoryCircuit as MemoryCircuit
from .circuits.memory import ObservableLayout as ObservableLayout
from .circuits.memory import ReadoutBasis as ReadoutBasis
from .circuits.memory import __all__ as __all__
from .circuits.memory import _check_source as _check_source
from .circuits.memory import _checked_basis as _checked_basis
from .circuits.memory import _declared_readout as _declared_readout
from .circuits.memory import _detector_count as _detector_count
from .circuits.memory import _detector_layout as _detector_layout
from .circuits.memory import _observable_layout as _observable_layout
from .circuits.memory import _pinned_checks as _pinned_checks
from .circuits.memory import _readout_basis_of as _readout_basis_of
from .circuits.memory import _readout_observables as _readout_observables
from .circuits.memory import _require_ancilla_bands as _require_ancilla_bands
from .circuits.memory import build_memory_circuit as build_memory_circuit
