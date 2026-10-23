"""Historical context imports; Core owns these compatibility aliases."""

from .execution.context import CodeCheck as CodeCheck
from .execution.context import DecoderContext as DecoderContext
from .execution.context import DecoderInputs as DecoderInputs
from .execution.context import DetectorErrorModel as DetectorErrorModel
from .execution.context import Integral as Integral
from .execution.context import Iterable as Iterable
from .execution.context import MeasurementMap as MeasurementMap
from .execution.context import MeasurementRef as MeasurementRef
from .execution.context import MemoryCircuit as MemoryCircuit
from .execution.context import NoiseModel as NoiseModel
from .execution.context import PhenomenologicalNoise as PhenomenologicalNoise
from .execution.context import __all__ as __all__
from .execution.context import _detector_checks as _detector_checks
from .execution.context import _measurement_columns as _measurement_columns
from .execution.context import _measurement_index as _measurement_index
from .execution.context import _measurement_row as _measurement_row
from .execution.context import _parity_row as _parity_row
from .execution.context import _projected as _projected
from .execution.context import dataclass as dataclass
from .execution.context import (
    decoder_context_from_memory_circuit as decoder_context_from_memory_circuit,
)
from .execution.context import torch as torch
