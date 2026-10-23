"""Historical context imports; Core owns these compatibility aliases."""

from collections.abc import Iterable as Iterable
from dataclasses import dataclass as dataclass
from numbers import Integral as Integral

import torch as torch

from flagquantum.noise import NoiseModel as NoiseModel
from flagquantum.qec.circuits.memory import MeasurementRef as MeasurementRef
from flagquantum.qec.circuits.memory import MemoryCircuit as MemoryCircuit
from flagquantum.qec.codes import CodeCheck as CodeCheck
from flagquantum.qec.dem import DetectorErrorModel as DetectorErrorModel
from flagquantum.qec.noise import PhenomenologicalNoise as PhenomenologicalNoise

from .execution.context import DecoderContext as DecoderContext
from .execution.context import DecoderInputs as DecoderInputs
from .execution.context import MeasurementMap as MeasurementMap
from .execution.context import __all__ as __all__
from .execution.context import _detector_checks as _detector_checks
from .execution.context import _measurement_columns as _measurement_columns
from .execution.context import _measurement_index as _measurement_index
from .execution.context import _measurement_row as _measurement_row
from .execution.context import _parity_row as _parity_row
from .execution.context import _projected as _projected
from .execution.context import (
    decoder_context_from_memory_circuit as decoder_context_from_memory_circuit,
)
