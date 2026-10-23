"""Historical sampling imports; Core owns these compatibility aliases."""

from collections.abc import Callable as Callable
from collections.abc import Mapping as Mapping
from collections.abc import Sequence as Sequence
from dataclasses import dataclass as dataclass
from numbers import Integral as Integral
from types import MappingProxyType as MappingProxyType
from typing import Literal as Literal

import torch as torch

from flagquantum.core.ir import CircuitIR as CircuitIR
from flagquantum.core.ir import Instruction as Instruction
from flagquantum.noise import KrausChannel as KrausChannel
from flagquantum.noise import NoiseModel as NoiseModel
from flagquantum.noise import bit_flip_channel as bit_flip_channel
from flagquantum.noise import phase_flip_channel as phase_flip_channel
from flagquantum.noise import y_flip_channel as y_flip_channel
from flagquantum.qec.circuits.memory import MeasurementRef as MeasurementRef
from flagquantum.qec.circuits.memory import MemoryCircuit as MemoryCircuit
from flagquantum.qec.dem import DemSample as DemSample
from flagquantum.qec.dem_construction import _check_rate_order as _check_rate_order
from flagquantum.qec.dem_construction import _data_family_rates as _data_family_rates
from flagquantum.qec.dem_construction import (
    _gate_bound_mechanisms as _gate_bound_mechanisms,
)
from flagquantum.qec.dem_construction import _MechanismKind as _MechanismKind
from flagquantum.qec.execution.signatures import _lowered_program as _lowered_program
from flagquantum.qec.noise import PhenomenologicalNoise as PhenomenologicalNoise
from flagquantum.simulation.stabilizer import (
    sample_noisy_measurements as sample_noisy_measurements,
)

from .execution.sampling import (
    _DATA_FAULT_CHANNEL_BY_FAMILY as _DATA_FAULT_CHANNEL_BY_FAMILY,
)
from .execution.sampling import _DATA_FAULT_CHANNELS as _DATA_FAULT_CHANNELS
from .execution.sampling import _MEASURE_OPCODE as _MEASURE_OPCODE
from .execution.sampling import _ROTATION_OPCODE as _ROTATION_OPCODE
from .execution.sampling import MeasurementSamples as MeasurementSamples
from .execution.sampling import __all__ as __all__
from .execution.sampling import _ChannelName as _ChannelName
from .execution.sampling import _checked_seed as _checked_seed
from .execution.sampling import _checked_shots as _checked_shots
from .execution.sampling import _gate_bound_locations as _gate_bound_locations
from .execution.sampling import _measurement_plan as _measurement_plan
from .execution.sampling import _measurement_samples as _measurement_samples
from .execution.sampling import _MeasurementPlan as _MeasurementPlan
from .execution.sampling import _noise_locations as _noise_locations
from .execution.sampling import _NoiseLocation as _NoiseLocation
from .execution.sampling import _noisy_program as _noisy_program
from .execution.sampling import _recorded_bits as _recorded_bits
from .execution.sampling import _rotation_template as _rotation_template
from .execution.sampling import _round_boundary_locations as _round_boundary_locations
from .execution.sampling import _sample_record as _sample_record
from .execution.sampling import sample_memory_circuit as sample_memory_circuit
from .execution.sampling import sample_memory_measurements as sample_memory_measurements
