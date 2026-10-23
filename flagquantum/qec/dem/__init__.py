"""Historical imports for :mod:`flagquantum.qec.dem.model`; Core-owned compatibility boundary."""

from __future__ import annotations

import math as math
from collections.abc import Callable as Callable
from collections.abc import Iterable as Iterable
from dataclasses import dataclass as dataclass
from enum import Enum as Enum
from numbers import Integral as Integral
from numbers import Real as Real
from typing import Any as Any

import torch as torch

from flagquantum.noise.model import NoiseModel as NoiseModel
from flagquantum.qec.circuits.memory import MemoryCircuit as MemoryCircuit
from flagquantum.qec.codes.checks import StabilizerCode as StabilizerCode
from flagquantum.qec.codes.matrices import CssCodeMatrices as CssCodeMatrices
from flagquantum.qec.codes.matrices import css_code_matrices as css_code_matrices
from flagquantum.qec.dem.alternatives import alternative_groups as alternative_groups
from flagquantum.qec.dem.alternatives import fold_marginals as fold_marginals
from flagquantum.qec.dem.alternatives import projected_ids as projected_ids
from flagquantum.qec.dem.alternatives import require_groups_fit as require_groups_fit
from flagquantum.qec.dem.alternatives import sample_groups as sample_groups
from flagquantum.qec.dem.alternatives import stated_ids as stated_ids
from flagquantum.qec.dem.construction import (
    _memory_circuit_entries as _memory_circuit_entries,
)
from flagquantum.qec.dem.matrices import _code_matrix_entries as _code_matrix_entries
from flagquantum.qec.dem.model import DemError as DemError
from flagquantum.qec.dem.model import DemMergeRule as DemMergeRule
from flagquantum.qec.dem.model import DemSample as DemSample
from flagquantum.qec.dem.model import DetectorErrorModel as DetectorErrorModel
from flagquantum.qec.dem.model import __all__ as __all__
from flagquantum.qec.dem.model import _combine_two as _combine_two
from flagquantum.qec.dem.model import _count as _count
from flagquantum.qec.dem.model import _error_id as _error_id
from flagquantum.qec.dem.model import _normalized_indices as _normalized_indices
from flagquantum.qec.dem.model import _probability as _probability
from flagquantum.qec.dem.text import _DETECTOR_PREFIX as _DETECTOR_PREFIX
from flagquantum.qec.dem.text import _OBSERVABLE_PREFIX as _OBSERVABLE_PREFIX
from flagquantum.qec.dem.text import _parse_stim_text as _parse_stim_text
from flagquantum.qec.noise import PhenomenologicalNoise as PhenomenologicalNoise
