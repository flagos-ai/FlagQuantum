"""Historical imports for :mod:`flagquantum.qec.dem.alternatives`; Core-owned compatibility boundary."""

from __future__ import annotations

from collections.abc import Callable as Callable
from collections.abc import Sequence as Sequence
from typing import TYPE_CHECKING as TYPE_CHECKING

import torch as torch

from flagquantum.qec.dem.alternatives import __all__ as __all__
from flagquantum.qec.dem.alternatives import alternative_groups as alternative_groups
from flagquantum.qec.dem.alternatives import fault_groups as fault_groups
from flagquantum.qec.dem.alternatives import fold_marginals as fold_marginals
from flagquantum.qec.dem.alternatives import projected_ids as projected_ids
from flagquantum.qec.dem.alternatives import require_groups_fit as require_groups_fit
from flagquantum.qec.dem.alternatives import sample_groups as sample_groups
from flagquantum.qec.dem.alternatives import stated_ids as stated_ids
