#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Adapted from Qualtran v0.7.0 rotation_synthesis/lattice/__init__.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/rotation_synthesis/lattice/__init__.py

"""Internal geometry and exact point enumeration for rotation synthesis."""

from ._geometry import Ellipse, Range, Rectangle
from ._grid_operators import GridOperator
from ._ipe import (
    enumerate_1d,
    enumerate_upright,
    get_overall_action,
    get_points_from_state,
)
from ._state import GridOperatorAction, SelingerState

__all__ = [
    "Ellipse",
    "GridOperator",
    "GridOperatorAction",
    "Range",
    "Rectangle",
    "SelingerState",
    "enumerate_1d",
    "enumerate_upright",
    "get_overall_action",
    "get_points_from_state",
]
