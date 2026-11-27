from tests._reference_dependencies import unavailable_reference

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

# Adapted from Qualtran v0.7.0 rotation_synthesis/lattice/test_utils_test.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/rotation_synthesis/lattice/test_utils_test.py

try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

from flagquantum.synth import _lattice as lattice
from tests.unit import rotation_lattice_fixtures as tu


def test_make_psd() -> None:
    rng = np.random.default_rng(0)
    for _ in range(10):
        matrix = tu.make_psd(rng)
        assert np.all(np.linalg.eigvals(matrix) >= 0)


def test_make_ellipse() -> None:
    for n in range(3, 5):
        es = tuple(tuple(tu.make_ellipses(n)))
        assert len(es) == n
        for e in es:
            assert isinstance(e, lattice.Ellipse)


def test_make_states() -> None:
    for n in range(3, 5):
        states = tuple(tuple(tu.make_states(n)))
        assert len(states) == n
        for state in states:
            assert isinstance(state, lattice.SelingerState)


def test_make_gridoperators() -> None:
    for n in range(3, 5):
        gos = tuple(tuple(tu.random_grid_operator(n, 3)))
        assert len(gos) == n
        for go in gos:
            assert isinstance(go, lattice.GridOperator)
