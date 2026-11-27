"""Historical public plans retain their concrete matrix-free values."""

import base64
import importlib
import json
import pickle
from pathlib import Path
from typing import get_type_hints

import pytest
import torch

import flagquantum.lindblad as fql
from flagquantum.simulation.numerics.matrix_free.hamiltonian import (
    PauliSum,
    PauliSumTerm,
)

pytestmark = pytest.mark.unit


def test_old_public_plan_pickle_executes_after_numerical_relocation() -> None:
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures/structure/matrix_free_public_plan_pickle.json"
    )
    data = json.loads(fixture.read_text())
    plan = pickle.loads(base64.b64decode(data["pickle_base64"]))
    result = fql.run(plan).expectation()
    torch.testing.assert_close(
        result, torch.tensor(data["expected_expectation"], dtype=result.dtype)
    )
    locator = importlib.import_module("flagquantum.simulation.matrix_free_hamiltonian")
    assert locator.PauliSum is PauliSum
    assert locator.PauliSumTerm is PauliSumTerm
    assert get_type_hints(locator.PauliSumTerm)["coefficient"] is complex


@pytest.mark.parametrize("name", ["autograd", "exponential", "linear_solve"])
def test_retired_numerical_imports_are_not_recreated(name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("flagquantum.simulation.matrix_free_" + name)
