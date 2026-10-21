"""Public Pauli requests obey the same effective-name rule as matrix requests."""

import pytest
import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.simulation.lindblad import EvolutionValidationError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("entry", [fql.plan, fql.run])
@pytest.mark.parametrize("generated_first", [False, True])
def test_generated_names_cannot_overwrite_explicit_outputs(entry, generated_first):
    outputs = (
        [fq.expectation(fq.Z(0)), fq.expectation(fq.X(0), name="expectation_0")]
        if generated_first
        else [fq.expectation(fq.Z(0), name="expectation_1"), fq.expectation(fq.X(0))]
    )
    with pytest.raises(EvolutionValidationError) as error:
        entry(torch.zeros((2, 2)), "0", [0.0, 0.1], outputs=outputs)
    assert error.value.code == "duplicate_output_name"


@pytest.mark.parametrize("entry", [fql.plan, fql.run])
def test_non_requests_are_still_refused(entry):
    with pytest.raises(TypeError, match="OutputRequest"):
        entry(torch.zeros((2, 2)), "0", [0.0, 0.1], outputs=[object()])
