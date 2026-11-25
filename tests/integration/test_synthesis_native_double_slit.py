"""Complete native double-slit state acceptance against the independent tape."""

import hashlib

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.cfd.double_slit_experiment.quantum_double_slit_experiment import (
    main as example,
)
from examples.synthesis.applications.cfd.double_slit_experiment.quantum_double_slit_experiment.comparisons.circuit import (
    build_reference,
)
from flagquantum.compiler.resource_estimation import ResourceEstimate
from tests.integration.test_synthesis_qmod_double_slit_gate_tape import (
    _original,
    reference,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("cycles", [1, 95])
def test_native_double_slit_matches_complete_independent_state(cycles: int) -> None:
    original = _original()
    assert (
        hashlib.sha256(original.read_bytes()).hexdigest()
        == "ec6de8a9d82cad74f9c98aa3255728a652d275194f77e6bdee2d9e3717d59736"
    )
    source = original.read_text()
    program = example.build_program(cycles=cycles)
    assert program.registers == {
        "x": tuple(range(4)),
        "y": tuple(range(4, 8)),
        "select": (8, 9),
        "edge": (10,),
        "scatter": (11,),
        "axis": (12,),
        "direction": (13,),
        "aux": (14,),
    }
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == 23
    assert resources.n_operations <= 100_000
    state = fq.run(program).statevector()
    assert state.numel() * state.element_size() == 128 * 1024**2
    actual = state.reshape(1 << 15, 1 << (resources.n_qubits - 15))
    if cycles == 1:
        source = source.replace("-598.477076962426", "-0.271")
    expected = reference(source, cycles)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-10, rtol=0)
    assert torch.count_nonzero(actual[:, 1:]) == 0
    independent = (
        fq.run(build_reference(cycles=cycles)).statevector().reshape(1 << 15, 1 << 9)
    )
    torch.testing.assert_close(independent[:, 0], expected, atol=1e-10, rtol=0)
    assert torch.count_nonzero(independent[:, 1:]) == 0
