"""Native fixed ansatz compositions preserve source angles, phase and bit order."""

import importlib
from pathlib import Path
from typing import Literal

import pytest

from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
CASES = (
    (
        "algorithms.quantum_linear_solvers.vqls.vqls_with_lcu",
        "algorithms/quantum_linear_solvers/vqls/vqls_with_lcu.qmod",
        "system_qubits",
        9,
    ),
    (
        "applications.chemistry.second_quantized_hamiltonian",
        "applications/chemistry/second_quantized_hamiltonian/second_quantized_hamiltonian.qmod",
        "state",
        12,
    ),
)


@pytest.mark.parametrize("case", CASES, ids=("vqls", "hea"))
@pytest.mark.parametrize("binding", range(3))
@pytest.mark.parametrize("objective", ("width", "depth"))
def test_original_native_independent_circuit_complete_states(
    case: tuple[str, str, str, int],
    binding: int,
    objective: Literal["width", "depth"],
) -> None:
    module, original, register, count = case
    native = importlib.import_module(f"examples.synthesis.{module}.main")
    reference = importlib.import_module(
        f"examples.synthesis.{module}.comparisons.circuit"
    )
    params = (
        reference.BINDINGS[binding]
        if binding < 2
        else tuple(7.1 - 1.23 * index for index in range(count))
    )
    assert len(params) == count and all(value != 0 for value in params)
    for entry in (
        native.main,
        ROOT / "tests/fixtures/classiq_qmod/corpus/original" / original,
    ):
        program = synth.synthesize(
            entry,
            classical_inputs={"params": params},
            objective=objective,
            max_workspace=0,
        )
        compare_states(program, reference.reference_circuit(params), (register,))
        assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("case", CASES, ids=("vqls", "hea"))
def test_fixed_binding_shape_and_measured_resource_budgets(
    case: tuple[str, str, str, int],
) -> None:
    module, _, _, count = case
    native = importlib.import_module(f"examples.synthesis.{module}.main")
    for size in (0, count - 1, count + 1):
        with pytest.raises(ValueError, match=f"requires {count} angles"):
            native.build_program([0.2] * size)
    program = native.build_program()
    resource = program.report["resources"]
    assert isinstance(resource, ResourceEstimate)
    for qubits, operations, depth, memory in (
        (resource.n_qubits - 1, None, None, None),
        (None, resource.n_operations - 1, None, None),
        (None, None, resource.depth - 1, None),
        (None, None, None, 1),
    ):
        with pytest.raises(ValueError):
            synth.synthesize(
                native.main,
                classical_inputs={"params": native.DEFAULT_PARAMS},
                max_qubits=qubits,
                max_operations=operations,
                max_depth=depth,
                memory_limit_bytes=memory,
            )
