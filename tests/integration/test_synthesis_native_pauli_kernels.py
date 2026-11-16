"""Compare full native parameterized kernels with originals and independent Circuits."""

from pathlib import Path
from typing import Literal

import pytest

from examples.synthesis.algorithms.machine_learning.qsvm.comparisons.circuit import (
    reference_circuit as qsvm_reference,
)
from examples.synthesis.algorithms.machine_learning.qsvm.main import main as qsvm_main
from examples.synthesis.applications.optimization.variational_quantum_imaginary_time_evolution.comparisons.circuit import (
    reference_circuit as imaginary_time_reference,
)
from examples.synthesis.applications.optimization.variational_quantum_imaginary_time_evolution.main import (
    main as imaginary_time_main,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/corpus/original"


@pytest.mark.parametrize(
    "first,second", [((0.23, -0.41), (-0.17, 0.37)), ((-0.29, 0.19), (0.31, -0.43))]
)
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_bloch_kernel_preserves_inverse_order_and_absolute_phase(
    first: tuple[float, float],
    second: tuple[float, float],
    objective: Literal["width", "depth"],
) -> None:
    binding = {"data1": first, "data2": second}
    source = ROOT / "algorithms/QML/qsvm/qsvm.qmod"
    retained = synth.synthesize(
        source, classical_inputs=binding, objective=objective, max_workspace=0
    )
    native = synth.synthesize(
        qsvm_main, classical_inputs=binding, objective=objective, max_workspace=0
    )
    for program in (retained, native):
        compare_states(program, qsvm_reference(first, second), ("qba",))
        assert program.report["workspace_qubits"] == 0
    assert native.report["resources"] == retained.report["resources"]


@pytest.mark.parametrize(
    "angles",
    [(0.13, -0.21, 0.32, -0.17, 0.29, -0.41), (-0.23, 0.11, -0.37, 0.19, -0.31, 0.43)],
)
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_all_six_imaginary_time_generators_keep_order_and_logical_bit_positions(
    angles: tuple[float, ...],
    objective: Literal["width", "depth"],
) -> None:
    source = (
        ROOT
        / "applications/optimization/variational_quantum_imaginary_time_evolution/variational_quantum_imaginary_time_evolution.qmod"
    )
    retained = synth.synthesize(
        source,
        classical_inputs={"angles": angles},
        objective=objective,
        max_workspace=0,
    )
    native = synth.synthesize(
        imaginary_time_main,
        classical_inputs={"angles": angles},
        objective=objective,
        max_workspace=0,
    )
    for program in (retained, native):
        compare_states(program, imaginary_time_reference(angles), ("nodes",))
        assert program.report["workspace_qubits"] == 0
    assert native.report["resources"] == retained.report["resources"]
