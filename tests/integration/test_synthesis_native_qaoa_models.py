"""Complete native QAOA programs preserve source costs, layers and bit order."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.optimization.max_clique import main as clique
from examples.synthesis.applications.optimization.max_clique.comparisons import (
    circuit as clique_reference,
)
from examples.synthesis.applications.optimization.max_independent_set import (
    main as independent,
)
from examples.synthesis.applications.optimization.max_independent_set.comparisons import (
    circuit as independent_reference,
)
from examples.synthesis.applications.optimization.max_k_vertex_cover import (
    main as cover,
)
from examples.synthesis.applications.optimization.max_k_vertex_cover.comparisons import (
    circuit as cover_reference,
)
from examples.synthesis.applications.optimization.set_partition import main as partition
from examples.synthesis.applications.optimization.set_partition.comparisons import (
    circuit as partition_reference,
)
from examples.synthesis.applications.physical_systems.ising_model import main as ising
from examples.synthesis.applications.physical_systems.ising_model.comparisons import (
    circuit as ising_reference,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler import estimate_resources
from flagquantum.compiler._construction.frontend_python import _QuantumFunction

pytestmark = pytest.mark.integration
CORPUS = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/corpus/original"


@dataclass(frozen=True)
class Model:
    path: str
    entry: _QuantumFunction
    reference: Callable[[tuple[float, ...]], fq.Circuit]
    build: Callable[[tuple[float, ...]], synth.Program]
    bindings: tuple[tuple[float, ...], ...]
    width: int


MODELS = (
    Model(
        "applications/optimization/max_clique/max_clique.qmod",
        clique.main,
        clique_reference.reference_circuit,
        clique.build_program,
        clique_reference.BINDINGS,
        7,
    ),
    Model(
        "applications/optimization/max_independent_set/max_independent_set.qmod",
        independent.main,
        independent_reference.reference_circuit,
        independent.build_program,
        independent_reference.BINDINGS,
        8,
    ),
    Model(
        "applications/optimization/max_k_vertex_cover/max_k_vertex_cover.qmod",
        cover.main,
        cover_reference.reference_circuit,
        cover.build_program,
        cover_reference.BINDINGS,
        10,
    ),
    Model(
        "applications/optimization/set_partition/set_partition.qmod",
        partition.main,
        partition_reference.reference_circuit,
        partition.build_program,
        partition_reference.BINDINGS,
        10,
    ),
    Model(
        "applications/physical_systems/ising_model/ising_model.qmod",
        ising.main,
        ising_reference.reference_circuit,
        ising.build_program,
        ising_reference.BINDINGS,
        6,
    ),
)


@pytest.mark.parametrize("model", MODELS, ids=lambda model: Path(model.path).stem)
@pytest.mark.parametrize("dataset", [0, 1])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_complete_native_and_original_states_match_independent_cost_circuit(
    model: Model, dataset: int, objective: Literal["width", "depth"]
) -> None:
    params = model.bindings[dataset]
    expected = model.reference(params)
    native = synth.synthesize(
        model.entry,
        classical_inputs={"params": params},
        objective=objective,
        max_workspace=0,
    )
    source = synth.synthesize(
        CORPUS / model.path, classical_inputs={"params": params}, objective=objective
    )
    source_register = "v.z" if model.width == 6 else "v.x"
    for program, register in ((native, "v"), (source, source_register)):
        compare_states(program, expected, (register,))
        assert program.report["workspace_qubits"] == 0
        assert estimate_resources(program.circuit).n_qubits == model.width
    # Changing source grouping can change the chosen physical phase realization;
    # the frozen original resources are checked separately, never overwritten.
    assert (
        estimate_resources(native.circuit).n_operations
        <= estimate_resources(source.circuit).n_operations
    )


@pytest.mark.parametrize("model", MODELS, ids=lambda model: Path(model.path).stem)
def test_native_models_reject_incomplete_parameter_vectors(model: Model) -> None:
    with pytest.raises(ValueError, match="real parameters"):
        model.build(model.bindings[0][:-1])


def test_cover_preserves_uncovered_edges_cardinality_and_constant_phase() -> None:
    assert cover_reference.cost((0,) * 10) == -271
    assert cover_reference.cost((1,) * 10) == -250
    params = (0.031, -0.047, 0.059, 0.0, 0.0, 0.0)
    program = cover.build_program(params)
    state = (
        fq.run(program)
        .statevector()
        .reshape((2,) * 10)
        .permute(program.registers["v"])
        .flatten()
    )
    for basis in (0, 1, 3, 31, 1023):
        bits = tuple((basis >> i) & 1 for i in range(10))
        expected = (
            torch.exp(
                torch.tensor(
                    1j * sum(params[:3]) * cover_reference.cost(bits),
                    dtype=torch.complex128,
                )
            )
            / 32
        )
        torch.testing.assert_close(state[basis], expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("params", [(0.17, -0.23), (0.31, -0.27, 0.19, -0.41)])
def test_cost_reference_matches_dense_diagonal_without_phase_alignment(
    params: tuple[float, ...],
) -> None:
    from examples.synthesis.cost_reference import cost_phase_circuit

    def cost(bits: tuple[int, ...]) -> float:
        return -0.37 + 0.43 * bits[0] - 0.19 * bits[1] + 0.61 * bits[0] * bits[1]

    values = torch.tensor(
        [cost((i & 1, (i >> 1) & 1)) for i in range(4)], dtype=torch.float64
    )
    direct = fq.Circuit(2, dtype=torch.complex128).h(0).h(1)
    layers = len(params) // 2
    for layer in range(layers):
        direct.any(0, 1, matrix=torch.diag(torch.exp(1j * params[layer] * values)))
        direct.rx(0, theta=params[layers + layer]).rx(1, theta=params[layers + layer])
    torch.testing.assert_close(
        fq.run(cost_phase_circuit(cost, 2, params)).statevector(),
        fq.run(direct).statevector(),
        atol=1e-12,
        rtol=0,
    )
