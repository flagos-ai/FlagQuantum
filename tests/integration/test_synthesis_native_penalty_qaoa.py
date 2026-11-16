"""Complete diagonal QAOA originals match native helpers and Circuit references."""

import importlib
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Case:
    source: str
    module: str
    width: int
    terms: int
    parameters: int


CASES = (
    Case(
        "applications/cybersecurity/link_monitoring/link_monitoring.qmod",
        "examples.synthesis.applications.cybersecurity.link_monitoring",
        12,
        21,
        2,
    ),
    Case(
        "applications/logistics/task_scheduling_problem/task_scheduling_problem.qmod",
        "examples.synthesis.applications.logistics.task_scheduling_problem",
        24,
        103,
        16,
    ),
    Case(
        "applications/optimization/low_autocorrelation_binary_sequences_problem/evidence_scaling_labs.qmod",
        "examples.synthesis.applications.optimization.low_autocorrelation_binary_sequences_problem.evidence_scaling_labs",
        13,
        162,
        6,
    ),
    Case(
        "applications/optimization/qaoa_in_qaoa/qaoa_in_qaoa.qmod",
        "examples.synthesis.applications.optimization.qaoa_in_qaoa",
        6,
        12,
        4,
    ),
    Case(
        "applications/optimization/rectangles_packing/rectangles_packing_grid.qmod",
        "examples.synthesis.applications.optimization.rectangles_packing.rectangles_packing_grid",
        17,
        102,
        20,
    ),
    Case(
        "applications/telecom/radio_access_network/radio_access_network_positioning_antennas.qmod",
        "examples.synthesis.applications.telecom.radio_access_network.radio_access_network_positioning_antennas",
        16,
        137,
        8,
    ),
    Case(
        "tutorials/technology_demonstrations/qaoa/qaoa_demonstration.qmod",
        "examples.synthesis.tutorials.technology_demonstrations.qaoa.qaoa_demonstration",
        7,
        61,
        20,
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: Path(case.source).stem)
def test_frozen_terms_widths_and_layer_counts_are_retained(case: Case) -> None:
    native = importlib.import_module(case.module + ".main")
    reference = importlib.import_module(case.module + ".comparisons.circuit")
    raw = (
        ROOT / "tests/fixtures/classiq_qmod/corpus/original" / case.source
    ).read_text()
    parsed = []
    for axes, coefficient in re.findall(
        r"PauliTerm\s*\{\s*pauli=\[(.*?)\],\s*coefficient=([^}]+)\}", raw, re.S
    ):
        paulis = re.findall(r"Pauli::([IXYZ])", axes)
        assert len(paulis) == case.width and set(paulis) <= {"I", "Z"}
        parsed.append(
            (
                tuple(i for i, axis in enumerate(paulis) if axis == "Z"),
                float(coefficient),
            )
        )
    assert len(parsed) == case.terms
    assert tuple(parsed) == reference.TERMS
    actual = tuple(
        (tuple(bit for bit, axis in term.factors if axis == "z"), term.coefficient)
        for term in native.HAMILTONIAN.terms
    )
    assert actual == tuple(parsed)
    assert case.parameters == native.PARAMETER_COUNT
    assert case.width == native.WIDTH
    assert all(
        len(params) == case.parameters and all(value != 0 for value in params)
        for params in reference.BINDINGS
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: Path(case.source).stem)
@pytest.mark.parametrize("dataset", [0, 1])
def test_complete_source_native_and_circuit_amplitudes(
    case: Case, dataset: int
) -> None:
    native = importlib.import_module(case.module + ".main")
    reference = importlib.import_module(case.module + ".comparisons.circuit")
    params = reference.BINDINGS[dataset]
    circuit = reference.reference_circuit(params)
    expected = fq.run(circuit).statevector().flatten()
    for entry in (
        native.main,
        ROOT / "tests/fixtures/classiq_qmod/corpus/original" / case.source,
    ):
        program = synth.synthesize(
            entry, classical_inputs={"params_list": params}, max_workspace=0
        )
        resource = program.report["resources"]
        assert isinstance(resource, ResourceEstimate)
        assert resource.n_qubits == case.width
        assert program.report["workspace_qubits"] == 0
        assert tuple(program.registers) == ("target",)
        state = fq.run(program).statevector().reshape((2,) * case.width)
        ordered = state.permute(program.registers["target"]).flatten()
        torch.testing.assert_close(ordered, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("case", CASES, ids=lambda case: Path(case.source).stem)
def test_objectives_preserve_instructions_and_actual_budgets_reject(case: Case) -> None:
    native = importlib.import_module(case.module + ".main")
    reference = importlib.import_module(case.module + ".comparisons.circuit")
    params = reference.BINDINGS[0]
    programs = []
    for objective in ("width", "depth"):
        program = synth.synthesize(
            native.main,
            classical_inputs={"params_list": params},
            objective=objective,
            max_workspace=0,
        )
        programs.append(program)
    assert programs[0].circuit.instructions == programs[1].circuit.instructions
    with pytest.raises(ValueError):
        synth.synthesize(
            native.main,
            classical_inputs={"params_list": params},
            max_qubits=case.width - 1,
        )
    with pytest.raises(ValueError):
        synth.synthesize(
            native.main, classical_inputs={"params_list": params}, max_operations=1
        )
    with pytest.raises(ValueError):
        synth.synthesize(
            native.main, classical_inputs={"params_list": params}, memory_limit_bytes=1
        )
    with pytest.raises(ValueError):
        native.build_program(params[:-1])
