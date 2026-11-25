"""Accept native BB84 preparation, actual observations and teaching postprocessing."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.community.paper_implementation_project.bb84.bb84_implementation.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.community.paper_implementation_project.bb84.bb84_implementation.main import (
    ALICE_BASES,
    BOB_BASES,
    MESSAGE,
    main,
)
from examples.synthesis.community.paper_implementation_project.bb84.bb84_implementation.overview.workflow import (
    Workflow,
    observe,
    postprocess,
    run_workflow,
)
from flagquantum import synth
from flagquantum.errors import PlanningError
from flagquantum.synth import Output, QBit, allocate, qfunc
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration

ZERO = (0,) * 8


def test_original_signal_inputs_and_matched_basis_positions() -> None:
    result = run_workflow()
    assert result.report.matching_positions == (0, 2, 5)
    assert result.report.alice_raw_key == (0, 1, 0)
    assert result.report.bob_raw_key == (0, 1, 0)
    assert result.report.error_rate == 0.0
    assert result.report.corrected_key == (0, 1, 0)
    assert result.report.final_key == (1,)
    assert result.selected == result.observations[0]
    assert sum(row.count for row in result.observations) == 2048
    for row in result.observations:
        assert all(row.bits[i] == MESSAGE[i] for i in (0, 2, 5))


def test_sampling_reproducibility_uses_real_named_readout() -> None:
    assert run_workflow().observations == run_workflow().observations
    assert run_workflow(seed=4).observations != run_workflow().observations


def test_no_matching_basis_aborts_explicitly() -> None:
    result = postprocess(ZERO, message=ZERO, alice_bases=ZERO, bob_bases=(1,) * 8)
    assert result.matching_positions == result.alice_raw_key == result.bob_raw_key == ()
    assert result.error_rate == 1.0 and result.final_key is None


def test_too_few_sifted_bits_produces_empty_parity_without_fake_bits() -> None:
    result = postprocess(
        ZERO, message=ZERO, alice_bases=ZERO, bob_bases=(0,) + (1,) * 7
    )
    assert result.corrected_key == (0,) and result.final_key == ()


def test_error_threshold_retains_original_teaching_behavior() -> None:
    below = postprocess((1,) + (0,) * 7, message=ZERO, alice_bases=ZERO, bob_bases=ZERO)
    assert below.error_rate == 0.125
    assert below.corrected_key == (0,) * 7 and below.final_key == (0, 0, 0)
    above = postprocess(
        (1, 1) + (0,) * 6, message=ZERO, alice_bases=ZERO, bob_bases=ZERO
    )
    assert (
        above.error_rate == 0.25
        and above.corrected_key == ()
        and above.final_key is None
    )


@pytest.mark.parametrize("values", ((0,), (0,) * 9, (2,) * 8, (True,) * 8))
@pytest.mark.parametrize(
    "name", ("message", "alice_bases", "bob_bases", "bob_measurements")
)
def test_invalid_protocol_values_fail(values: tuple[int, ...], name: str) -> None:
    bindings: dict[str, tuple[int, ...]] = dict(
        message=MESSAGE,
        alice_bases=ALICE_BASES,
        bob_bases=BOB_BASES,
        bob_measurements=ZERO,
    )
    bindings[name] = values
    with pytest.raises(ValueError, match=name):
        postprocess(**bindings)


@pytest.mark.parametrize(
    "shots,seed",
    ((0, 3), (100001, 3), (True, 3), (2048, -1), (2048, True), (2048, 2**63)),
)
def test_sampling_controls_fail(shots: int, seed: int) -> None:
    with pytest.raises(ValueError):
        run_workflow(shots=shots, seed=seed)


@pytest.mark.parametrize(
    "message,alice,bob",
    (
        (MESSAGE, ALICE_BASES, BOB_BASES),
        ((1,) * 8, ZERO, ZERO),
        ((0, 1) * 4, (0, 1) * 4, (1, 0) * 4),
    ),
)
def test_independent_full_complex_states_and_named_readout(
    message: tuple[int, ...], alice: tuple[int, ...], bob: tuple[int, ...]
) -> None:
    workflow = run_workflow(message=message, alice_bases=alice, bob_bases=bob)
    program = workflow.program
    actual = (
        fq.run(program)
        .statevector()
        .reshape((2,) * 8)
        .permute(program.registers["qba"])
        .flatten()
    )
    expected = fq.run(reference_circuit(message, alice, bob)).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    readout = dict(program.probabilities(fq.run(program), registers=("qba",)))
    for word, probability in enumerate(expected.abs().square()):
        assert readout.get((word,), 0.0) == pytest.approx(float(probability), abs=1e-12)
    for row in workflow.observations:
        assert all(row.bits[i] == message[i] for i in range(8) if alice[i] == bob[i])
    assert sum(row.count for row in workflow.observations) == 2048
    assert program.circuit.n_wires == 8 and program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("index", range(8))
def test_each_numeric_bit_is_the_same_logical_signal(index: int) -> None:
    message = tuple(int(i == index) for i in range(8))
    workflow = run_workflow(message=message, alice_bases=ZERO, bob_bases=ZERO)
    assert len(workflow.observations) == 1
    assert workflow.selected.word == 1 << index
    assert workflow.selected.bits == message and workflow.selected.count == 2048


def test_clean_notebook_executes_complete_original_workflow() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/community/paper_implementation_project/bb84/bb84_implementation/overview/walkthrough.ipynb"
    )
    notebook = json.loads(path.read_text())
    assert all(
        cell.get("execution_count") is None and not cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    namespace, count = execute_code_cells(path)
    assert count == 4
    workflow = namespace["workflow"]
    assert isinstance(workflow, Workflow)
    assert sum(row.count for row in workflow.observations) == 2048
    assert workflow.report.matching_positions == (0, 2, 5)
    assert workflow.report.alice_raw_key == workflow.report.bob_raw_key == (0, 1, 0)
    assert workflow.report.error_rate == 0 and workflow.report.final_key == (1,)


@pytest.mark.parametrize("budget", ("width", "operations", "depth", "memory"))
def test_native_program_resource_refusals(budget: str) -> None:
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if budget == "width":
            synth.synthesize(main, max_qubits=7)
        elif budget == "operations":
            synth.synthesize(main, max_operations=1)
        elif budget == "depth":
            synth.synthesize(main, max_depth=0)
        else:
            synth.synthesize(main, memory_limit_bytes=1)


@qfunc
def wrong_port(q: Output[QBit]) -> None:
    allocate(q)


def test_incompatible_named_register_refused() -> None:
    with pytest.raises(ValueError, match="eight-bit qba"):
        observe(synth.synthesize(wrong_port))
