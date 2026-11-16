"""AO-00 executable workflow, coherent cleanup and independent XOR checks."""

from dataclasses import replace

import pytest

from benchmarks.synthesis.arithmetic_baseline import (
    measure,
    sentinel_targets,
    synthesize_case,
)
from benchmarks.synthesis.expressions import EXPRESSIONS, Case
from benchmarks.synthesis.verification import basis_checks, coherent_check
from flagquantum.core.ir import Instruction

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("expression", EXPRESSIONS, ids=lambda row: row.name)
def test_all_expression_families_have_exact_coherent_and_basis_oracles(
    expression,
) -> None:
    row = measure(Case(expression, (1,) * len(expression.arguments)))
    assert row.status == "compiled"
    assert row.verification is not None
    assert row.verification.basis_domain_exhaustive
    assert row.verification.coherent_states == 2
    assert row.verification.workspace_nonzero_probability == 0


def test_energy_all_256_input_target_pairs_and_full_complex_states() -> None:
    row = measure(Case(EXPRESSIONS[4], (2, 2)))
    assert row.resources is not None and row.verification is not None
    assert (row.resources.allocated_qubits, row.resources.workspace_qubits) == (16, 8)
    assert (row.resources.operations, row.resources.depth) == (81, 59)
    assert row.resources.planned_complex128_state_bytes == 1024**2
    assert row.verification.basis_checks == 256
    assert row.verification.coherent_states == 2
    assert row.verification.maximum_amplitude_error < 1e-12


def test_oracles_detect_dirty_workspace_wrong_outputs_and_relative_phases() -> None:
    case = Case(EXPRESSIONS[4], (1, 1))
    program = synthesize_case(case)
    row = program.selection.selected
    assert row is not None and row.arithmetic is not None
    dirty = replace(
        program.arithmetic,
        instructions=(
            *program.arithmetic.instructions,
            Instruction("x", (program.circuit.n_wires - 1,)),
        ),
    )
    with pytest.raises(AssertionError, match="mismatch"):
        basis_checks(case, replace(program, arithmetic=dirty))
    with pytest.raises(AssertionError, match="amplitude"):
        coherent_check(case, program, phased=True)


def test_larger_case_is_static_and_does_not_execute_a_quantum_state(
    monkeypatch,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("large static case must not execute")

    monkeypatch.setattr("flagquantum.run", forbidden)
    row = measure(Case(EXPRESSIONS[4], (8, 8)))
    assert row.status == "compiled"
    assert row.verification is not None
    assert not row.verification.basis_domain_exhaustive
    assert row.verification.coherent_states == 0
    assert row.verification.maximum_amplitude_error is None


@pytest.mark.usefixtures("legacy_register_selection")
def test_qmod_manual_and_python_costs_include_preparation_and_target_rewrites() -> None:
    rows = sentinel_targets()
    assert len(rows) == 9
    by_key = {(row.implementation, row.profile): row.resources for row in rows}
    for profile in ("logical", "exact_basis_all_to_all", "exact_basis_line"):
        native, qmod = by_key["python", profile], by_key["qmod", profile]
        assert replace(native, circuit_hash="") == replace(qmod, circuit_hash="")
    manual = by_key["manual", "logical"]
    assert (
        manual.allocated_qubits,
        manual.workspace_qubits,
        manual.operations,
        manual.depth,
    ) == (9, 1, 15, 11)
    assert (
        by_key["python", "exact_basis_all_to_all"].operation_counts.get("ccx", 0) == 0
    )
    assert (
        by_key["python", "exact_basis_line"].two_qubit_operations
        > by_key["python", "exact_basis_all_to_all"].two_qubit_operations
    )


def test_helper_expansion_and_exchanged_spelling_improve_without_rewriting_history() -> (
    None
):
    from benchmarks.synthesis.arithmetic_baseline import spelling_variants

    helper, exchanged = spelling_variants()
    assert (
        helper.resources.allocated_qubits,
        helper.resources.operations,
        helper.resources.depth,
    ) == (16, 81, 59)
    assert (
        exchanged.resources.allocated_qubits,
        exchanged.resources.operations,
        exchanged.resources.depth,
    ) == (16, 81, 59)
    # AO-01 closes this spelling gap; the checked AO-00 measurement stays frozen.
    import json
    from pathlib import Path

    frozen = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "benchmarks/synthesis/baselines/ao00.json"
        ).read_text()
    )
    old = frozen["spelling_variants"][1]["resources"]
    assert (
        old["allocated_qubits"],
        old["workspace_qubits"],
        old["operations"],
        old["depth"],
    ) == (20, 12, 170, 126)
    assert exchanged.resources.workspace_qubits == 8


def test_independent_manual_reference_preserves_all_256_input_target_pairs() -> None:
    from itertools import product

    from benchmarks.synthesis.arithmetic_baseline import manual_reference
    from benchmarks.synthesis.verification import classical_apply, encode

    reference = manual_reference()
    oracle_circuit = replace(
        reference,
        instructions=tuple(g for g in reference.instructions if g.name != "h"),
    )
    wires = ((0, 1), (2, 3), (4, 5, 6, 7))
    for x, y, target in product(range(4), range(4), range(16)):
        assert classical_apply(oracle_circuit, encode((x, y, target), wires)) == encode(
            (x, y, target ^ (x * x + y)), wires
        )


def test_primary_qfunc_example_prepares_composes_and_improves_frozen_costs() -> None:
    from benchmarks.synthesis.arithmetic_baseline import costs, workspace_of
    from examples.synthesis.tutorials.arithmetic.arithmetic_baseline.main import (
        main,
    )
    from flagquantum import synth

    case = Case(EXPRESSIONS[4], (2, 2))
    native = synth.synthesize(main, memory_limit_bytes=64 * 1024**2)
    from pathlib import Path

    original = (
        Path(__file__).resolve().parents[1]
        / "fixtures/classiq_qmod/teaching_sources/tutorials/arithmetic/arithmetic_implementations/comparisons/source.qmod"
    )
    migrated = synth.synthesize(original, memory_limit_bytes=64 * 1024**2)
    for program in (native, migrated):
        # The full state checks all sixteen logical outcomes and zero workspace.
        coherent_check(case, program, phased=False)
        resource = costs(program.circuit, workspace_of(program))
        assert (
            resource.allocated_qubits,
            resource.workspace_qubits,
            resource.operations,
            resource.depth,
        ) == (9, 1, 14, 9)
