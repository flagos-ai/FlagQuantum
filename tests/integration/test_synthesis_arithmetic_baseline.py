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
    assert (row.resources.allocated_qubits, row.resources.workspace_qubits) == (18, 10)
    assert (row.resources.operations, row.resources.depth) == (166, 124)
    assert row.resources.planned_complex128_state_bytes == 4 * 1024**2
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


def test_helper_expansion_and_exchanged_spelling_record_existing_cost_difference() -> (
    None
):
    from benchmarks.synthesis.arithmetic_baseline import spelling_variants

    helper, exchanged = spelling_variants()
    assert (
        helper.resources.allocated_qubits,
        helper.resources.operations,
        helper.resources.depth,
    ) == (18, 166, 124)
    assert (
        exchanged.resources.allocated_qubits,
        exchanged.resources.operations,
        exchanged.resources.depth,
    ) == (20, 170, 126)
    # AO-00 records this gap; normalization/optimization belongs to later AO PRs.
    assert exchanged.resources.workspace_qubits == 12


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
