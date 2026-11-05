"""Full original Pauli workloads, sequence integrity and actual resource bounds."""

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem._qmod.adapter import SUPPORTED_SOURCE_VERSION, parse_qmod
from tools.validate_qmod_large_pauli import FIXTURES, validate

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("case", ["example_exponentiation", "hamiltonian_evolution"])
def test_original_large_pauli_full_state_and_exact_sample_sequence(case: str) -> None:
    report = validate(case)
    assert report.term_count == 551 and report.qubits == 12 and report.workspace == 0
    assert report.source_pauli_factors == (
        2000 if case == "hamiltonian_evolution" else 551
    )
    assert report.all_owned_rotations_match_reference_sequence
    if case == "hamiltonian_evolution":
        assert report.seed == 0 and report.sample_count == 2000
        assert len(report.sampled_indices) == 2000
        assert (
            report.sampled_indices_sha256
            == "e5e359ee6b71f74c13ee5cebe72575f3bed29ffbe2b931dafc78f1aff542e988"
        )


@pytest.mark.parametrize("count", [1025, 2000, 2048])
def test_sample_boundaries_preserve_all_factors(count: int) -> None:
    source = f"""qfunc main(output q:qbit[1]) {{allocate(q);
      qdrift(SparsePauliOp {{terms=[SparsePauliTerm {{paulis=[IndexedPauli {{pauli=Pauli::Y,index=0}}],coefficient=-0.5}}],num_qubits=1}},0.37,{count},q);
    }}"""
    imported = parse_qmod(
        source, filename="samples.qmod", source_version=SUPPORTED_SOURCE_VERSION
    )
    assert (
        sum(
            op.name == "qreg.pauli_rotation"
            for op in imported.program.body.blocks[0].operations
        )
        == count
    )
    program = synth.synthesize(source, source_format="qmod")
    reference = fq.Circuit(1, dtype=torch.complex128).ry(0, theta=-0.37)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("constraint", "value"),
    [
        ("max_qubits", 11),
        ("max_operations", 100),
        ("max_depth", 100),
        ("memory_limit_bytes", 1024),
    ],
)
def test_large_pauli_programs_keep_final_resource_constraints(
    constraint: str, value: int
) -> None:
    with pytest.raises(
        ValueError, match="budget|qubit|operation|depth|memory|limit|candidate"
    ):
        synth.synthesize(
            FIXTURES / "example_exponentiation.qmod", **{constraint: value}
        )


def test_inverse_helper_uses_negated_compact_angles_without_losing_global_phase() -> (
    None
):
    source = """qfunc evolve(q:qbit[]) {
      suzuki_trotter([PauliTerm {pauli=[Pauli::Y],coefficient=-0.37},PauliTerm {pauli=[Pauli::I],coefficient=0.21}],0.5,2,3,q);
    }
    qfunc main(output q:qbit[1]) {allocate(q);H(q[0]);evolve(q);invert {evolve(q);}}
    """
    program = synth.synthesize(source, source_format="qmod")
    expected = fq.Circuit(1, dtype=torch.complex128).h(0)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(expected).statevector(),
        atol=1e-12,
        rtol=0,
    )


def test_original_large_pauli_is_deterministic() -> None:
    path = FIXTURES / "hamiltonian_evolution.qmod"
    left = synth.synthesize(path)
    right = synth.synthesize(path)
    assert left.circuit.instructions == right.circuit.instructions


def test_three_form_user_example() -> None:
    from examples.synthesis_large_pauli import run_example

    run_example()
