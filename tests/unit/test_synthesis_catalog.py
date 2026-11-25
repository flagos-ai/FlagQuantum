"""Classified examples retain source identity, honest coverage and executable cases."""

import ast
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from tools.run_qmod_catalog import run_case
from tools.synthesis_catalog import (
    CATALOG,
    ROOT,
    case_readme,
    example_path,
    load_catalog,
    render,
    verify_catalog,
)

pytestmark = pytest.mark.unit


def test_native_user_library_contains_no_importer_sources_or_calls() -> None:
    library = ROOT / "examples/synthesis"
    assert not list(library.rglob("*.qmod"))
    for path in library.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("flagquantum.")
            ):
                assert not any(
                    part.startswith("_") for part in node.module.split(".")
                ), path
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not re.search(
                    r"\bqfunc\s+\w+\s*\([^)]*\)\s*\{", node.value
                ), path
            if isinstance(node, ast.Call):
                assert not any(
                    keyword.arg == "source_format"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == "qmod"
                    for keyword in node.keywords
                ), path


def test_expanded_inventory_separates_quantum_and_workflow_acceptance() -> None:
    data = json.loads((CATALOG.parent / "native_library_inventory.json").read_text())
    catalog_paths = {
        case["native_example"].casefold(): case["native_example"]
        for case in load_catalog()["cases"]
        if case["native_example"]
    }
    assert set(data["categories"]) == {
        "algorithms",
        "applications",
        "community",
        "functions",
        "tutorials",
    }
    groups = data["groups"]
    assert len(groups) == 417 == len({group["id"] for group in groups})
    sources = [source for group in groups for source in group["sources"]]
    assert len(sources) == 577 == len({source["path"] for source in sources})
    assert Counter(Path(source["path"]).suffix for source in sources) == {
        ".qmod": 216,
        ".ipynb": 213,
        ".py": 148,
    }
    for group in groups:
        assert group["category"] == Path(group["id"]).parts[0]
        assert group["workflow_status"] in {"pending", "validated"}
        if group["workflow_status"] == "validated":
            proof = group["workflow_validation"]
            assert group["native_program"]
            assert proof["reviewed_code_cells"]
            assert (ROOT / proof["acceptance_test"]).is_file()
            notebook = next(
                source for source in group["sources"] if source["kind"] == "notebook"
            )
            assert proof["git_blob_sha1"] == notebook["git_blob_sha1"]
            assert proof["upstream_commit"] == data["upstream_commit"]
            provenance = ROOT / group["native_program"]["path"]
            assert (
                json.loads(provenance.with_name("provenance.json").read_text()) == proof
            )
        if group["native_program"]:
            native_path = group["native_program"]["path"]
            assert (ROOT / native_path).is_file()
            if native_path.casefold() in catalog_paths:
                assert native_path == catalog_paths[native_path.casefold()]


def test_complete_catalog_preserves_sources_and_generated_navigation() -> None:
    catalog = load_catalog()
    verify_catalog(catalog)
    for path, text in render(catalog).items():
        assert path.read_text(encoding="utf-8") == text
    assert {case["category"] for case in catalog["cases"]} == {
        "tutorials",
        "algorithms",
        "applications",
        "community",
        "functions",
    }
    assert CATALOG.is_file()


@pytest.mark.parametrize(
    "source",
    [
        "def reference_circuit() -> None:\n    return None\n",
        'if __name__ == "__main__":\n    pass\n',
    ],
)
def test_catalog_rejects_reference_only_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    catalog = load_catalog()
    comparison = catalog["cases"][0]["native_comparison"]
    assert comparison is not None
    reference_only = tmp_path / "circuit.py"
    reference_only.write_text(source)

    def comparison_path(path: str) -> Path:
        return reference_only if path == comparison else example_path(path)

    monkeypatch.setattr("tools.synthesis_catalog.example_path", comparison_path)
    with pytest.raises(ValueError, match="executable comparison"):
        verify_catalog(catalog)


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "source_digest", "bindings", "revision"]
)
def test_catalog_rejects_incomplete_or_mislabelled_examples(change: str) -> None:
    catalog = deepcopy(load_catalog())
    if change == "missing":
        catalog["cases"].pop()
    elif change == "duplicate":
        catalog["cases"][-1] = catalog["cases"][0]
    elif change == "source_digest":
        catalog["cases"][0]["original"]["sha256"] = "0" * 64
    elif change == "bindings":
        catalog["cases"][0]["classical_inputs"] = {"invented": 0.4}
    else:
        catalog["cases"][0]["approved"]["revised"] = True
    with pytest.raises(ValueError):
        verify_catalog(catalog)


def test_native_coverage_requires_the_exact_original_source() -> None:
    catalog = deepcopy(load_catalog())
    native = next(case for case in catalog["cases"] if case["native_example"])
    other = next(case for case in catalog["cases"] if case["id"] != native["id"])
    other["native_example"] = native["native_example"]
    other["native_comparison"] = native["native_comparison"]
    with pytest.raises(ValueError, match="not tied to this original"):
        verify_catalog(catalog)


def test_example_assets_cannot_escape_the_catalog_directory() -> None:
    with pytest.raises(ValueError, match="outside the synthesis examples"):
        example_path("examples/synthesis/../../../source.qmod")


def test_original_and_approved_swap_inputs_have_distinct_results() -> None:
    catalog = load_catalog()
    case = next(
        case
        for case in catalog["cases"]
        if case["id"] == "algorithms/quantum_primitives/swap_test/swap_test.qmod"
    )
    original = run_case(case, catalog, revision="original")
    approved = run_case(case, catalog, revision="approved")
    assert original["stage"] == "frontend_refused"
    assert approved["stage"] == "execution_smoke_passed"
    assert original["source_sha256"] != approved["source_sha256"]
    assert approved["operations"] == case["recorded_approved"]["operations"]


def test_original_simple_case_uses_its_recorded_resources() -> None:
    catalog = load_catalog()
    case = next(
        case
        for case in catalog["cases"]
        if case["id"].endswith("bernstein_vazirani.qmod")
    )
    observed = run_case(case, catalog, revision="original")
    assert observed["stage"] == "execution_smoke_passed"
    for field in ("qubits", "operations", "depth"):
        assert observed[field] == case["recorded_approved"][field]


def test_companion_preferences_keep_the_source_basename() -> None:
    catalog = load_catalog()
    case = next(case for case in catalog["cases"] if case["companion_files"])
    case["companion_files"][0]["path"] = catalog["license"]["path"]
    case["companion_files"][0]["path"] += ".synthesis_options.json"
    with pytest.raises(ValueError, match="companion preferences"):
        verify_catalog(catalog)


def test_autocallable_companion_precision_is_consumed_after_relocation() -> None:
    catalog = load_catalog()
    case = next(case for case in catalog["cases"] if case["companion_files"])
    observed = run_case(case, catalog, revision="original")
    assert observed["stage"] == "execution_smoke_passed"
    for field in ("qubits", "operations", "depth"):
        assert observed[field] == case["recorded_original"][field]


@pytest.mark.parametrize("validated", [False, True])
def test_workflow_guide_does_not_determine_notebook_acceptance(validated: bool) -> None:
    catalog = load_catalog()
    case = next(
        case
        for case in catalog["cases"]
        if case["id"].endswith("/facility_location.qmod")
    )
    assert (example_path(case["directory"]) / "WORKFLOW.md").is_file()
    text = case_readme(case, catalog, notebook_validated=validated)
    assert ("complete local Notebook workflow are validated" in text) == validated
    assert ("quantum-program guide" in text) == (not validated)


def test_rendered_workflow_claims_follow_inventory_acceptance() -> None:
    catalog = load_catalog()
    inventory = json.loads(
        (CATALOG.parent / "native_library_inventory.json").read_text()
    )
    validated = {
        (group["id"], group["native_program"]["path"])
        for group in inventory["groups"]
        if group["kind"] == "notebook_workflow"
        and group["workflow_status"] == "validated"
        and group["native_program"]
    }
    assert len(validated) == 212
    assert (
        "community/basic_examples/logical_qubits/logical_qubits_by_alice_and_bob",
        "examples/synthesis/community/basic_examples/logical_qubits/logical_qubits_by_alice_and_bob/overview/main.py",
    ) in validated
    assert (
        "applications/chemistry/projection_based_embedding/projected_based_embedding_tutorial",
        "examples/synthesis/applications/chemistry/projection_based_embedding/projected_based_embedding_tutorial/overview/main.py",
    ) in validated
    assert (
        "applications/physical_systems/maxwell_equation/maxwell_2d_simulation",
        "examples/synthesis/applications/physical_systems/maxwell_equation/maxwell_2d_simulation/overview/main.py",
    ) in validated
    assert (
        "applications/plasma/vlasov_ampere/vlasov_ampere",
        "examples/synthesis/applications/plasma/vlasov_ampere/overview/main.py",
    ) in validated
    assert (
        "applications/plasma/vlasov_ampere/vlasov_ampere_qiskit",
        "examples/synthesis/applications/plasma/vlasov_ampere/vlasov_ampere_qiskit/overview/main.py",
    ) in validated
    assert (
        "applications/telecom/resiliency_planning/resiliency_planning_AMD",
        "examples/synthesis/applications/telecom/resiliency_planning/resiliency_planning_amd/overview/main.py",
    ) in validated
    assert (
        "applications/telecom/resiliency_planning/resiliency_planning",
        "examples/synthesis/applications/telecom/resiliency_planning/overview/main.py",
    ) in validated
    assert (
        "applications/logistics/task_scheduling_problem/task_scheduling_problem",
        "examples/synthesis/applications/logistics/task_scheduling_problem/overview/main.py",
    ) in validated
    assert (
        "community/Hackathons/QInnovision_2025/advection_equation/advection_equation_winning_submission",
        "examples/synthesis/community/hackathons/qinnovision_2025/advection_equation/winning/overview/main.py",
    ) in validated
    assert (
        "community/Hackathons/QInnovision_2025/advection_equation/advection_equation_2nd_place_submission",
        "examples/synthesis/community/hackathons/qinnovision_2025/advection_equation/second_place/overview/main.py",
    ) in validated
    assert (
        "applications/telecom/network_traffic_optimization/network_traffic_optimization",
        "examples/synthesis/applications/telecom/network_traffic_optimization/overview/main.py",
    ) in validated
    assert (
        "applications/logistics/vehicle_routing_problem/vehicle_routing_problem",
        "examples/synthesis/applications/logistics/vehicle_routing_problem/overview/main.py",
    ) in validated
    assert (
        "applications/chemistry/tensor_hypercontraction/tensor_hypercontraction",
        "examples/synthesis/applications/chemistry/tensor_hypercontraction/overview/main.py",
    ) in validated
    assert (
        "applications/chemistry/protein_folding/protein_folding_with_quantum_walk/qfold",
        "examples/synthesis/applications/chemistry/protein_folding/protein_folding_with_quantum_walk/qfold/overview/main.py",
    ) in validated
    assert (
        "tutorials/basic_tutorials/the_classiq_tutorial/execution_tutorial",
        "examples/synthesis/tutorials/basics/quantum_programming/execution_tutorial/overview/main.py",
    ) in validated
    assert (
        "tutorials/basic_tutorials/the_classiq_tutorial/execution_tutorial_part2",
        "examples/synthesis/tutorials/basics/quantum_programming/execution_tutorial_part2/overview/main.py",
    ) in validated
    assert (
        "community/Hackathons/iQuHack_2025/Challenge/classiq_iQuHack_2025_final",
        "examples/synthesis/community/hackathons/iquhack_2025/challenge_template/overview/main.py",
    ) in validated
    assert (
        "community/Hackathons/iQuHack_2025/Challenge_solution/our_solution/classiq_iQuHack_2025_final_sol",
        "examples/synthesis/community/hackathons/iquhack_2025/gaussian_reference/overview/main.py",
    ) in validated
    assert (
        "community/Hackathons/iQuHack_2025/Challenge_solution/winning_solution/classiq_iQuHack_2025_quantum_tree_sol",
        "examples/synthesis/community/hackathons/iquhack_2025/winning_gaussian/overview/main.py",
    ) in validated
    assert (
        "applications/chemistry/classiq_chemistry_application/classiq_chemistry_application",
        "examples/synthesis/applications/chemistry/lih_chemistry/overview/main.py",
    ) in validated
    assert (
        "community/Hackathons/iQuHack_2025/Workshop/WS_iQuHack_2025_final",
        "examples/synthesis/community/hackathons/iquhack_2025/workshop/overview/main.py",
    ) in validated
    assert (
        "algorithms/QML/qgan/qgan_bars_and_strips",
        "examples/synthesis/algorithms/machine_learning/qgan/qgan_bars_and_strips/overview/main.py",
    ) in validated
    assert (
        "tutorials/basic_tutorials/mcx/mcx",
        "examples/synthesis/tutorials/basics/mcx/overview/main.py",
    ) in validated
    assert (
        "community/basic_examples/hw_aware_synthesis/hw_aware_synthesis",
        "examples/synthesis/community/basic_examples/hw_aware_synthesis/hw_aware_synthesis/overview/main.py",
    ) in validated
    assert (
        "algorithms/number_theory_and_cryptography/elliptic_curves/elliptic_curve_discrete_log",
        "examples/synthesis/algorithms/number_theory_and_cryptography/elliptic_curves/elliptic_curve_discrete_log/overview/main.py",
    ) in validated
    assert (
        "community/paper_implementation_project/bpde/bpde",
        "examples/synthesis/community/paper_implementation_project/bpde/overview/main.py",
    ) in validated
    assert (
        "applications/finance/credit_card_fraud/credit_card_fraud",
        "examples/synthesis/applications/finance/credit_card_fraud/overview/main.py",
    ) in validated
    assert (
        "applications/optimization/low_autocorrelation_binary_sequences_problem/evidence_scaling_labs",
        "examples/synthesis/applications/optimization/low_autocorrelation_binary_sequences_problem/evidence_scaling_labs/overview/main.py",
    ) in validated
    assert (
        "tutorials/technology_demonstrations/discrete_quantum_walk_circle/discrete_quantum_walk_circle",
        "examples/synthesis/tutorials/technology_demonstrations/discrete_quantum_walk_circle/overview/main.py",
    ) in validated
    assert (
        "tutorials/advanced_tutorials/discrete_quantum_walk/discrete_quantum_walk",
        "examples/synthesis/tutorials/advanced_tutorials/discrete_quantum_walk/overview/main.py",
    ) in validated
    assert (
        "algorithms/QML/qsvm/qsvm",
        "examples/synthesis/algorithms/machine_learning/qsvm/overview/main.py",
    ) in validated
    assert (
        "tutorials/technology_demonstrations/oracle_generation/3sat_oracles",
        "examples/synthesis/tutorials/technology_demonstrations/oracle_generation/three_sat_oracles/overview/main.py",
    ) in validated
    assert (
        "applications/finance/autocallable_options/quantum_autocallable_option_pricing",
        "examples/synthesis/applications/finance/autocallable_options/quantum_autocallable_option_pricing/overview/main.py",
    ) in validated
    assert (
        "applications/finance/brownian_chebyshev_polynomials/brownian_chebyshev_polynomials",
        "examples/synthesis/applications/finance/brownian_chebyshev_polynomials/overview/main.py",
    ) in validated
    assert (
        "applications/automotive/cooling_systems_optimization/cooling_systems_optimization",
        "examples/synthesis/applications/automotive/cooling_systems_optimization/overview/main.py",
    ) in validated
    assert (
        "applications/CFD/double_slit_experiment/quantum_double_slit_experiment",
        "examples/synthesis/applications/cfd/double_slit_experiment/quantum_double_slit_experiment/overview/main.py",
    ) in validated
    assert (
        "community/paper_implementation_project/explicit_quantum_circuits_for_block_encoding/quantum_walks_via_efficient_blockencoding",
        "examples/synthesis/community/paper_implementation_project/explicit_quantum_circuits_for_block_encoding/quantum_walks_via_efficient_blockencoding/overview/main.py",
    ) in validated
    assert (
        "algorithms/quantum_differential_equations_solvers/time_marching/time_marching",
        "examples/synthesis/algorithms/quantum_differential_equations_solvers/time_marching/overview/main.py",
    ) in validated
    assert (
        "tutorials/technology_demonstrations/hamiltonian_evolution/hamiltonian_evolution",
        "examples/synthesis/tutorials/technology_demonstrations/hamiltonian_evolution/overview/main.py",
    ) in validated
    assert (
        "applications/image_processing/quantum_hadamard_edge_detection/quantum_image_edge_detection",
        "examples/synthesis/applications/image_processing/quantum_hadamard_edge_detection/overview/main.py",
    ) in validated
    assert (
        "tutorials/basic_tutorials/the_classiq_tutorial/Qmod_tutorial_part2",
        "examples/synthesis/tutorials/basics/quantum_programming/qmod_tutorial_part2/overview/main.py",
    ) in validated
    assert (
        "community/paper_implementation_project/quantum_state_preparation_without_coherent_arithmetic/stateprep_guassian_using_qsvt",
        "examples/synthesis/community/paper_implementation_project/quantum_state_preparation_without_coherent_arithmetic/stateprep_guassian_using_qsvt/overview/main.py",
    ) in validated
    assert (
        "algorithms/quantum_primitives/quantum_oracle_sketching_boolean/quantum_oracle_sketching_boolean",
        "examples/synthesis/algorithms/quantum_primitives/quantum_oracle_sketching_boolean/overview/main.py",
    ) in validated
    assert (
        "algorithms/hamiltonian_simulation/hamiltonian_simulation_with_block_encoding/hamiltonian_simulation_qsvt",
        "examples/synthesis/algorithms/hamiltonian_simulation/hamiltonian_simulation_with_block_encoding/hamiltonian_simulation_qsvt/overview/main.py",
    ) in validated
    assert (
        "algorithms/quantum_linear_solvers/qsvt_matrix_inversion/qsvt_matrix_inversion",
        "examples/synthesis/algorithms/quantum_linear_solvers/qsvt_matrix_inversion/overview/main.py",
    ) in validated
    assert (
        "applications/CFD/QLS_for_hybrid_solvers/qls_qsvt",
        "examples/synthesis/applications/cfd/qls_for_hybrid_solvers/qls_qsvt/overview/main.py",
    ) in validated
    assert (
        "applications/CFD/QLS_for_hybrid_solvers/qls_chebyshev_lcu",
        "examples/synthesis/applications/cfd/qls_for_hybrid_solvers/qls_chebyshev_lcu/overview/main.py",
    ) in validated
    assert (
        "applications/CFD/QLS_for_hybrid_solvers/chebyshev_approximation",
        "examples/synthesis/applications/cfd/qls_for_hybrid_solvers/chebyshev_approximation/overview/main.py",
    ) in validated
    assert (
        "algorithms/quantum_phase_estimation/qpe_with_qubitization/qpe_for_molecule_with_qubitization",
        "examples/synthesis/algorithms/quantum_phase_estimation/qpe_with_qubitization/qpe_for_molecule_with_qubitization/overview/main.py",
    ) in validated
    assert (
        "applications/chemistry/qpe_for_molecules/qpe_for_molecules",
        "examples/synthesis/applications/chemistry/qpe_for_molecules/overview/main.py",
    ) in validated
    assert (
        "applications/chemistry/molecule_eigensolver/molecule_eigensolver",
        "examples/synthesis/applications/chemistry/molecule_eigensolver/overview/main.py",
    ) in validated
    assert (
        "applications/chemistry/molecular_energy_curve/molecular_energy_curve",
        "examples/synthesis/applications/chemistry/molecular_energy_curve/overview/main.py",
    ) in validated
    assert (
        "tutorials/basic_tutorials/exponentiation/example_exponentiation",
        "examples/synthesis/tutorials/basics/exponentiation/example_exponentiation/overview/main.py",
    ) in validated
    assert (
        "applications/CFD/heat_eq_qsvt/heat_eq_qsvt",
        "examples/synthesis/applications/cfd/heat_eq_qsvt/overview/main.py",
    ) in validated
    assert (
        "tutorials/technology_demonstrations/qaoa/qaoa_demonstration",
        "examples/synthesis/tutorials/technology_demonstrations/qaoa/qaoa_demonstration/overview/main.py",
    ) in validated
    assert (
        "applications/finance/hybrid_hhl_for_portfolio_optimization/portfolio_optimization_with_hhl",
        "examples/synthesis/applications/finance/hybrid_hhl_for_portfolio_optimization/portfolio_optimization_with_hhl/overview/main.py",
    ) in validated
    assert (
        "applications/finance/portfolio_optimization_hhl/HHL_portfolio",
        "examples/synthesis/applications/finance/portfolio_optimization_hhl/hhl_portfolio/overview/main.py",
    ) in validated
    assert (
        "applications/finance/rainbow_options/rainbow_options_integration_method",
        "examples/synthesis/applications/finance/rainbow_options/rainbow_options_integration_method/overview/main.py",
    ) in validated
    assert (
        "tutorials/workshops/finance_workshops/rainbow_options_workshop_bruteforce",
        "examples/synthesis/tutorials/workshops/finance_workshops/rainbow_options_workshop_bruteforce/overview/main.py",
    ) in validated
    assert (
        "applications/finance/rainbow_options/rainbow_options_direct_method",
        "examples/synthesis/applications/finance/rainbow_options/rainbow_options_direct_method/overview/main.py",
    ) in validated
    assert (
        "applications/finance/rainbow_options/rainbow_options_bruteforce_method",
        "examples/synthesis/applications/finance/rainbow_options/rainbow_options_bruteforce_method/overview/main.py",
    ) in validated
    assert (
        "applications/chemistry/protein_folding/protein_folding_with_qaoa/protein_folding",
        "examples/synthesis/applications/chemistry/protein_folding/protein_folding_with_qaoa/protein_folding/overview/main.py",
    ) in validated
    assert (
        "algorithms/quantum_differential_equations_solvers/lchs/lchs",
        "examples/synthesis/algorithms/quantum_differential_equations_solvers/lchs/overview/main.py",
    ) in validated
    assert (
        "algorithms/hamiltonian_simulation/hamiltonian_simulation_with_block_encoding/hamiltonian_simulation_gqsp",
        "examples/synthesis/algorithms/hamiltonian_simulation/hamiltonian_simulation_with_block_encoding/hamiltonian_simulation_gqsp/overview/main.py",
    ) in validated
    assert (
        "algorithms/amplitude_amplification_and_estimation/qsvt_fixed_point_amplitude_amplification/qsvt_fixed_point_amplitude_amplification",
        "examples/synthesis/algorithms/amplitude_amplification/qsvt_fixed_point/overview/main.py",
    ) in validated
    assert (
        "algorithms/quantum_state_preparation/adapt_vqe/adapt_vqe",
        "examples/synthesis/algorithms/quantum_state_preparation/adapt_vqe/overview/main.py",
    ) in validated
    assert (
        "applications/physical_systems/hhl_lanchester/hhl_lanchester",
        "examples/synthesis/applications/physical_systems/hhl_lanchester/overview/main.py",
    ) in validated
    assert (
        "community/paper_implementation_project/quantum_algo_for_solving_linear_differential_equations/harmonic_oscillator",
        "examples/synthesis/community/paper_implementation_project/quantum_algo_for_solving_linear_differential_equations/harmonic_oscillator/overview/main.py",
    ) in validated
    assert (
        "applications/chemistry/quantum_drude_oscillator/quantum_drude_oscillator",
        "examples/synthesis/applications/chemistry/quantum_drude_oscillator/overview/main.py",
    ) in validated
    assert (
        "algorithms/search_and_optimization/quantum_likelihood_estimation/quantum_likelihood_estimation",
        "examples/synthesis/algorithms/search_and_optimization/quantum_likelihood_estimation/overview/main.py",
    ) in validated
    assert (
        "algorithms/search_and_optimization/dqi/dqi_max_xorsat",
        "examples/synthesis/algorithms/search_and_optimization/dqi/dqi_max_xorsat/overview/main.py",
    ) in validated
    assert (
        "algorithms/quantum_state_preparation/gibbs/quantum_thermal_state_preparation",
        "examples/synthesis/algorithms/quantum_state_preparation/gibbs/quantum_thermal_state_preparation/overview/main.py",
    ) in validated
    assert (
        "community/paper_implementation_project/block_encoding/select_structures_BE",
        "examples/synthesis/community/paper_implementation_project/block_encoding/select_structures_be/overview/main.py",
    ) in validated
    assert (
        "applications/logistics/traveling_salesman_problem/traveling_salesman_problem",
        "examples/synthesis/applications/logistics/traveling_salesman_problem/overview/main.py",
    ) in validated
    assert (
        "applications/optimization/rectangles_packing/rectangles_packing_grid",
        "examples/synthesis/applications/optimization/rectangles_packing/rectangles_packing_grid/overview/main.py",
    ) in validated
    assert (
        "applications/CFD/qlbm/qlbm",
        "examples/synthesis/applications/cfd/qlbm/overview/main.py",
    ) in validated
    assert (
        "algorithms/hamiltonian_simulation/hamiltonian_simulation_with_block_encoding/hamiltonian_simulation_qubitization",
        "examples/synthesis/algorithms/hamiltonian_simulation/hamiltonian_simulation_with_block_encoding/hamiltonian_simulation_qubitization/overview/main.py",
    ) in validated
    assert (
        "applications/optimization/variational_quantum_imaginary_time_evolution/variational_quantum_imaginary_time_evolution",
        "examples/synthesis/applications/optimization/variational_quantum_imaginary_time_evolution/overview/main.py",
    ) in validated
    assert (
        "community/paper_implementation_project/quantum_compression_algorithm_for_symmetric_states/quantum_compression_algorithm_for_symmetric_states",
        "examples/synthesis/community/paper_implementation_project/quantum_compression_algorithm_for_symmetric_states/overview/main.py",
    ) in validated
    assert (
        "algorithms/quantum_linear_solvers/adiabatic_linear_solvers/solving_qlsp_with_aqc",
        "examples/synthesis/algorithms/quantum_linear_solvers/adiabatic_linear_solvers/solving_qlsp_with_aqc/overview/main.py",
    ) in validated
    assert (
        "algorithms/quantum_linear_solvers/vqls/vqls_with_lcu",
        "examples/synthesis/algorithms/quantum_linear_solvers/vqls/vqls_with_lcu/overview/main.py",
    ) in validated
    assert (
        "applications/CFD/QLS_for_hybrid_solvers/verify_block_encoding",
        "examples/synthesis/applications/cfd/qls_for_hybrid_solvers/verify_block_encoding/overview/main.py",
    ) in validated
    assert (
        "algorithms/quantum_state_preparation/fermionic_gaussian/fermionic_gaussian",
        "examples/synthesis/algorithms/quantum_state_preparation/fermionic_gaussian/overview/main.py",
    ) in validated
    assert (
        "applications/optimization/adapt_qaoa/adapt_qaoa",
        "examples/synthesis/applications/optimization/adapt_qaoa/overview/main.py",
    ) in validated
    assert (
        "algorithms/quantum_walks/glued_trees/glued_trees",
        "examples/synthesis/algorithms/quantum_walks/glued_trees/overview/main.py",
    ) in validated
    assert (
        "algorithms/search_and_optimization/grover_mixer_qaoa/gm_qaoa",
        "examples/synthesis/algorithms/search_and_optimization/grover_mixer_qaoa/gm_qaoa/overview/main.py",
    ) in validated
    assert (
        "applications/physical_systems/fermi_hubbard_model_1D/fermi_hubbard_1D",
        "examples/synthesis/applications/physical_systems/fermi_hubbard_model_1d/fermi_hubbard_1d/overview/main.py",
    ) in validated
    assert (
        "applications/optimization/set_cover/set_cover",
        "examples/synthesis/applications/optimization/set_cover/overview/main.py",
    ) in validated
    assert (
        "applications/optimization/minimum_dominating_set/minimum_dominating_set",
        "examples/synthesis/applications/optimization/minimum_dominating_set/overview/main.py",
    ) in validated
    assert (
        "applications/optimization/qaoa_in_qaoa/qaoa_in_qaoa",
        "examples/synthesis/applications/optimization/qaoa_in_qaoa/overview/main.py",
    ) in validated
    assert (
        "applications/optimization/robust_posture_optimization/robust_posture_optimization",
        "examples/synthesis/applications/optimization/robust_posture_optimization/overview/main.py",
    ) in validated
    assert (
        "applications/logistics/facility_location/facility_location",
        "examples/synthesis/applications/logistics/facility_location/overview/main.py",
    ) in validated
    assert (
        "applications/telecom/radio_access_network/radio_access_network_positioning_antennas",
        "examples/synthesis/applications/telecom/radio_access_network/radio_access_network_positioning_antennas/overview/main.py",
    ) in validated
    assert (
        "applications/optimization/electric_grid_optimization/electric_grid_optimization",
        "examples/synthesis/applications/optimization/electric_grid_optimization/overview/main.py",
    ) in validated
    assert (
        "applications/optimization/kidney_exchange/kidney_exchange_problem",
        "examples/synthesis/applications/optimization/kidney_exchange/kidney_exchange_problem/overview/main.py",
    ) in validated
    assert (
        "community/paper_implementation_project/Block_encoding-ND_Laplacian/ND_Laplacian_BE",
        "examples/synthesis/community/paper_implementation_project/block_encoding_nd_laplacian/overview/main.py",
    ) in validated
    assert (
        "applications/finance/autocallable_options/partial_exponential_state_preparation",
        "examples/synthesis/applications/finance/autocallable_options/partial_exponential_state_preparation/overview/main.py",
    ) in validated
    assert (
        "applications/finance/value_at_risk/value_at_risk",
        "examples/synthesis/applications/finance/value_at_risk/overview/main.py",
    ) in validated
    assert (
        "applications/finance/option_pricing/option_pricing",
        "examples/synthesis/applications/finance/option_pricing/overview/main.py",
    ) in validated
    assert (
        "tutorials/workshops/finance_workshops/Option_Pricing_Workshop",
        "examples/synthesis/tutorials/workshops/finance_workshops/option_pricing_workshop/overview/main.py",
    ) in validated
    assert (
        "applications/benchmarking/randomized_benchmarking/randomized_benchmarking",
        "examples/synthesis/applications/benchmarking/randomized_benchmarking/overview/main.py",
    ) in validated
    assert (
        "tutorials/advanced_tutorials/high_level_modeling_flexible_qpe/high_level_modeling_flexible_qpe",
        "examples/synthesis/tutorials/advanced_tutorials/high_level_modeling_flexible_qpe/overview/main.py",
    ) in validated
    assert (
        "applications/benchmarking/quantum_volume/quantum_volume",
        "examples/synthesis/applications/benchmarking/quantum_volume/overview/main.py",
    ) in validated
    assert (
        "tutorials/technology_demonstrations/QPE/qpe_for_grover_operator/qpe_for_grover_operator",
        "examples/synthesis/tutorials/technology_demonstrations/QPE/qpe_for_grover_operator/overview/main.py",
    ) in validated
    assert (
        "tutorials/technology_demonstrations/hhl/hhl_example",
        "examples/synthesis/tutorials/technology_demonstrations/hhl/hhl_example/overview/main.py",
    ) in validated
    assert (
        "algorithms/quantum_differential_equations_solvers/discrete_poisson_solver/discrete_poisson_solver",
        "examples/synthesis/algorithms/quantum_differential_equations_solvers/discrete_poisson_solver/overview/main.py",
    ) in validated
    assert (
        "algorithms/quantum_linear_solvers/hhl/hhl",
        "examples/synthesis/algorithms/quantum_linear_solvers/hhl/overview/main.py",
    ) in validated
    assert (
        "tutorials/technology_demonstrations/QPE/qpe_for_unitary_matrix/qpe_for_unitary_matrix",
        "examples/synthesis/tutorials/technology_demonstrations/QPE/qpe_for_unitary_matrix/overview/main.py",
    ) in validated
    assert (
        "algorithms/quantum_primitives/gradient_estimation/gradient_estimation",
        "examples/synthesis/algorithms/quantum_primitives/gradient_estimation/overview/main.py",
    ) in validated
    assert (
        "tutorials/workshops/hhl_workshop/hhl_workshop",
        "examples/synthesis/tutorials/workshops/hhl_workshop/overview/main.py",
    ) in validated
    assert (
        "tutorials/workshops/oracle_workshop/oracles_workshop",
        "examples/synthesis/tutorials/workshops/oracle_workshop/oracles_workshop/overview/main.py",
    ) in validated
    assert (
        "tutorials/workshops/algo_design_QCE_tutorial/algo_design_QCE_tutorial_part_I",
        "examples/synthesis/tutorials/workshops/algo_design_qce_tutorial/algo_design_qce_tutorial_part_i/overview/main.py",
    ) in validated
    assert (
        "tutorials/workshops/algo_design_QCE_tutorial/algo_design_QCE_tutorial_part_II",
        "examples/synthesis/tutorials/workshops/algo_design_qce_tutorial/algo_design_qce_tutorial_part_ii/overview/main.py",
    ) in validated
    assert (
        "tutorials/advanced_tutorials/discrete_quantum_walk_2d/discrete_time_quantum_walk",
        "examples/synthesis/tutorials/advanced_tutorials/discrete_quantum_walk_2d/discrete_time_quantum_walk/overview/main.py",
    ) in validated
    assert (
        "tutorials/basic_tutorials/the_classiq_tutorial/Qmod_tutorial_part1",
        "examples/synthesis/tutorials/basics/quantum_programming/qmod_tutorial_part1/overview/main.py",
    ) in validated
    assert (
        "applications/physical_systems/quantum_chaos/quantum_sawtooth_map",
        "examples/synthesis/applications/physical_systems/quantum_chaos/quantum_sawtooth_map/overview/main.py",
    ) in validated
    assert (
        "tutorials/workshops/combinatorial_workshop/combinatorial_qmod_workshop_for_maxcut",
        "examples/synthesis/tutorials/workshops/combinatorial_workshop/combinatorial_qmod_workshop_for_maxcut/overview/main.py",
    ) in validated
    assert (
        "applications/cybersecurity/link_monitoring/link_monitoring",
        "examples/synthesis/applications/cybersecurity/link_monitoring/overview/main.py",
    ) in validated
    assert (
        "tutorials/workshops/finance_workshops/combi_workshop_equality_constriants_PO",
        "examples/synthesis/tutorials/workshops/finance_workshops/combi_workshop_equality_constriants_po/overview/main.py",
    ) in validated
    assert (
        "tutorials/workshops/finance_workshops/combi_workshop_Inequality_constriants_PO",
        "examples/synthesis/tutorials/workshops/finance_workshops/combi_workshop_inequality_constriants_po/overview/main.py",
    ) in validated
    assert (
        "applications/cybersecurity/patching_management/patch_min_vertex_cover",
        "examples/synthesis/applications/cybersecurity/patching_management/patch_min_vertex_cover/overview/main.py",
    ) in validated
    assert (
        "applications/finance/portfolio_optimization/portfolio_optimization",
        "examples/synthesis/applications/finance/portfolio_optimization/overview/main.py",
    ) in validated
    assert (
        "applications/cybersecurity/whitebox_fuzzing/whitebox_fuzzing",
        "examples/synthesis/applications/cybersecurity/whitebox_fuzzing/overview/main.py",
    ) in validated
    assert (
        "tutorials/technology_demonstrations/hardware_aware_mcx/hardware_aware_mcx",
        "examples/synthesis/tutorials/technology_demonstrations/hardware_aware_mcx/overview/main.py",
    ) in validated
    assert (
        "tutorials/basic_tutorials/qmci_pi_estimation/qmci_pi_estimation",
        "examples/synthesis/tutorials/basics/qmci_pi_estimation/overview/main.py",
    ) in validated
    assert (
        "algorithms/number_theory_and_cryptography/shor/shor",
        "examples/synthesis/algorithms/number_theory_and_cryptography/shor/overview/main.py",
    ) in validated
    assert (
        "algorithms/number_theory_and_cryptography/discrete_log/discrete_log",
        "examples/synthesis/algorithms/number_theory_and_cryptography/discrete_log/overview/main.py",
    ) in validated
    assert (
        "algorithms/number_theory_and_cryptography/hidden_shift/hidden_shift",
        "examples/synthesis/algorithms/number_theory_and_cryptography/hidden_shift/overview/main.py",
    ) in validated
    assert (
        "algorithms/quantum_phase_estimation/qpe_for_matrix/qpe_for_matrix",
        "examples/synthesis/algorithms/quantum_phase_estimation/qpe_for_matrix/overview/main.py",
    ) in validated
    assert (
        "algorithms/search_and_optimization/grover/grover",
        "examples/synthesis/algorithms/search_and_optimization/grover/grover/overview/main.py",
    ) in validated
    assert (
        "algorithms/metrology/classical_shadow_tomography",
        "examples/synthesis/algorithms/metrology/classical_shadow_tomography/overview/main.py",
    ) in validated
    assert (
        "algorithms/QML/quantum_autoencoder/quantum_autoencoder",
        "examples/synthesis/algorithms/machine_learning/quantum_autoencoder/overview/main.py",
    ) in validated
    assert (
        "algorithms/QML/hybrid_qnn/hybrid_qnn_for_subset_majority",
        "examples/synthesis/algorithms/machine_learning/hybrid_qnn/hybrid_qnn_for_subset_majority/overview/main.py",
    ) in validated
    assert (
        "tutorials/basic_tutorials/qml_with_classiq_guide/qml_with_classiq_guide",
        "examples/synthesis/tutorials/basics/qml/overview/main.py",
    ) in validated
    assert (
        "algorithms/foundational/quantum_teleportation/quantum_teleportation",
        "examples/synthesis/algorithms/foundational/quantum_teleportation/overview/main.py",
    ) in validated
    assert (
        "community/paper_implementation_project/bb84/bb84_implementation",
        "examples/synthesis/community/paper_implementation_project/bb84/bb84_implementation/overview/main.py",
    ) in validated
    assert (
        "applications/optimization/min_graph_coloring/min_graph_coloring",
        "examples/synthesis/applications/optimization/min_graph_coloring/overview/main.py",
    ) in validated
    assert (
        "applications/optimization/max_induced_k_color_subgraph/max_induced_k_color_subgraph",
        "examples/synthesis/applications/optimization/max_induced_k_color_subgraph/overview/main.py",
    ) in validated
    assert (
        "applications/optimization/integer_linear_programming/integer_linear_programming",
        "examples/synthesis/applications/optimization/integer_linear_programming/overview/main.py",
    ) in validated
    assert (
        "applications/optimization/max_k_vertex_cover/max_k_vertex_cover",
        "examples/synthesis/applications/optimization/max_k_vertex_cover/overview/main.py",
    ) in validated
    assert (
        "applications/optimization/set_partition/set_partition",
        "examples/synthesis/applications/optimization/set_partition/overview/main.py",
    ) in validated
    assert (
        "tutorials/technology_demonstrations/classiq_paper/qsvt/classiq_qsvt",
        "examples/synthesis/tutorials/technology_demonstrations/classiq_paper/qsvt/classiq_qsvt/overview/main.py",
    ) in validated
    assert (
        "tutorials/technology_demonstrations/classiq_paper/quantum_walk/qiskit_discrete_quantum_walk",
        "examples/synthesis/tutorials/technology_demonstrations/classiq_paper/quantum_walk/qiskit_discrete_quantum_walk/overview/main.py",
    ) in validated
    assert (
        "tutorials/technology_demonstrations/classiq_paper/quantum_walk/tket_discrete_quantum_walk",
        "examples/synthesis/tutorials/technology_demonstrations/classiq_paper/quantum_walk/tket_discrete_quantum_walk/overview/main.py",
    ) in validated
    assert (
        "tutorials/technology_demonstrations/classiq_paper/quantum_walk/pennylane_catalyst_discrete_quantum_walk",
        "examples/synthesis/tutorials/technology_demonstrations/classiq_paper/quantum_walk/pennylane_catalyst_discrete_quantum_walk/overview/main.py",
    ) in validated
    assert (
        "tutorials/technology_demonstrations/classiq_paper/quantum_walk/classiq_discrete_quantum_walk",
        "examples/synthesis/tutorials/technology_demonstrations/classiq_paper/quantum_walk/classiq_discrete_quantum_walk/overview/main.py",
    ) in validated
    assert (
        "applications/chemistry/second_quantized_hamiltonian/second_quantized_hamiltonian",
        "examples/synthesis/applications/chemistry/second_quantized_hamiltonian/overview/main.py",
    ) in validated
    assert (
        "applications/physical_systems/ising_model/ising_model",
        "examples/synthesis/applications/physical_systems/ising_model/overview/main.py",
    ) in validated
    assert (
        "algorithms/amplitude_amplification_and_estimation/qmc_user_defined/qmc_user_defined",
        "examples/synthesis/algorithms/amplitude_amplification_and_estimation/qmc_user_defined/overview/main.py",
    ) in validated
    assert (
        "algorithms/amplitude_amplification_and_estimation/quantum_counting/quantum_counting",
        "examples/synthesis/algorithms/amplitude_amplification_and_estimation/quantum_counting/overview/main.py",
    ) in validated
    assert (
        "algorithms/quantum_primitives/gqsp/gqsp",
        "examples/synthesis/algorithms/quantum_primitives/gqsp/overview/main.py",
    ) in validated
    assert (
        "algorithms/amplitude_amplification_and_estimation/oblivious_amplitude_amplification/oblivious_amplitude_amplification",
        "examples/synthesis/algorithms/amplitude_amplification_and_estimation/oblivious_amplitude_amplification/overview/main.py",
    ) in validated
    assert (
        "functions/qmod_library_reference/classiq_open_library/qsvt/qsvt",
        "examples/synthesis/functions/qsvt/overview/main.py",
    ) in validated
    assert (
        "functions/qmod_library_reference/classiq_open_library/qct_qst/qct_qst",
        "examples/synthesis/functions/qct_qst/overview/main.py",
    ) in validated
    assert (
        "tutorials/basic_tutorials/quantumwalk_complex_network/quantumwalk_complex_network",
        "examples/synthesis/tutorials/basics/quantumwalk_complex_network/overview/main.py",
    ) in validated
    for operation in ("or", "xor"):
        assert (
            f"functions/function_usage_examples/arithmetic/bitwise_{operation}/bitwise_{operation}_example",
            f"examples/synthesis/functions/arithmetic/bitwise_{operation}/overview/main.py",
        ) in validated
    for operation in ("comparator", "negation"):
        assert (
            f"functions/function_usage_examples/arithmetic/{operation}/{operation}_example",
            f"examples/synthesis/functions/arithmetic/{operation}/overview/main.py",
        ) in validated
    for operation, source_name in (
        ("multiplication", "multiplication"),
        ("subtraction", "subtraction_example"),
        ("extremum", "extremum_example"),
    ):
        assert (
            f"functions/function_usage_examples/arithmetic/{operation}/{source_name}",
            f"examples/synthesis/functions/arithmetic/{operation}/overview/main.py",
        ) in validated
    for group, path in (
        (
            "functions/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_exponential_state",
            "examples/synthesis/functions/state_preparation/prepare_exponential_state/overview/main.py",
        ),
        (
            "functions/function_usage_examples/mcx/mcx_example",
            "examples/synthesis/functions/mcx/mcx_example/overview/main.py",
        ),
        (
            "functions/qmod_library_reference/qmod_core_library/hamiltonian_evolution/qdrift/qdrift",
            "examples/synthesis/functions/hamiltonian_evolution/qdrift/overview/main.py",
        ),
        (
            "functions/qmod_library_reference/classiq_open_library/linear_pauli_rotations/linear_pauli_rotations",
            "examples/synthesis/functions/linear_pauli_rotations/overview/main.py",
        ),
    ):
        assert (group, path) in validated
    for group, path in (
        (
            "functions/qmod_library_reference/classiq_open_library/grover_operator/grover_operator",
            "examples/synthesis/functions/grover_operator/overview/main.py",
        ),
        (
            "tutorials/technology_demonstrations/arithmetic_expressions/arithmetic_expressions",
            "examples/synthesis/tutorials/technology_demonstrations/arithmetic_expressions/overview/main.py",
        ),
        (
            "functions/qmod_library_reference/classiq_open_library/variational_data_encoding/variational_data_encoding",
            "examples/synthesis/functions/data_encoding/variational_data_encoding/overview/main.py",
        ),
        (
            "functions/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_partial_uniform_state",
            "examples/synthesis/functions/state_preparation/prepare_partial_uniform_state/overview/main.py",
        ),
    ):
        assert (group, path) in validated
    for group, path in (
        (
            "algorithms/quantum_primitives/hadamard_test/hadamard_test",
            "examples/synthesis/algorithms/quantum_primitives/hadamard_test/overview/main.py",
        ),
        (
            "tutorials/basic_tutorials/quantum_primitives/hadamard_test_tutorial/hadamard_test_tutorial",
            "examples/synthesis/tutorials/basics/quantum_primitives/hadamard_test_tutorial/overview/main.py",
        ),
        (
            "algorithms/quantum_primitives/swap_test/swap_test",
            "examples/synthesis/algorithms/quantum_primitives/swap_test/overview/main.py",
        ),
        (
            "functions/qmod_library_reference/classiq_open_library/qpe/qpe",
            "examples/synthesis/functions/qpe/overview/main.py",
        ),
    ):
        assert (group, path) in validated
    for group, path in (
        (
            "tutorials/basic_tutorials/quantum_primitives/linear_combination_of_unitaries/linear_combination_of_unitaries",
            "examples/synthesis/tutorials/basics/quantum_primitives/linear_combination_of_unitaries/overview/main.py",
        ),
        (
            "algorithms/hamiltonian_simulation/hamiltonian_simulation_guide/hamiltonian_simulation_guide",
            "examples/synthesis/algorithms/hamiltonian_simulation/hamiltonian_simulation_guide/overview/main.py",
        ),
    ):
        assert (group, path) in validated
    for group, path in (
        (
            "algorithms/foundational/simon/simon",
            "examples/synthesis/algorithms/foundational/simon/overview/main.py",
        ),
        (
            "tutorials/basic_tutorials/grover_graph_coloring/grover_graph_coloring",
            "examples/synthesis/tutorials/basics/grover_graph_coloring/overview/main.py",
        ),
    ):
        assert (group, path) in validated
    for group, path in (
        (
            "community/basic_examples/vqe/vqe",
            "examples/synthesis/community/basics/vqe/overview/main.py",
        ),
        (
            "tutorials/workshops/grover_workshop/grover_workshop",
            "examples/synthesis/tutorials/workshops/grover_workshop/grover_workshop/overview/main.py",
        ),
        (
            "algorithms/hamiltonian_simulation/hamiltonian_simulation_with_block_encoding/jacobi_anger_expansion",
            "examples/synthesis/algorithms/hamiltonian_simulation/hamiltonian_simulation_with_block_encoding/jacobi_anger_expansion/overview/main.py",
        ),
    ):
        assert (group, path) in validated
    rendered = render(catalog)
    assert "**212/213**" in rendered[ROOT / "examples/synthesis/CATALOG.md"]
    for case in catalog["cases"]:
        if case["native_example"]:
            text = rendered[example_path(case["directory"]) / "README.md"]
            assert ("complete local Notebook workflow are validated" in text) == (
                (case["id"][:-5], case["native_example"]) in validated
            ), case["id"]
