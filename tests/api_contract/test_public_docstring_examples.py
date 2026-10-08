"""Public workflow examples remain executable and free of external effects."""

import ast
import doctest
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum.algorithms.arithmetic import (
    adder_circuit,
    adder_wires,
)
from flagquantum.algorithms.chemistry import (
    coupler_hardware_efficient_ansatz,
    double_excitation,
    excitation_operator,
    single_excitation,
    uccsd_ansatz,
    uccsd_excitations,
    uccsd_factors,
)
from flagquantum.algorithms.chemistry_integrals import (
    MolecularGeometry,
    MolecularIntegrals,
    molecular_integrals,
)
from flagquantum.algorithms.cobyla import CobylaOptimizer
from flagquantum.algorithms.data_encoding import (
    amplitude_encode,
    angular_encode,
)
from flagquantum.algorithms.folding import fold_program
from flagquantum.algorithms.logical_resources import (
    estimate_logical_resources,
    surface_code_qubits_per_logical,
)
from flagquantum.algorithms.molecular import (
    HartreeFockSolution,
    MolecularHamiltonian,
    create_molecular_hamiltonian,
)
from flagquantum.algorithms.nelder_mead import NelderMeadOptimizer
from flagquantum.algorithms.spsa import SPSAOptimizer
from flagquantum.algorithms.trotter import (
    pauli_exponential_circuit,
    trotter_circuit,
)
from flagquantum.algorithms.variational import (
    maxcut_hamiltonian,
    run_qaoa,
)
from flagquantum.compiler import Layout
from flagquantum.compiler.openqasm_import import (
    import_openqasm,
    import_openqasm_to_ir,
)
from flagquantum.compiler.resource_estimation import estimate_resources
from flagquantum.compiler.translate import translate
from flagquantum.ecosystem.cirq import run as run_cirq
from flagquantum.ecosystem.extensions.target_sdk import target_capability_snapshot
from flagquantum.ecosystem.pennylane import run as run_pennylane
from flagquantum.ecosystem.qiskit import run as run_qiskit
from flagquantum.ecosystem.simulators import recommend as recommend_simulator
from flagquantum.lindblad import plan as plan_lindblad_evolution
from flagquantum.observables.boson import BosonOperator
from flagquantum.observables.boson import position as boson_position
from flagquantum.observables.fermion import (
    FermionOperator,
    jordan_wigner,
    parity_encoding,
)
from flagquantum.operators import SuperOperator
from flagquantum.qec import (
    BeliefPropagationOsdDecoder,
    CssCode,
    CssCodeMatrices,
    FloquetCode,
    MeasurementPhase,
    SubsystemCode,
    bivariate_bicycle_code,
    qldpc_code,
    reed_muller_code,
    ring_floquet_code,
    tesseract_code,
    tesseract_column_swap,
    tesseract_free_cnot_pairs,
)
from flagquantum.runtime import planner
from flagquantum.runtime.executors.statevector import gather_distributed_statevector
from flagquantum.simulation.lindblad import evolve_density_matrix
from flagquantum.simulation.lindblad_adjoint import adjoint_gradient
from flagquantum.simulation.pauli import exponential_pauli_operator
from flagquantum.simulation.stabilizer import pauli_readout
from flagquantum.simulation.unitary import get_unitary

pytestmark = pytest.mark.unit

# Every entry whose docstrings carry examples. `fq.plan` and `planner.plan` are
# different functions that document different things, so both are listed. The
# Lindblad planner and the density-matrix integrator are two more distinct entry
# points; translation, unitary access, and static resource counting are three
# more. The fermionic operator and its Jordan-Wigner mapping are two functions
# in one module, and both of their docstrings carry an example, while the
# superoperator class documents its three constructors, its two views, and its
# accumulation protocol in one example. The bosonic operator class and the
# bosonic position operator are two more, and the position operator is listed
# under a module-local name because `fq.position` does not exist. The exponential
# of a Pauli product is the fourth entry from a module that is not a documented
# workflow namespace. The chemistry
# module contributes seven: the generator, the excitation census, the two
# circuit builders, and the UCCSD and hardware-efficient products, each of which
# documents a different workflow step. The Trotter module contributes two more: the
# exponential of one Pauli word, and the product formula built from a Hamiltonian's
# terms. The logical resource module contributes two: the reporting entry point,
# whose example is the whole footprint of one program at one distance, and the patch
# size helper, whose example is that figure at three distances without a program.
# The data-encoding module contributes two: the amplitude encoder, whose example is
# the prepared two-amplitude state, and the angle encoder, whose example is the
# product state of two zero rotations. The arithmetic module contributes two: the
# adder's constructor, whose example is that circuit's wire count and its gate
# census at three bits, and the register map, whose example is the four wire
# groups that census is read through. The Nelder-Mead optimizer contributes
# one: its example is the same two-qubit Pauli energy the SPSA entry beside it
# minimizes, so the two optimizer units are compared on one objective rather
# than each being merely present. The trust-region optimizer contributes one: its
# example minimizes that same Pauli energy under two constraints, and the two
# numbers it asserts are the energy the constraints allow and the boundary point
# they pin the run to, so the constrained unit is distinguished from the
# unconstrained one it sits beside rather than merely present next to it. The
# folding module contributes one: its
# example is the shortest statement that separates the two quantities a plan
# reports, so a reader sees the realized instruction count beside the factor
# that was asked for rather than a single number standing for both. The adjoint
# route into the Lindblad engine is
# listed beside the forward integrator because its example is the same kind of
# statement about a different capability: the cotangent's shape is the shape of
# the state the trajectory started from, which is what distinguishes a reverse
# pass over the trajectory from a derivative of a returned trajectory. The
# integral module contributes three: the geometry, whose example is the basis it
# expands into, the integral set, whose example is the one- and two-electron
# values a Hartree-Fock solve consumes, and the entry point that builds one. The
# driver module contributes three more: the Hartree-Fock solution, the returned
# Hamiltonian with its energies, and the geometry-to-Hamiltonian entry point. The
# error-correction package contributes thirteen: the code record a caller
# constructs,
# the matrices that record is read back as, the bivariate-bicycle family, whose
# example is the smallest member of the family rather than the published one,
# because the published instance spends its time in the distance search and an
# example is not evidence of a distance, the punctured Reed-Muller family, whose
# example is its smallest member and whose two numbers are the two family
# distances rather than the code's distance alone, the subsystem record, whose
# example is the two-by-two Bacon-Shor code because that is the smallest member of
# a family whose matrices are not enough to state it, the published tesseract,
# whose example is the five numbers that separate its reading of sixteen data
# qubits from the Calderbank-Shor-Steane reading of the same check rows, the
# permutation of that family's free-CNOT gadget, whose example is the whole
# sixteen-wire tuple because the tuple is the fact rather than a summary of it,
# the protected pairs that gadget couples, whose two pairs are the declared action
# the permutation is checked against, and the
# route that derives the
# same record from a caller's check matrices, whose example is the Steane code
# written as its three checks rather than as a record, and the hyperedge decoder's
# model constructor, whose example is the name the registry reaches it by, since
# that name is what the constructor exists for. The dynamic-code record contributes
# three more: the record itself, whose example states a schedule of four phases by
# hand and reads a period of two back off it, the measurement phase the schedule is
# built from, whose example is a named round and the two numbers a round reports
# about its own operators, and the published two-phase ring, whose example is the
# six numbers that separate a schedule which protects qubits from one that does not.
ENTRIES = (
    adder_circuit,
    adder_wires,
    adjoint_gradient,
    amplitude_encode,
    angular_encode,
    BeliefPropagationOsdDecoder.from_detector_error_model,
    BosonOperator,
    CobylaOptimizer,
    CssCode,
    CssCodeMatrices,
    FermionOperator,
    FloquetCode,
    HartreeFockSolution,
    Layout,
    MeasurementPhase,
    MolecularGeometry,
    MolecularHamiltonian,
    MolecularIntegrals,
    NelderMeadOptimizer,
    SuperOperator,
    SPSAOptimizer,
    SubsystemCode,
    boson_position,
    bivariate_bicycle_code,
    coupler_hardware_efficient_ansatz,
    create_molecular_hamiltonian,
    exponential_pauli_operator,
    double_excitation,
    estimate_logical_resources,
    estimate_resources,
    evolve_density_matrix,
    excitation_operator,
    fold_program,
    fq.Circuit,
    fq.Module,
    fq.Observable,
    fq.compile,
    fq.expectation,
    fq.from_openqasm,
    fq.gradient,
    fq.plan,
    fq.run,
    fq.train,
    gather_distributed_statevector,
    get_unitary,
    import_openqasm,
    import_openqasm_to_ir,
    jordan_wigner,
    maxcut_hamiltonian,
    molecular_integrals,
    pauli_exponential_circuit,
    pauli_readout,
    parity_encoding,
    plan_lindblad_evolution,
    planner.plan,
    qldpc_code,
    reed_muller_code,
    recommend_simulator,
    ring_floquet_code,
    run_cirq,
    run_pennylane,
    run_qaoa,
    run_qiskit,
    single_excitation,
    surface_code_qubits_per_logical,
    tesseract_code,
    tesseract_column_swap,
    tesseract_free_cnot_pairs,
    target_capability_snapshot,
    translate,
    trotter_circuit,
    uccsd_ansatz,
    uccsd_excitations,
    uccsd_factors,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _modules_with_examples() -> set[str]:
    """Return the importable module names whose docstrings carry an example."""
    modules = set()
    for path in sorted((_REPOSITORY_ROOT / "flagquantum").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(
                node,
                (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                continue
            docstring = ast.get_docstring(node)
            if docstring and ">>>" in docstring:
                relative = path.relative_to(_REPOSITORY_ROOT).with_suffix("")
                parts = list(relative.parts)
                if parts[-1] == "__init__":
                    parts.pop()
                modules.add(".".join(parts))
                break
    return modules


def test_primary_public_docstring_examples() -> None:
    runner = doctest.DocTestRunner()
    finder = doctest.DocTestFinder()

    for entry in ENTRIES:
        name = f"{entry.__module__}.{entry.__name__}"
        for example in finder.find(entry, name=name):
            runner.run(example)

    failures, _ = runner.summarize()
    assert failures == 0


def test_every_module_with_examples_is_covered() -> None:
    """Fail when an example appears in a module this gate does not run.

    The list of entries above is what makes an example executable, so a
    docstring written in a module outside that list would look tested while
    nothing ran it. Adding the module's public entry to ``ENTRIES`` is the fix.
    """
    covered = {entry.__module__ for entry in ENTRIES}
    uncovered = sorted(_modules_with_examples() - covered)

    assert uncovered == [], (
        "docstring examples in modules that no doctest entry runs: " f"{uncovered}"
    )
