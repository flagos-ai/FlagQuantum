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
from flagquantum.algorithms.data_encoding import (
    amplitude_encode,
    angular_encode,
)
from flagquantum.algorithms.logical_resources import (
    estimate_logical_resources,
    surface_code_qubits_per_logical,
)
from flagquantum.algorithms.nelder_mead import NelderMeadOptimizer
from flagquantum.algorithms.spsa import SPSAOptimizer
from flagquantum.algorithms.trotter import (
    pauli_exponential_circuit,
    trotter_circuit,
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
from flagquantum.qec import CssCodeMatrices
from flagquantum.runtime import planner
from flagquantum.runtime.executors.statevector import gather_distributed_statevector
from flagquantum.simulation.lindblad import evolve_density_matrix
from flagquantum.simulation.lindblad_adjoint import adjoint_gradient
from flagquantum.simulation.pauli import exponential_pauli_operator
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
# than each being merely present. The adjoint route into the Lindblad engine is
# listed beside the forward integrator because its example is the same kind of
# statement about a different capability: the cotangent's shape is the shape of
# the state the trajectory started from, which is what distinguishes a reverse
# pass over the trajectory from a derivative of a returned trajectory.
ENTRIES = (
    adder_circuit,
    adder_wires,
    adjoint_gradient,
    amplitude_encode,
    angular_encode,
    BosonOperator,
    CssCodeMatrices,
    FermionOperator,
    Layout,
    NelderMeadOptimizer,
    SuperOperator,
    SPSAOptimizer,
    boson_position,
    coupler_hardware_efficient_ansatz,
    exponential_pauli_operator,
    double_excitation,
    estimate_logical_resources,
    estimate_resources,
    evolve_density_matrix,
    excitation_operator,
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
    pauli_exponential_circuit,
    parity_encoding,
    plan_lindblad_evolution,
    planner.plan,
    recommend_simulator,
    run_cirq,
    run_pennylane,
    run_qiskit,
    single_excitation,
    surface_code_qubits_per_logical,
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
