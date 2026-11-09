"""Verify bounded native products, ownership barriers and the complete AQC source."""

import math
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_linear_solvers.adiabatic_linear_solvers.solving_qlsp_with_aqc.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.algorithms.quantum_linear_solvers.adiabatic_linear_solvers.solving_qlsp_with_aqc.main import (
    main,
)
from examples.synthesis.algorithms.quantum_linear_solvers.adiabatic_linear_solvers.solving_qlsp_with_aqc.physics import (
    H0,
    H1,
    TIMES,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate, _pauli_products
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_qmod_aqc_qlsp import aqc_reference, original
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_complete_aqc_matches_source_and_two_independent_circuit_references(
    objective: Literal["width", "depth"],
) -> None:
    source = original()
    native = synth.synthesize(
        main, classical_inputs={"times": TIMES}, objective=objective, max_workspace=0
    )
    retained = synth.synthesize(source, objective=objective, max_workspace=0)
    assert len(TIMES) == 200
    assert (len(H0.terms), len(H1.terms)) == (10, 16)
    assert native.report["resources"] == retained.report["resources"]
    assert native.report["workspace_qubits"] == 0
    resources = native.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_operations == 4542
    for reference in (reference_circuit(), aqc_reference(source.read_text())):
        compare_states(native, reference, ("qba",))


OPERATOR = "-0.2*fq.I()+0.4*(fq.X(0)@fq.Z(1))+0.3*fq.Y(0)"


def compact_entry(
    *, barrier: str = "", helper: bool = False, order: int = 2
) -> _QuantumFunction:
    prelude = ""
    count = 256 // order
    call = f"suzuki_trotter(OPERATOR,t,{order},{count},q)"
    if helper:
        prelude = (
            "@qfunc\ndef evolve(t:float,q:QArray[QBit]):\n"
            f"    for step in range(8):\n        {call}\n"
        )
        calls = "    evolve(t,q)\n"
    else:
        calls = "    for step in range(8):\n        " + call + "\n"
    return program_definition(
        "import flagquantum as fq\nfrom flagquantum.synth import *\n"
        f"OPERATOR={OPERATOR}\n"
        + prelude
        + "@qfunc\ndef main(t:float,s:Output[QBit],q:Output[QArray[QBit]]):\n"
        "    allocate(s)\n    H(s)\n    allocate(2,q)\n    H(q[0])\n    S(q[0])\n"
        + calls
        + barrier
    )


def dense_step(
    positions: dict[int, int],
    terms: list[tuple[tuple[tuple[int, str], ...], float]],
    step: float,
    order: int,
) -> list[list[complex]]:
    """Independent dense tensor products replace the compile-time mask algorithm."""
    dtype = torch.complex128
    axes = {
        "I": torch.eye(2, dtype=dtype),
        "X": torch.tensor([[0, 1], [1, 0]], dtype=dtype),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=dtype),
        "Z": torch.diag(torch.tensor([1, -1], dtype=dtype)),
    }
    eye = torch.eye(1 << len(positions), dtype=dtype)
    product = eye
    sequence = terms if order == 1 else terms + terms[::-1]
    for support, coefficient in sequence:
        factors = {positions[wire]: axis for wire, axis in support}
        pauli = torch.ones((1, 1), dtype=dtype)
        for bit in reversed(range(len(positions))):
            pauli = torch.kron(pauli, axes[factors.get(bit, "I")])
        angle = coefficient * step
        product = (math.cos(angle) * eye - 1j * math.sin(angle) * pauli) @ product
    return [[complex(value) for value in row] for row in product.tolist()]


@pytest.mark.parametrize("time", [0.17, -0.21])
@pytest.mark.parametrize("helper", [False, True])
@pytest.mark.parametrize("order", [1, 2])
def test_replaced_products_keep_coherent_offset_targets_phase_and_helper_barriers(
    monkeypatch: pytest.MonkeyPatch,
    time: float,
    helper: bool,
    order: int,
) -> None:
    entry = compact_entry(barrier="    H(q[1])\n    X(s)\n", helper=helper, order=order)
    before = synth.synthesize(entry, classical_inputs={"t": time}, max_workspace=0)
    calls = []

    def replacement(
        positions: dict[int, int],
        terms: list[tuple[tuple[tuple[int, str], ...], float]],
        step: float,
        order: int,
    ) -> list[list[complex]]:
        calls.append((len(terms), order))
        return dense_step(positions, terms, step, order)

    monkeypatch.setattr(_pauli_products, "pauli_step", replacement)
    after = synth.synthesize(entry, classical_inputs={"t": time}, max_workspace=0)
    assert calls == [(3, order)] * 8
    reference = fq.Circuit(3, dtype=torch.complex128).h(0).h(2).s(2)
    terms = [((), -0.2), (((0, "X"), (1, "Z")), 0.4), (((0, "Y"),), 0.3)]
    matrix = torch.tensor(
        dense_step({0: 0, 1: 1}, terms, time / 256, order), dtype=torch.complex128
    )
    reference.any(1, 2, matrix=torch.linalg.matrix_power(matrix, (256 // order) * 8)).h(
        1
    ).x(0)
    for program in (before, after):
        compare_states(program, reference, ("s", "q"))
        assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("barrier", ["    H(q[0])\n", "    X(q[1])\n"])
def test_intervening_effects_are_not_commuted_through_products(barrier: str) -> None:
    entry = program_definition(
        "import flagquantum as fq\nfrom flagquantum.synth import *\n"
        f"OPERATOR={OPERATOR}\n@qfunc\ndef main(q:Output[QArray[QBit]]):\n"
        "    allocate(2,q)\n"
        "    for i in range(8):\n        suzuki_trotter(OPERATOR,0.17,2,128,q)\n"
        + barrier
        + "    for j in range(8):\n        suzuki_trotter(OPERATOR,-0.21,2,128,q)\n"
    )
    program = synth.synthesize(entry)
    reference = fq.Circuit(2, dtype=torch.complex128)
    terms = [((), -0.2), (((0, "X"), (1, "Z")), 0.4), (((0, "Y"),), 0.3)]
    for index, time in enumerate((0.17, -0.21)):
        matrix = torch.tensor(
            dense_step({0: 0, 1: 1}, terms, time / 256, 2), dtype=torch.complex128
        )
        reference.any(0, 1, matrix=torch.linalg.matrix_power(matrix, 128 * 8))
        if index == 0:
            if "H(" in barrier:
                reference.h(1)
            else:
                reference.x(0)
    compare_states(program, reference, ("q",))


@pytest.mark.parametrize(
    "budget", ["max_operations", "max_depth", "max_qubits", "memory_limit_bytes"]
)
def test_compaction_preserves_physical_resource_budget_rejections(budget: str) -> None:
    with pytest.raises((synth.SynthesisError, PlanningError, HybridVerificationError)):
        synth.synthesize(
            main,
            classical_inputs={"times": TIMES},
            max_operations=20 if budget == "max_operations" else None,
            max_depth=20 if budget == "max_depth" else None,
            max_qubits=2 if budget == "max_qubits" else None,
            memory_limit_bytes=1 if budget == "memory_limit_bytes" else None,
        )


@pytest.mark.parametrize(
    "change,diagnostic",
    [
        ("q:Const[QArray[QBit]]", "borrow|Const|const"),
        ("partial", "complete register"),
        ("uninitialized", "initializ"),
    ],
)
def test_pending_products_do_not_bypass_storage_and_view_validation(
    change: str, diagnostic: str
) -> None:
    signature = (
        "q:Const[QNum[2]]" if change.startswith("q:") else "q:Output[QArray[QBit]]"
    )
    allocate = (
        ""
        if change in {"uninitialized", "q:Const[QArray[QBit]]"}
        else "    allocate(2,q)\n"
    )
    target = "q[1:]" if change == "partial" else "q"
    if change == "uninitialized":
        signature = "q:Output[QNum[2]]"
    entry = program_definition(
        "import flagquantum as fq\nfrom flagquantum.synth import *\nOPERATOR=fq.X(0)\n"
        f"@qfunc\ndef main({signature}):\n"
        + allocate
        + f"    for i in range(16):\n        suzuki_trotter(OPERATOR,0.3,2,256,{target})\n"
    )
    with pytest.raises(
        (synth.SynthesisError, HybridVerificationError), match=diagnostic
    ):
        synth.synthesize(entry)


def test_roundoff_envelope_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def invalid(matrix: list[list[complex]]) -> list[list[complex]]:
        raise ValueError("compact product exceeds the roundoff envelope")

    monkeypatch.setattr(_pauli_products, "roundoff_unitary", invalid)
    with pytest.raises(
        (synth.SynthesisError, HybridVerificationError), match="roundoff envelope"
    ):
        synth.synthesize(compact_entry(), classical_inputs={"t": 0.17})


def test_qmod_consumer_accepts_replaced_compiler_products_without_adapter_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.integration.test_synthesis_qmod_aqc_qlsp import SMALL

    before = synth.synthesize(SMALL, source_format="qmod")
    calls = []

    def replacement(
        positions: dict[int, int],
        terms: list[tuple[tuple[tuple[int, str], ...], float]],
        step: float,
        order: int,
    ) -> list[list[complex]]:
        calls.append(order)
        return dense_step(positions, terms, step, order)

    monkeypatch.setattr(_pauli_products, "pauli_step", replacement)
    after = synth.synthesize(SMALL, source_format="qmod")
    assert calls == [2, 1]
    torch.testing.assert_close(
        fq.run(before).statevector(), fq.run(after).statevector(), atol=1e-12, rtol=0
    )


def test_roundoff_correction_preserves_identity_phase_and_rejects_nonunitary_products() -> (
    None
):
    angle = 0.27
    matrix = [
        [complex(math.cos(angle), math.sin(angle)) * (1 + 1e-14), 0j],
        [0j, complex(math.cos(angle), math.sin(angle)) * (1 + 1e-14)],
    ]
    corrected = _pauli_products.roundoff_unitary(matrix)
    expected = torch.eye(2, dtype=torch.complex128) * complex(
        math.cos(angle), math.sin(angle)
    )
    torch.testing.assert_close(
        torch.tensor(corrected, dtype=torch.complex128), expected, atol=1e-14, rtol=0
    )
    with pytest.raises(ValueError, match="roundoff envelope"):
        _pauli_products.roundoff_unitary([[1.01 + 0j, 0j], [0j, 1 + 0j]])


def test_more_than_one_buffer_preserves_all_calls_and_single_qubit_global_phase() -> (
    None
):
    entry = program_definition(
        "import flagquantum as fq\nfrom flagquantum.synth import *\n"
        "OPERATOR=0.3*fq.Y(0)-0.2*fq.I()\n"
        "@qfunc\ndef main(q:Output[QBit]):\n    allocate(q)\n    H(q)\n"
        "    for i in range(256):\n        suzuki_trotter(OPERATOR,0.013,2,16,q)\n"
        "    suzuki_trotter(OPERATOR,-0.021,1,3,q)\n"
    )
    terms = [(((0, "Y"),), 0.3), ((), -0.2)]
    first = torch.tensor(
        dense_step({0: 0}, terms, 0.013 / 32, 2), dtype=torch.complex128
    )
    last = torch.tensor(
        dense_step({0: 0}, terms, -0.021 / 3, 1), dtype=torch.complex128
    )
    reference = fq.Circuit(1, dtype=torch.complex128).h(0)
    reference.any(0, matrix=torch.linalg.matrix_power(first, 256 * 16))
    reference.any(0, matrix=torch.linalg.matrix_power(last, 3))
    compare_states(synth.synthesize(entry), reference, ("q",))


def test_switching_targets_flushes_products_without_changing_register_bit_order() -> (
    None
):
    entry = program_definition(
        "import flagquantum as fq\nfrom flagquantum.synth import *\nOPERATOR=fq.Y(0)\n"
        "@qfunc\ndef main(a:Output[QBit],b:Output[QBit]):\n"
        "    allocate(a)\n    allocate(b)\n    H(a)\n    H(b)\n"
        "    for i in range(16):\n        suzuki_trotter(OPERATOR,0.013,2,256,a)\n"
        "    for j in range(16):\n        suzuki_trotter(OPERATOR,-0.021,2,256,b)\n"
    )
    reference = fq.Circuit(2, dtype=torch.complex128).h(0).h(1)
    reference.ry(0, theta=2 * 16 * 0.013).ry(1, theta=-2 * 16 * 0.021)
    compare_states(synth.synthesize(entry), reference, ("a", "b"))


def test_wider_target_and_one_oversized_call_keep_existing_effect_limit() -> None:
    for width, calls in ((4, 2), (3, 1)):
        entry = program_definition(
            "import flagquantum as fq\nfrom flagquantum.synth import *\n"
            "OPERATOR=" + "+".join(["fq.X(0)"] * 256) + "\n"
            "@qfunc\ndef main(q:Output[QArray[QBit]]):\n"
            f"    allocate({width},q)\n    for i in range({calls}):\n"
            "        suzuki_trotter(OPERATOR,0.01,1,32,q)\n"
        )
        with pytest.raises(
            (synth.SynthesisError, HybridVerificationError),
            match="4096 effect operations",
        ):
            synth.synthesize(entry)
