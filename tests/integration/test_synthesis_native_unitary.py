"""Static matrix capture shares existing unitary checks and explicit bit order."""

import ast
import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def definition(
    matrix: str, body: str, *, prefix: str = "", width: int = 2
) -> _QuantumFunction:
    return program_definition(
        "from flagquantum.synth import *\n" + prefix + "\n"
        "@qfunc\ndef main(sentinel: Output[QBit], q: Output[QArray[QBit]]):\n"
        "    allocate(sentinel)\n    X(sentinel)\n"
        f"    allocate({width},q)\n"
        + "\n".join("    " + line for line in body.splitlines())
        + f"\n    unitary({matrix},q)\n"
    )


def random_unitary(width: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(941 + width)
    matrix = torch.randn(
        1 << width, 1 << width, generator=generator, dtype=torch.complex128
    )
    orthogonal: torch.Tensor
    orthogonal, _ = torch.linalg.qr(matrix)
    return orthogonal * torch.exp(torch.tensor(0.37j, dtype=torch.complex128))


@pytest.mark.parametrize(
    ("width", "basis"), [(w, b) for w in (1, 2, 3) for b in range(1 << w)]
)
def test_all_small_unitary_columns_preserve_absolute_phase_and_offset_wires(
    width: int, basis: int
) -> None:
    matrix = random_unitary(width)
    preparation = "\n".join(
        f"X(q[{bit}])" for bit in range(width) if basis & (1 << bit)
    )
    entry = definition(repr(matrix.tolist()), preparation, width=width)
    reference = fq.Circuit(width + 1, dtype=torch.complex128).x(0)
    for bit in range(width):
        if basis & (1 << bit):
            reference.x(width - bit)
    reference.any(*range(1, width + 1), matrix=matrix)
    compare_states(synth.synthesize(entry), reference, ("sentinel", "q"))


@pytest.mark.parametrize("target", ["q", "q[:]", "q[0:2]"])
def test_whole_array_and_complete_slices_have_identical_matrix_order(
    target: str,
) -> None:
    matrix = random_unitary(2)
    entry = program_definition(
        "from flagquantum.synth import *\n"
        f"MATRIX = {tuple(tuple(row) for row in matrix.tolist())!r}\n"
        "@qfunc\ndef main(q: Output[QArray[QBit]]):\n"
        "    allocate(2,q)\n    hadamard_transform(q)\n    RZ(-0.49,q[0])\n    phase(1,0.31)\n"
        f"    unitary(MATRIX,{target})\n"
    )
    reference = fq.Circuit(2, dtype=torch.complex128).h(0).h(1).rz(1, theta=-0.49)
    reference.any(
        0,
        matrix=torch.eye(2, dtype=torch.complex128)
        * torch.exp(torch.tensor(0.31j, dtype=torch.complex128)),
    )
    reference.any(0, 1, matrix=matrix)
    compare_states(synth.synthesize(entry), reference, ("q",))


@pytest.mark.parametrize(
    ("target", "wires"),
    [
        ("q[1:3]", (1, 2)),
        ("(q[1],q[2])", (1, 2)),
        ("[q[1],q[2]]", (1, 2)),
        ("(q[2],q[1])", (2, 1)),
    ],
)
def test_local_slices_and_explicit_scalar_lists_preserve_their_documented_order(
    target: str, wires: tuple[int, int]
) -> None:
    matrix = random_unitary(2)
    entry = program_definition(
        "from flagquantum.synth import *\n"
        f"MATRIX = {tuple(tuple(row) for row in matrix.tolist())!r}\n"
        "@qfunc\ndef helper(q: QArray[QBit]):\n"
        f"    unitary(MATRIX,{target})\n"
        "@qfunc\ndef main(q: Output[QArray[QBit]]):\n"
        "    allocate(4,q)\n    hadamard_transform(q)\n    RZ(0.73,q[1])\n    helper(q)\n"
    )
    reference = fq.Circuit(4, dtype=torch.complex128)
    for wire in range(4):
        reference.h(wire)
    reference.rz(2, theta=0.73).any(*wires, matrix=matrix)
    compare_states(synth.synthesize(entry), reference, ("q",))


@pytest.mark.parametrize("width", [4, 5])
def test_existing_matrix_width_boundary_and_zero_workspace_identity(width: int) -> None:
    matrix = tuple(
        tuple(int(i == j) for j in range(1 << width)) for i in range(1 << width)
    )
    entry = definition("MATRIX", "X(q[0])", prefix=f"MATRIX={matrix!r}", width=width)
    reference = fq.Circuit(width + 1, dtype=torch.complex128).x(0).x(width)
    compare_states(synth.synthesize(entry), reference, ("sentinel", "q"))


@pytest.mark.parametrize(
    ("matrix", "prefix", "diagnostic"),
    [
        ("[[1,0],[0,2]]", "", "register.unitary"),
        ("[[1,0,0],[0,1,0]]", "", "register.unitary"),
        ("[[True,0],[0,1]]", "", "program.matrix"),
        ("[[1e999,0],[0,1]]", "", "register.unitary"),
        ("MATRIX", "MATRIX=[[1,0],[0,1]]", "program.matrix"),
        ("MATRIX", "MATRIX=((1,0),[0,1])", "program.matrix"),
        ("MATRIX", "MATRIX=((1,0),(0,1j))", "register.unitary"),
        (
            "make_matrix()",
            "def make_matrix():\n    raise AssertionError('host matrix called')",
            "program.matrix",
        ),
        ("[[1/2,0],[0,1]]", "", "program.matrix"),
        ("[]", "", "program.matrix"),
        ("MATRIX", "MATRIX=tuple((1,0) for _ in range(33))", "program.matrix"),
    ],
)
def test_malformed_matrices_fail_before_execution(
    matrix: str, prefix: str, diagnostic: str
) -> None:
    with pytest.raises(HybridVerificationError, match=diagnostic):
        synth.synthesize(definition(matrix, "", prefix=prefix))


def test_shadowed_matrix_data_and_duplicate_scalar_locations_are_refused() -> None:
    for target in ("(q[0],q[0])", "[q[1],q[1]]"):
        entry = program_definition(
            "from flagquantum.synth import *\n"
            "@qfunc\ndef main(q: Output[QArray[QBit]]):\n    allocate(2,q)\n"
            f"    unitary([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],{target})\n"
        )
        with pytest.raises(HybridVerificationError, match="register.unitary"):
            synth.synthesize(entry)
    with pytest.raises(HybridVerificationError, match="program.matrix"):
        synth.synthesize(definition("q", ""))


def test_const_and_uninitialized_targets_are_refused() -> None:
    for qualifier, initialization in (("Const", "allocate(q)"), ("Output", "")):
        source = "from flagquantum.synth import *\n"
        if qualifier == "Const":
            source += (
                "@qfunc\ndef helper(q: Const[QNum[1]]):\n    unitary([[0,1],[1,0]],q)\n"
            )
            source += "@qfunc\ndef main(q: Output[QNum[1]]):\n    allocate(q)\n    helper(q)\n"
        else:
            source += (
                "@qfunc\ndef main(q: Output[QNum[1]]):\n    unitary([[0,1],[1,0]],q)\n"
            )
        with pytest.raises(HybridVerificationError, match="unitary|initialized"):
            synth.synthesize(program_definition(source))


@pytest.mark.parametrize("name", ["unitary", "quantum_volume"])
def test_native_originals_retain_every_complex_matrix_entry_and_full_state(
    name: str,
) -> None:
    import importlib

    catalog = json.loads(
        (ROOT / "tests/fixtures/classiq_qmod/corpus_catalog.json").read_text()
    )
    case = next(c for c in catalog["cases"] if c["id"].endswith(f"/{name}.qmod"))
    native = importlib.import_module(case["native_example"][:-3].replace("/", "."))
    source = (ROOT / case["original"]["path"]).read_text()
    start = source.index("[", source.index("unitary("))
    depth = 0
    end = start
    for end in range(start, len(source)):
        depth += (source[end] == "[") - (source[end] == "]")
        if not depth:
            break
    assert (
        tuple(tuple(row) for row in ast.literal_eval(source[start : end + 1]))
        == native.MATRIX
    )
    comparison = importlib.import_module(
        case["native_comparison"][:-3].replace("/", ".")
    )
    registers = ("x",) if name == "unitary" else ("target",)
    for program in (
        native.build_program(),
        synth.synthesize(ROOT / case["original"]["path"]),
    ):
        compare_states(program, comparison.reference_circuit(), registers)
        assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    "call",
    [
        "unitary()",
        "unitary([[0,1],[1,0]])",
        "unitary([[0,1],[1,0]],q,q)",
        "unitary(matrix=[[0,1],[1,0]],target=q)",
        "unitary(*([[0,1],[1,0]],q))",
    ],
)
def test_invalid_call_shapes_fail_closed(call: str) -> None:
    entry = program_definition(
        "from flagquantum.synth import *\n@qfunc\ndef main(q: Output[QBit]):\n    allocate(q)\n    "
        + call
        + "\n"
    )
    with pytest.raises(HybridVerificationError, match="program.signature"):
        synth.synthesize(entry)


def test_host_numeric_conversion_hooks_are_not_executed() -> None:
    entry = definition(
        "MATRIX",
        "",
        prefix="class Evil:\n    def __complex__(self):\n        raise AssertionError('conversion hook executed')\nMATRIX=((Evil(),0),(0,1))",
    )
    with pytest.raises(HybridVerificationError, match="built-in numbers"):
        synth.synthesize(entry)


def test_more_than_five_target_bits_are_refused_before_matrix_lowering() -> None:
    entry = definition("[[0,1],[1,0]]", "", width=6)
    with pytest.raises(HybridVerificationError, match="one to five"):
        synth.synthesize(entry)


@pytest.mark.parametrize(
    ("qubits", "operations", "memory"),
    [(1, None, None), (None, 1, None), (None, None, 16)],
)
def test_existing_resource_budgets_include_the_decomposed_matrix(
    qubits: int | None, operations: int | None, memory: int | None
) -> None:
    entry = definition(repr(random_unitary(2).tolist()), "")
    with pytest.raises((synth.SynthesisError, PlanningError)):
        synth.synthesize(
            entry,
            max_qubits=qubits,
            max_operations=operations,
            memory_limit_bytes=memory,
        )


def test_notebook_scope_separates_static_matrix_workflow_from_quantum_volume_study() -> (
    None
):
    inventory = json.loads(
        (ROOT / "tests/fixtures/classiq_qmod/native_library_inventory.json").read_text()
    )
    static = next(
        g
        for g in inventory["groups"]
        if g["id"]
        == "functions/qmod_library_reference/qmod_core_library/unitary/unitary"
    )
    assert static["workflow_status"] == "validated"
    assert [
        cell["index"] for cell in static["workflow_validation"]["reviewed_code_cells"]
    ] == [2, 3]
    volume = next(
        g
        for g in inventory["groups"]
        if g["id"] == "applications/benchmarking/quantum_volume/quantum_volume"
    )
    assert volume["native_program"]
    assert volume["workflow_status"] == "pending"
