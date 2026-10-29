"""Native grammar, numeric adaptation and early bounded rejection without an SDK."""

from pathlib import Path

import pytest

from flagquantum.compiler._construction.selection import select_register_program
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.ecosystem._qmod.adapter import (
    SUPPORTED_SOURCE_VERSION,
    QmodImport,
    parse_qmod,
)
from flagquantum.ecosystem._qmod.syntax import MAX_BYTES, MAX_TOKENS

pytestmark = pytest.mark.unit
CORPUS = Path(__file__).resolve().parents[2] / "fixtures/classiq_qmod"


def _parse(source: str) -> QmodImport:
    return parse_qmod(
        source, filename="native_case.qmod", source_version=SUPPORTED_SOURCE_VERSION
    )


@pytest.mark.parametrize(
    "filename,widths,inferred,total",
    [
        ("X.qmod", {"q": 1}, {}, 1),
        ("synthesis_tutorial.qmod", {"x": 3, "y": 6}, {"y": 6}, 23),
        (
            "multiplication_2vars_example.qmod",
            {"a": 3, "b": 3, "res": 6},
            {"a": 3, "b": 3, "res": 6},
            24,
        ),
    ],
)
def test_unmodified_originals_parse_and_lower_without_shrinking_declarations(
    filename: str, widths: dict[str, int], inferred: dict[str, int], total: int
) -> None:
    source = (CORPUS / filename).read_bytes()
    imported = parse_qmod(
        source.decode(),
        filename=str(CORPUS / filename),
        source_version=SUPPORTED_SOURCE_VERSION,
    )
    selection = select_register_program(imported.program, imported.functions)
    assert selection.complete and selection.selected is not None
    assert {name: len(ports) for name, ports in selection.registers.items()} == widths
    assert dict(imported.inferred_widths) == inferred
    assert imported.source_version == SUPPORTED_SOURCE_VERSION
    assert selection.selected.resources.n_qubits == total
    assert (CORPUS / filename).read_bytes() == source


def test_width_inference_uses_register_domains_not_prepared_values() -> None:
    imported = _parse(
        "qfunc main(output a:qnum, output b:qnum, output res:qnum) { a=4; b=5; res=a*b; }"
    )
    assert imported.inferred_widths == {"a": 3, "b": 3, "res": 6}
    # One observed value 20 fits five bits; the declared operand domains reach 49.
    assert imported.functions[0].body.blocks[0].operations[-1].operands[
        0
    ].type.parameters == (6,)
    with pytest.raises(TypeError):
        imported.inferred_widths["res"] = 5


@pytest.mark.parametrize(
    "expression,expected", [("2+3*4", 14), ("(2+3)*4", 20), ("(1+2)**2", 9), ("0", 0)]
)
def test_classical_literal_folding_initializes_quantum_outputs(
    expression: str, expected: int
) -> None:
    imported = _parse(f"qfunc main(output result:qnum) {{ result={expression}; }}")
    assert not imported.functions
    op = next(
        op
        for op in imported.program.body.blocks[0].operations
        if op.name == "qreg.initialize"
    )
    assert op.attributes["value"] == expected
    assert imported.inferred_widths["result"] == max(1, expected.bit_length())


def test_quantum_aliases_and_classical_subexpressions_use_existing_records() -> None:
    imported = _parse(
        "qfunc main(output x:qnum<2>, output y:qnum) { allocate(x); y=x*x+(1+2); }"
    )
    graph = imported.functions[0]
    block = graph.body.blocks[0]
    assert len(block.arguments) == 1
    multiply = next(op for op in block.operations if op.name == "qint.multiply")
    assert multiply.operands[0] == multiply.operands[1] == block.arguments[0]
    constants = [
        op.attributes["value"] for op in block.operations if op.name == "arith.constant"
    ]
    assert constants == [3]


def test_explicit_width_and_allocation_sizes_are_preserved() -> None:
    imported = _parse(
        "qfunc main(output x:qnum, output y:qnum<8>) { allocate(3,x); y=x**2+1; }"
    )
    selection = select_register_program(imported.program, imported.functions)
    assert {name: len(ports) for name, ports in selection.registers.items()} == {
        "x": 3,
        "y": 8,
    }
    assert dict(imported.inferred_widths) == {"x": 3}


def test_filename_and_comments_do_not_change_semantic_identities() -> None:
    source = "qfunc main(output q:qbit) { allocate(q); H(q); S(q); Z(q); X(q); }"
    first = _parse(source)
    second = parse_qmod(
        "// preface\r\n" + source.replace("H(q);", "/* block\ncomment */ H(q);"),
        filename="renamed.qmod",
        source_version=SUPPORTED_SOURCE_VERSION,
    )
    assert first.program.semantic_identity == second.program.semantic_identity
    assert not second.functions
    assert second.program.location.filename == "renamed.qmod"


def test_relabelled_source_is_compiled_generically_without_filename_dispatch() -> None:
    imported = _parse(
        "qfunc main(output left:qnum, output right:qnum, output answer:qnum) { left=2; right=3; answer=left*right+left; }"
    )
    selection = select_register_program(imported.program, imported.functions)
    assert tuple(selection.registers) == ("left", "right", "answer")
    assert imported.inferred_widths == {"left": 2, "right": 2, "answer": 4}


@pytest.mark.parametrize(
    "source,code",
    [
        ("", "qmod.syntax"),
        ("qfunc main() {}", "construction.register.return"),
        ("qfunc main(output q:qbit) { q=2; }", "qmod.width"),
        ("qfunc main(output x:qnum,output y:qnum) { y=x+1; }", "qmod.width"),
        ("qfunc helper(output q:qbit) { allocate(q); }", "qmod.entry"),
        ("qfunc main(input q:qbit) {}", "qmod.role"),
        ("qfunc main(const q:qbit) {}", "qmod.role"),
        ("qfunc main(q:qbit) {}", "qmod.role"),
        ("qfunc main(output a:qbit,output a:qbit) {}", "qmod.name"),
        ("qfunc main(output q:qarray) {}", "qmod.type"),
        ("qfunc main(output q:qbit[]) {}", "qmod.type"),
        ("qfunc main(output q:qnum<3,SIGNED,0>) {}", "qmod.type"),
        ("qfunc main(output q:qbit<1>) {}", "qmod.type"),
        ("qfunc main(output q:qnum<0>) {}", "qmod.width"),
        ("qfunc main(output q:qnum<257>) {}", "qmod.width"),
        ("qfunc main(output q:qbit,) {}", "qmod.syntax"),
        ("qfunc main(output q:qbit) { local:qbit; }", "qmod.statement"),
        ("qfunc main(output q:qbit) { q^=1; }", "qmod.statement"),
        ("qfunc main(output q:qbit) { q+=1; }", "qmod.statement"),
        ("qfunc main(output q:qbit) { q=-1; }", "qmod.expression"),
        ("qfunc main(output q:qbit) { q=true; }", "qmod.name"),
        ("qfunc main(output q:qbit) { q=missing; }", "qmod.name"),
        ("qfunc main(output q:qbit) { missing=1; }", "qmod.name"),
        ("qfunc main(output q:qnum) { allocate(q); }", "qmod.width"),
        ("qfunc main(output q:qnum<3>) { allocate(2,q); }", "qmod.width"),
        ("qfunc main(output q:qbit) { allocate(); }", "qmod.signature"),
        ("qfunc main(output q:qnum) { allocate(0,q); }", "qmod.width"),
        ("qfunc main(output q:qnum) { allocate(257,q); }", "qmod.width"),
        ("qfunc main(output q:qbit) { allocate(q,q); }", "qmod.width"),
        ("qfunc main(output q:qbit) { allocate(1,0,q); }", "qmod.signature"),
        ("qfunc main(output q:qbit) { allocate(1,1); }", "qmod.signature"),
        ("qfunc main(output q:qnum<3>) { allocate(q); X(q); }", "qmod.type"),
        ("qfunc main(output q:qbit) { allocate(q); X(); }", "qmod.signature"),
        ("qfunc main(output q:qbit) { allocate(q); X(q+1); }", "qmod.signature"),
        ("qfunc main(output q:qbit) { allocate(q); drop(q); }", "qmod.call"),
        ("qfunc main(output q:qbit) { allocate(q); free(q); }", "qmod.call"),
        ("qfunc main(output q:qbit) { allocate(q); bind(q,q); }", "qmod.call"),
        ("qfunc main(output q:qbit) { unknown(q); }", "qmod.call"),
        ("qfunc main(output q:qnum<2>) { allocate(q); q=q**3; }", "qmod.operator"),
        ("qfunc main(output q:qbit) { q=1.5; }", "qmod.operator"),
        ("qfunc main(output q:qbit) { q=1/2; }", "qmod.operator"),
        ("qfunc main(output q:qnum<2>) { allocate(q); X(q[0]); }", "qmod.operator"),
        ("qfunc main(output q:qbit) { q=sin(1); }", "qmod.operator"),
        ("qfunc main(output q:qbit) { q=1; } cscope {}", "qmod.syntax"),
        ("qfunc main(output q:qbit) { q=1 }", "qmod.syntax"),
        ("qfunc main(output q:qbit) { q=", "qmod.expression"),
        ("/* unfinished", "qmod.token"),
        ("$", "qmod.token"),
    ],
)
def test_unsupported_or_malformed_source_is_not_partially_admitted(
    source: str, code: str
) -> None:
    with pytest.raises(HybridVerificationError) as caught:
        _parse(source)
    assert caught.value.diagnostics[0].code == code
    assert "native_case.qmod:" in caught.value.diagnostics[0].message


@pytest.mark.parametrize(
    "body,code",
    [
        ("X(q);", "construction.register.uninitialized"),
        ("allocate(q); allocate(q);", "construction.register.initialized"),
        ("q=1; q=0;", "construction.register.initialized"),
        ("", "construction.register.return"),
    ],
)
def test_initialization_errors_are_delegated_to_the_shared_verifier(
    body: str, code: str
) -> None:
    with pytest.raises(HybridVerificationError) as caught:
        _parse(f"qfunc main(output q:qbit) {{ {body} }}")
    assert caught.value.diagnostics[0].code == code
    assert "native_case.qmod:" in caught.value.diagnostics[0].message


def test_diagnostic_retains_exact_native_line_and_column() -> None:
    source = "qfunc main(output q:qbit) {\n  allocate(q);\n  X(missing);\n}"
    with pytest.raises(HybridVerificationError) as caught:
        _parse(source)
    assert caught.value.diagnostics[0].message.startswith("native_case.qmod:3:5:")


@pytest.mark.parametrize(
    "source,code",
    [
        (" " * (MAX_BYTES + 1), "qmod.budget.source"),
        ("; " * (MAX_TOKENS + 1), "qmod.budget.tokens"),
        ("qfunc main(output q:qnum) { q=" + "9" * 79 + "; }", "qmod.budget.number"),
        ("qfunc main(output q:qnum) { q=" + str(1 << 256) + "; }", "qmod.width"),
        (
            "qfunc main(output q:qnum) { q=" + "(" * 34 + "1" + ")" * 34 + "; }",
            "qmod.budget.depth",
        ),
        (
            "qfunc main(output q:qnum) { q=" + "+".join(["1"] * 34) + "; }",
            "qmod.budget.depth",
        ),
        (
            "qfunc main(output x:qnum<256>, output y:qnum) { allocate(x); y=x*x; }",
            "qmod.width",
        ),
        (
            "qfunc main(output x:qnum<256>) { allocate(x); "
            + "hadamard_transform(x);" * 16
            + " }",
            "qmod.budget.expansion",
        ),
    ],
)
def test_source_and_generated_model_budgets_fail_early(source: str, code: str) -> None:
    with pytest.raises(HybridVerificationError) as caught:
        _parse(source)
    assert caught.value.diagnostics[0].code == code
    assert "native_case.qmod:" in caught.value.diagnostics[0].message


@pytest.mark.parametrize("version", ["latest", "main", "unknown"])
def test_unknown_source_versions_are_explicitly_refused(version: str) -> None:
    with pytest.raises(HybridVerificationError, match="qmod.version"):
        parse_qmod("qfunc main(output q:qbit) { q=1; }", source_version=version)


def test_capture_never_executes_source_reads_files_or_invokes_synthesis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("source adaptation must not execute, read, or synthesize")

    monkeypatch.setattr(
        "flagquantum.compiler._construction.frontend_python.capture", forbidden
    )
    monkeypatch.setattr(
        "flagquantum.compiler._construction.selection.select_register_program",
        forbidden,
    )
    monkeypatch.setattr(Path, "read_text", forbidden)
    parsed = _parse("qfunc main(output q:qbit) { allocate(q); X(q); }")
    assert not parsed.functions
    destination = tmp_path / "must-not-exist"
    malicious = f"__import__('pathlib').Path({str(destination)!r}).write_text('bad')"
    with pytest.raises(HybridVerificationError):
        _parse(malicious)
    assert not destination.exists()


@pytest.mark.parametrize(
    "source,filename", [(None, "x"), (b"qfunc main", "x"), ("", None), ("", " ")]
)
def test_invalid_transport_arguments_are_refused(source, filename) -> None:
    with pytest.raises(TypeError, match="strings"):
        parse_qmod(source, filename=filename, source_version=SUPPORTED_SOURCE_VERSION)


def test_compiler_combined_node_budget_remains_authoritative() -> None:
    source = (
        "qfunc main(output x:qnum<255>,output q:qbit) { allocate(x); allocate(q); "
        + "hadamard_transform(x);" * 16
        + "H(q);" * 14
        + " }"
    )
    with pytest.raises(HybridVerificationError) as caught:
        _parse(source)
    assert caught.value.diagnostics[0].code == "construction.register.budget"
    assert "native_case.qmod:" in caught.value.diagnostics[0].message
