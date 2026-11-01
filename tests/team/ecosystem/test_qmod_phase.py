"""Native sign phases reuse the verified one-bit register borrow contract."""

import pytest

from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.ecosystem._qmod.adapter import SUPPORTED_SOURCE_VERSION, parse_qmod

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "kind,allocation", [("qbit", "x"), ("qnum<1>", "x"), ("qnum", "1,x")]
)
def test_native_phase_builds_owned_one_bit_callee(kind, allocation):
    imported = parse_qmod(
        f"qfunc main(output x:{kind}) {{ allocate({allocation}); phase(x*x,pi); }}",
        filename="predicate.qmod",
        source_version=SUPPORTED_SOURCE_VERSION,
    )
    operations = imported.program.body.blocks[0].operations
    phase = next(op for op in operations if op.name == "qreg.phase")
    graph = imported.functions[0]
    block = graph.body.blocks[0]
    assert phase.attributes == {"callee": graph.name}
    assert len(phase.operands) == 2  # Effect and one unique immutable borrow.
    assert block.operations[0].operands == (block.arguments[0],) * 2
    assert block.operations[-1].operands[0].type.parameters == (1,)
    assert phase.location.filename == "predicate.qmod"
    assert imported.source_version == SUPPORTED_SOURCE_VERSION


@pytest.mark.parametrize(
    "declaration,body,code",
    [
        ("output x:qbit", "allocate(x); phase();", "qmod.signature"),
        ("output x:qbit", "allocate(x); phase(pi);", "qmod.signature"),
        ("output x:qbit", "allocate(x); phase(x,pi,pi);", "qmod.signature"),
        ("output phase:qbit", "allocate(phase); phase(phase,pi);", "qmod.signature"),
        ("output x:qbit", "allocate(x); phase(x,1);", "qmod.coefficient"),
        ("output x:qbit", "allocate(x); phase(x,0);", "qmod.coefficient"),
        ("output x:qbit", "allocate(x); phase(x,pi+0);", "qmod.coefficient"),
        ("output pi:qbit", "allocate(pi); phase(pi,pi);", "qmod.coefficient"),
        ("output x:qbit", "allocate(x); phase(x,missing);", "qmod.coefficient"),
        ("output x:qbit", "allocate(x); phase(x,pi/2);", "qmod.operator"),
        ("output x:qbit", "allocate(x); phase(1,pi);", "qmod.phase"),
        ("output x:qbit", "allocate(x); phase(0,pi);", "qmod.phase"),
        ("output x:qbit", "allocate(x); phase(x+1,pi);", "qmod.phase"),
        ("output x:qnum<2>", "x=0; phase(x,pi);", "qmod.phase"),
        ("output x:qnum", "phase(x,pi);", "qmod.width"),
        ("output x:qbit", "allocate(x); phase(missing,pi);", "qmod.name"),
        ("output x:qbit", "allocate(x); phase(x==1,pi);", "qmod.operator"),
    ],
)
def test_unsupported_native_phase_fails_with_source_location(declaration, body, code):
    with pytest.raises(HybridVerificationError) as error:
        parse_qmod(
            f"qfunc main({declaration}) {{\n{body}\n}}",
            filename="reject.qmod",
            source_version=SUPPORTED_SOURCE_VERSION,
        )
    diagnostic = error.value.diagnostics[0]
    assert diagnostic.code == code
    assert "reject.qmod:2:" in diagnostic.message


def test_typed_but_uninitialized_borrow_uses_existing_register_verifier():
    with pytest.raises(HybridVerificationError, match="initializ"):
        parse_qmod(
            "qfunc main(output x:qbit) { phase(x,pi); }",
            source_version=SUPPORTED_SOURCE_VERSION,
        )


def test_quantum_dependent_constant_phase_keeps_its_quantum_argument():
    imported = parse_qmod(
        "qfunc main(output x:qbit) { allocate(x); phase(x*0+1,pi); }",
        source_version=SUPPORTED_SOURCE_VERSION,
    )
    block = imported.functions[0].body.blocks[0]
    assert len(block.arguments) == 1
    assert [op.name for op in block.operations] == [
        "arith.constant",
        "qint.multiply",
        "arith.constant",
        "qint.add",
        "qint.return",
    ]
