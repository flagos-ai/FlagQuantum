"""Native roles translate to the existing Compiler Const/call contract."""

import pytest

from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.ecosystem._qmod.adapter import (
    SUPPORTED_SOURCE_VERSION,
    QmodImport,
    parse_qmod,
)

pytestmark = pytest.mark.unit


def parse(source: str) -> QmodImport:
    return parse_qmod(
        source, filename="const.qmod", source_version=SUPPORTED_SOURCE_VERSION
    )


@pytest.mark.parametrize(
    "signature,call",
    [
        ("const x:qnum<2>,const y:qnum<2>,output result:qnum", "x,y,result"),
        ("output result:qnum,const y:qnum<2>,const x:qnum<2>", "result,y,x"),
        ("const x:qnum<2>,output result:qnum,const y:qnum<2>", "x,result,y"),
    ],
)
def test_source_order_is_translated_to_const_first_owned_call_abi(
    signature: str, call: str
) -> None:
    imported = parse(
        f"qfunc compute({signature}) {{ result=x*x+y; }} qfunc main(output x:qnum<2>,output y:qnum<2>,output result:qnum) {{ allocate(x); allocate(y); compute({call}); }}"
    )
    block = imported.program.body.blocks[0]
    registers = {
        op.attributes["name"]: op.results[0]
        for op in block.operations
        if op.name == "qreg.declare"
    }
    invocation = next(op for op in block.operations if op.name == "qreg.call")
    expected = (
        (registers["x"], registers["y"], registers["result"])
        if signature.startswith("const x")
        else (registers["y"], registers["x"], registers["result"])
    )
    assert invocation.operands[1:] == expected
    helper = next(f for f in imported.functions if f.name == "compute")
    assert len(helper.body.blocks[0].arguments) == 3
    assert (
        len(
            [op for op in helper.body.blocks[0].operations if op.name == "qreg.declare"]
        )
        == 1
    )
    assert imported.inferred_widths == {"result": 4}


@pytest.mark.parametrize(
    "signature,body,code",
    [
        ("const q:qnum", "", "qmod.width"),
        ("input q:qbit", "", "qmod.role"),
        ("input const q:qbit", "", "qmod.role"),
        ("output const q:qbit", "", "qmod.role"),
        ("q:qbit", "allocate(q);", "qmod.scope"),
        ("const q:qbit[]", "", "qmod.type"),
        ("const q:qbit,const q:qbit", "", "qmod.name"),
        ("const q:qbit", "H(q);", "construction.register.const"),
        ("const q:qbit", "Z(q);", "construction.register.const"),
        ("const q:qbit", "X(q); X(q);", "construction.register.const"),
        ("const q:qbit", "allocate(q);", "construction.register.const"),
        ("const q:qbit", "q=1;", "construction.register.const"),
        ("const q:qbit", "q=q*q;", "construction.register.const"),
    ],
)
def test_invalid_unused_const_definitions_do_not_borrow_caller_state(
    signature: str, body: str, code: str
) -> None:
    with pytest.raises(HybridVerificationError) as error:
        parse(
            f"qfunc helper({signature}) {{ {body} }}\nqfunc main(output q:qbit) {{ q=0; }}"
        )
    assert error.value.diagnostics[0].code == code
    assert "const.qmod:1:" in error.value.diagnostics[0].message


@pytest.mark.parametrize(
    "declarations,body,code",
    [
        ("output x:qnum", "borrow(x); allocate(2,x);", "qmod.width"),
        (
            "output x:qnum<2>",
            "borrow(x); allocate(x);",
            "construction.register.uninitialized",
        ),
        ("output x:qnum<3>", "allocate(x); borrow(x);", "qmod.signature"),
        ("output x:qbit", "allocate(x); borrow(x);", "qmod.signature"),
    ],
)
def test_const_actuals_need_existing_known_exact_kind_and_width(
    declarations: str, body: str, code: str
) -> None:
    with pytest.raises(HybridVerificationError) as error:
        parse(
            f"qfunc borrow(const x:qnum<2>) {{}}\nqfunc main({declarations}) {{ {body} }}"
        )
    assert error.value.diagnostics[0].code == code
    assert "const.qmod:2:" in error.value.diagnostics[0].message


def test_const_cannot_be_a_nested_output_and_outputs_cannot_alias_borrows() -> None:
    source = "qfunc fresh(output q:qbit) { allocate(q); } qfunc wrong(const q:qbit) { fresh(q); } qfunc main(output q:qbit) { allocate(q); }"
    with pytest.raises(HybridVerificationError, match="construction.register.const"):
        parse(source)
    with pytest.raises(HybridVerificationError, match="construction.register.alias"):
        parse(
            "qfunc copy(const x:qnum<2>,output out:qnum) { out=x; } qfunc main(output x:qnum<2>) { allocate(x); copy(x,x); }"
        )


def test_empty_const_helpers_are_valid_but_not_a_source_entry() -> None:
    imported = parse(
        "qfunc unused(const q:qbit) {} qfunc main(output q:qbit) { allocate(q); }"
    )
    assert [f.name for f in imported.functions] == ["unused"]
    with pytest.raises(HybridVerificationError, match="qmod.role"):
        parse("qfunc main(const q:qbit) {}")


def test_empty_call_tree_keeps_the_compiler_expansion_budget() -> None:
    definitions = ["qfunc f0(const q:qbit) {}"]
    definitions.extend(
        f"qfunc f{i}(const q:qbit) {{ f{i-1}(q); f{i-1}(q); }}" for i in range(1, 13)
    )
    with pytest.raises(HybridVerificationError, match="construction.register.budget"):
        parse(
            " ".join(definitions)
            + " qfunc main(output q:qbit) { allocate(q); f12(q); }"
        )
