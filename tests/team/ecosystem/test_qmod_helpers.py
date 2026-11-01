"""Fresh-output helper definitions remain bounded owned-register producers."""

import pytest

from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.ecosystem._qmod.adapter import SUPPORTED_SOURCE_VERSION, parse_qmod

pytestmark = pytest.mark.unit


def parse(source):
    return parse_qmod(
        source, filename="helpers.qmod", source_version=SUPPORTED_SOURCE_VERSION
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_forward_nested_repeated_helpers_deduplicate_owned_arithmetic(reverse):
    pieces = [
        "qfunc leaf(output a:qnum<1>,output b:qnum) { allocate(a); hadamard_transform(a); b=a*a; }",
        "qfunc wrap(output q:qnum<1>,output r:qnum) { leaf(q,r); }",
        "qfunc main(output a:qnum,output b:qnum,output c:qnum,output d:qnum) { wrap(a,b); wrap(c,d); }",
    ]
    imported = parse("\n".join(reversed(pieces) if reverse else pieces))
    assert dict(imported.inferred_widths) == dict.fromkeys(["a", "b", "c", "d"], 1)
    assert [f.name for f in imported.functions] == ["leaf.expr2"]
    computes = [
        op
        for op in imported.program.body.blocks[0].operations
        if op.name == "qreg.compute"
    ]
    assert len(computes) == 2
    assert computes[0].operands[1:] != computes[1].operands[1:]
    assert all(op.location.filename == "helpers.qmod" for op in computes)


@pytest.mark.parametrize(
    "suffix,code",
    [
        ("qfunc main(output q:qbit) { seed(q); }", None),
        ("qfunc main(output q:qbit) { seed(); }", "qmod.signature"),
        ("qfunc main(output q:qbit) { seed(q,q); }", "qmod.signature"),
        ("qfunc main(output q:qbit) { seed(1); }", "qmod.signature"),
        ("qfunc main(output q:qbit) { seed(q+1); }", "qmod.signature"),
        ("qfunc main(output q:qnum<1>) { seed(q); }", "qmod.signature"),
        ("qfunc main(output q:qbit) { seed(missing); }", "qmod.name"),
        ("qfunc main(output seed:qbit) { seed(seed); }", "qmod.call"),
        (
            "qfunc main(output q:qbit) { q=0; seed(q); }",
            "construction.register.initialized",
        ),
        (
            "qfunc main(output q:qbit) { seed(q); seed(q); }",
            "construction.register.initialized",
        ),
    ],
)
def test_helper_signature_and_freshness(suffix, code):
    source = "qfunc seed(output q:qbit) { allocate(q); }\n" + suffix
    if code is None:
        parse(source)
    else:
        with pytest.raises(HybridVerificationError) as error:
            parse(source)
        assert error.value.diagnostics[0].code == code
        assert "helpers.qmod:2:" in error.value.diagnostics[0].message


@pytest.mark.parametrize(
    "source,code",
    [
        (
            "qfunc main(output q:qbit) { q=0; } qfunc main(output q:qbit) { q=0; }",
            "qmod.name",
        ),
        (
            "qfunc h(output q:qbit) { q=0; } qfunc h(output q:qbit) { q=0; } qfunc main(output q:qbit) { q=0; }",
            "qmod.name",
        ),
        (
            "qfunc allocate(output q:qbit) { q=0; } qfunc main(output q:qbit) { q=0; }",
            "qmod.name",
        ),
        (
            "qfunc phase(output q:qbit) { q=0; } qfunc main(output q:qbit) { q=0; }",
            "qmod.name",
        ),
        (
            "qfunc h(output q:qbit) { h(q); } qfunc main(output q:qbit) { q=0; }",
            "qmod.recursion",
        ),
        (
            "qfunc a(output q:qbit) { b(q); } qfunc b(output q:qbit) { a(q); } qfunc main(output q:qbit) { q=0; }",
            "qmod.recursion",
        ),
        ("qfunc main(output q:qbit) { main(q); }", "qmod.call"),
        (
            "qfunc h(output q:qbit) { unknown(q); } qfunc main(output q:qbit) { q=0; }",
            "qmod.call",
        ),
        (
            "qfunc h(output q:qbit) {} qfunc main(output q:qbit) { q=0; }",
            "construction.register.return",
        ),
        (
            "qfunc h(output q:qbit) { H(q); } qfunc main(output q:qbit) { q=0; h(q); }",
            "construction.register.uninitialized",
        ),
        (
            "qfunc h(output q:qbit) { q=outside; } qfunc main(output outside:qbit) { outside=0; }",
            "qmod.name",
        ),
        ("qfunc h(const q:qbit) {} qfunc main(output q:qbit) { q=0; }", "qmod.role"),
    ],
)
def test_invalid_definitions_are_rejected_even_when_unused(source, code):
    with pytest.raises(HybridVerificationError) as error:
        parse(source)
    assert error.value.diagnostics[0].code == code
    assert "helpers.qmod:" in error.value.diagnostics[0].message


@pytest.mark.parametrize("width", [1, 3])
def test_caller_width_never_coerces_verified_output(width):
    with pytest.raises(HybridVerificationError, match="qmod.(width|signature)"):
        parse(
            f"qfunc seed(output q:qnum<2>) {{ allocate(q); }} qfunc main(output x:qnum<{width}>) {{ seed(x); }}"
        )


def test_distinct_output_formals_cannot_alias_one_actual():
    with pytest.raises(HybridVerificationError, match="qmod.alias"):
        parse(
            "qfunc pair(output a:qbit,output b:qbit) { a=0; b=0; } qfunc main(output x:qbit) { pair(x,x); }"
        )


def test_definition_count_budget_is_explicit():
    helpers = [f"qfunc f{i}(output q:qbit) {{ q=0; }}" for i in range(32)]
    with pytest.raises(HybridVerificationError, match="qmod.budget.functions"):
        parse(" ".join(helpers) + " qfunc main(output q:qbit) { q=0; }")


def test_maximum_acyclic_chain_is_admitted_without_recursive_source_expansion():
    helpers = ["qfunc f0(output q:qbit) { allocate(q); }"]
    helpers.extend(f"qfunc f{i}(output q:qbit) {{ f{i-1}(q); }}" for i in range(1, 31))
    imported = parse(" ".join(helpers) + " qfunc main(output x:qbit) { f30(x); }")
    assert [op.name for op in imported.program.body.blocks[0].operations] == [
        "qreg.declare",
        "qreg.allocate",
        "qreg.return",
    ]


def test_pi_coefficient_symbol_cannot_be_redefined_as_helper():
    with pytest.raises(HybridVerificationError, match="qmod.name"):
        parse("qfunc pi(output q:qbit) { q=0; } qfunc main(output q:qbit) { q=0; }")


def test_expansion_budget_counts_verified_inlined_effects():
    helper = "qfunc h(output q:qbit) { allocate(q); " + "H(q);" * 512 + " }"
    outputs = ",".join(f"output q{i}:qbit" for i in range(9))
    calls = " ".join(f"h(q{i});" for i in range(9))
    with pytest.raises(HybridVerificationError, match="qmod.budget.expansion"):
        parse(helper + f" qfunc main({outputs}) {{ {calls} }}")
