"""Native scope syntax delegates lifetime/Const proof to existing Compiler records."""

import pytest

from flagquantum.compiler._construction.register_profile import register_ports
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.ecosystem._qmod import context
from flagquantum.ecosystem._qmod.adapter import (
    SUPPORTED_SOURCE_VERSION,
    QmodImport,
    parse_qmod,
)

pytestmark = pytest.mark.unit


def parse(
    body: str, *, declarations: str = "flag:qbit;", helpers: str = ""
) -> QmodImport:
    return parse_qmod(
        f"{helpers} qfunc main(output x:qnum<2>,output y:qnum<2>) {{ allocate(x); allocate(y); {declarations} {body} }}",
        filename="scope.qmod",
        source_version=SUPPORTED_SOURCE_VERSION,
    )


@pytest.mark.parametrize("kind", ["qbit", "qnum<1>", "qnum"])
def test_temporary_declarations_remain_hidden_and_use_owned_lifecycle(
    kind: str,
) -> None:
    imported = parse(
        "within { flag=x<y; } apply { phase(flag,pi); }", declarations=f"flag:{kind};"
    )
    ops = imported.program.body.blocks[0].operations
    local = next(op.results[0] for op in ops if op.name == "qreg.local")
    assert local.type.parameters == (1,)
    assert local not in ops[-1].operands
    assert not imported.inferred_widths
    assert [op.name for op in ops] == [
        "qreg.declare",
        "qreg.declare",
        "qreg.allocate",
        "qreg.allocate",
        "qreg.local",
        "qreg.compute",
        "qreg.phase",
        "qreg.uncompute",
        "qreg.release",
        "qreg.return",
    ]
    assert all(
        op.location.filename == "scope.qmod" for op in ops if op.location is not None
    )


def test_same_source_variable_rebinds_with_new_identity_and_reuses_zero_slots() -> None:
    imported = parse(
        "within { flag=x<y; } apply { phase(flag,pi); } within { flag=x==y; } apply { phase(flag,pi); }",
        declarations="flag:qnum;",
    )
    ops = imported.program.body.blocks[0].operations
    locals_ = [op for op in ops if op.name == "qreg.local"]
    assert len({op.results[0].id for op in locals_}) == 2
    assert len({op.attributes["name"] for op in locals_}) == 2
    names, bindings, outputs = register_ports(imported.program)
    assert names == {"x": (0, 1), "y": (2, 3)} and outputs == 4
    assert {bindings[op.results[0].id] for op in locals_} == {(4,)}
    assert not imported.inferred_widths


def test_chained_preparation_cleans_in_reverse_dependency_order() -> None:
    imported = parse(
        "within { energy=x+y; flag=energy<4; } apply { phase(flag,pi); }",
        declarations="energy:qnum; flag:qbit;",
    )
    ops = imported.program.body.blocks[0].operations
    values = [op.results[0] for op in ops if op.name == "qreg.local"]
    assert [v.type.parameters for v in values] == [(3,), (1,)]
    assert [op.operands[1] for op in ops if op.name == "qreg.uncompute"] == values[::-1]
    assert [op.operands[1] for op in ops if op.name == "qreg.release"] == values[::-1]


def test_unused_unbound_declarations_allocate_no_quantum_storage() -> None:
    imported = parse("", declarations="unused:qnum;")
    assert not imported.functions
    assert not any(
        op.name == "qreg.local" for op in imported.program.body.blocks[0].operations
    )


def test_scope_words_remain_valid_helper_and_variable_names() -> None:
    imported = parse(
        "within(x); within {flag=x<y;} apply {within(x);apply(flag);}",
        helpers="qfunc within(const apply:qnum<2>) {} qfunc apply(const within:qbit) {phase(within,pi);}",
    )
    assert [
        op.attributes["callee"]
        for op in imported.program.body.blocks[0].operations
        if op.name == "qreg.call"
    ] == ["within", "within", "apply"]


def test_empty_scope_named_helper_keeps_the_existing_register_contract() -> None:
    with pytest.raises(HybridVerificationError, match="register.return"):
        parse("within();", declarations="", helpers="qfunc within() {}")


def test_explicit_width_stays_fixed_across_scopes_and_empty_apply_is_valid() -> None:
    imported = parse(
        "within { flag=x<y; } apply {} within { flag=x+y; } apply {}",
        declarations="flag:qnum<3>;",
    )
    assert [
        op.results[0].type.parameters
        for op in imported.program.body.blocks[0].operations
        if op.name == "qreg.local"
    ] == [(3,), (3,)]


def test_quantum_dependent_fixed_value_is_not_folded_into_literal_preparation() -> None:
    imported = parse("within { flag=x*0+1; } apply { phase(flag,pi); }")
    assert any(
        op.name == "qreg.compute" for op in imported.program.body.blocks[0].operations
    )
    assert not any(
        op.name == "qreg.initialize"
        for op in imported.program.body.blocks[0].operations
    )


def test_bare_local_width_is_inferred_once_then_retained() -> None:
    with pytest.raises(HybridVerificationError, match="qmod.width"):
        parse(
            "within {flag=x<y;} apply {} within {flag=x+y;} apply {}",
            declarations="flag:qnum;",
        )


@pytest.mark.parametrize("depth", [32, 33])
def test_lexical_scope_depth_has_an_independent_bounded_limit(depth: int) -> None:
    body = "phase(t0,pi);"
    for index in reversed(range(depth)):
        body = f"within {{t{index}=x<y;}} apply {{{body}}}"
    declarations = "".join(f"t{index}:qbit;" for index in range(depth))
    if depth == 33:
        with pytest.raises(HybridVerificationError, match="qmod.budget.depth"):
            parse(body, declarations=declarations)
    else:
        imported = parse(body, declarations=declarations)
        assert (
            sum(
                op.name == "qreg.local"
                for op in imported.program.body.blocks[0].operations
            )
            == depth
        )


def test_explicit_local_allocation_requires_drop_before_return() -> None:
    with pytest.raises(HybridVerificationError, match="drop every hidden register"):
        parse("allocate(flag);")


@pytest.mark.parametrize(
    "body",
    [
        "flag=x<y;",
        "phase(flag,pi);",
        "X(flag);",
        "within { flag=x<y; } apply {} phase(flag,pi);",
        "within { flag=x<y; } apply {} read(flag);",
    ],
)
def test_local_references_are_not_available_outside_scope(body: str) -> None:
    with pytest.raises(HybridVerificationError, match="qmod.scope"):
        parse(body, helpers="qfunc read(const b:qbit) {}")


@pytest.mark.parametrize(
    "body,code",
    [
        ("within {} apply {}", "qmod.scope"),
        ("within { flag=1; } apply {}", "qmod.scope"),
        ("within { flag=1<2; } apply {}", "qmod.scope"),
        ("within { allocate(flag); } apply {}", "qmod.scope"),
        ("within { H(flag); } apply {}", "qmod.scope"),
        ("within { x=y; } apply {}", "qmod.scope"),
        ("within { flag=x<y; } apply { H(x); }", "qmod.scope"),
        ("within { flag=x<y; } apply { X(flag); }", "qmod.scope"),
        ("within { flag=x<y; } apply { flag=x==y; }", "qmod.scope"),
        ("within { flag=x<y; } apply { allocate(x); }", "qmod.scope"),
        ("within { other:qbit; } apply {}", "qmod.scope"),
        ("within { flag=x<y; } apply { other:qbit; }", "qmod.scope"),
        ("within { within {flag=x<y;} apply {} } apply {}", "qmod.scope"),
        (
            "within {flag=x<y;} apply { within {flag=x<y;} apply {} }",
            "register.initialized",
        ),
        ("within {flag=x<y;} phase(flag,pi);", "qmod.syntax"),
        ("apply {phase(flag,pi);}", "qmod.statement"),
        ("within {flag=x<y;", "qmod.syntax"),
        ("within {flag=x<y;} apply {phase(x+y,pi);}", "qmod.phase"),
        ("within {flag=x<y;flag=x==y;} apply {}", "register.initialized"),
        ("within {flag=x<y;} apply {} within {flag=x+y;} apply {}", "qmod.width"),
    ],
)
def test_unsupported_scope_forms_are_explicitly_refused(body: str, code: str) -> None:
    with pytest.raises(HybridVerificationError, match=code):
        parse(body)


@pytest.mark.parametrize(
    "declarations",
    ["flag:qbit; flag:qbit;", "x:qbit;", "pi:qbit;", "phase:qbit;", "X:qbit;"],
)
def test_local_names_cannot_shadow_existing_symbols(declarations: str) -> None:
    with pytest.raises(HybridVerificationError, match="qmod.name"):
        parse("", declarations=declarations)


def test_apply_calls_require_const_only_formals() -> None:
    with pytest.raises(HybridVerificationError, match="qmod.scope"):
        parse(
            "within { flag=x<y; } apply { prepare(flag); }",
            helpers="qfunc prepare(output out:qbit) { allocate(out); }",
        )


def test_helper_prepare_calls_are_not_inferred_as_invertible_blocks() -> None:
    with pytest.raises(HybridVerificationError, match="qmod.scope"):
        parse(
            "within { prepare(flag); } apply {}",
            helpers="qfunc prepare(output out:qbit) { allocate(out); }",
        )


def test_compiler_owns_const_mutation_refusal_even_through_apply_helpers() -> None:
    with pytest.raises(HybridVerificationError, match="register.const"):
        parse(
            "within {flag=x<y;} apply { mutate(flag); }",
            helpers="qfunc mutate(const b:qbit) { X(b); }",
        )


def test_unprepared_local_reads_are_rejected_by_compiler() -> None:
    with pytest.raises(HybridVerificationError, match="register.uninitialized"):
        parse(
            "within {flag=x<y;} apply { read(other); }",
            declarations="flag:qbit; other:qbit;",
            helpers="qfunc read(const b:qbit) {}",
        )


def test_scalar_source_kinds_still_match_const_helper_formals() -> None:
    with pytest.raises(HybridVerificationError, match="qmod.signature"):
        parse(
            "within {flag=x<y;} apply { read(flag); }",
            declarations="flag:qnum;",
            helpers="qfunc read(const b:qbit) {}",
        )


def test_unused_helpers_obey_the_same_scope_contract() -> None:
    with pytest.raises(HybridVerificationError, match="qmod.scope"):
        parse(
            "",
            declarations="",
            helpers="qfunc unused(const x:qbit) { t:qbit; within {t=0;} apply {} }",
        )


@pytest.mark.parametrize("cap", [2, 4, 5])
def test_generated_local_and_cleanup_operations_share_the_existing_budget(
    monkeypatch: pytest.MonkeyPatch,
    cap: int,
) -> None:
    monkeypatch.setattr(context, "MAX_EFFECT_OPERATIONS", cap)
    with pytest.raises(HybridVerificationError, match="qmod.budget.expansion"):
        parse("within {flag=x<y;} apply {}")
