"""Four half relations preserve their distinct retained-difference contracts."""

from typing import Literal

import pytest

from flagquantum.compiler.registers.half_ordering import (
    half_compare_ir,
    half_compare_outputs,
)
from tests.unit.test_half_comparison import (
    assert_half_source,
    check_half_amplitudes,
    layout,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("dtype", ["unsigned", "signed", "montgomery"])
@pytest.mark.parametrize("relation", ["gt", "ge", "lt", "le"])
def test_half_relation_source_stream_output_binding_and_model(
    n: int, dtype: str, relation: Literal["gt", "ge", "lt", "le"]
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.comparison import (
        LinearDepthHalfGreaterThan,
        LinearDepthHalfGreaterThanEqual,
        LinearDepthHalfLessThan,
        LinearDepthHalfLessThanEqual,
    )

    cls = {"unsigned": QUInt, "signed": QInt, "montgomery": QMontgomeryUInt}[dtype]
    source_cls = {
        "gt": LinearDepthHalfGreaterThan,
        "ge": LinearDepthHalfGreaterThanEqual,
        "lt": LinearDepthHalfLessThan,
        "le": LinearDepthHalfLessThanEqual,
    }[relation]
    a, b, target, difference, signs, bits = layout(n)
    signed = dtype == "signed"
    compute = half_compare_ir(
        a, b, difference, target[0], signs, relation=relation, signed=signed
    )
    erase = half_compare_ir(
        a,
        b,
        difference,
        target[0],
        signs,
        relation=relation,
        signed=signed,
        adjoint=True,
        classical_bits=bits,
    )
    assert half_compare_outputs(a, b, difference, target[0], relation=relation) == {
        "a": a,
        "b": b,
        "c": difference,
        "target": target,
    }
    assert_half_source(
        n,
        signed,
        source_cls(cls(n)),
        compute,
        erase,
        flipped=relation in ("ge", "le"),
        erased_outputs=half_compare_outputs(
            a, b, difference, target[0], relation=relation, adjoint=True
        ),
    )


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("relation", ["gt", "ge", "lt", "le"])
def test_half_relations_full_complex_outputs_and_all_branches(
    n: int, signed: bool, relation: Literal["gt", "ge", "lt", "le"]
) -> None:
    a, b, target, difference, signs, bits = layout(n)
    compute = half_compare_ir(
        a, b, difference, target[0], signs, relation=relation, signed=signed
    )
    erase = half_compare_ir(
        a,
        b,
        difference,
        target[0],
        signs,
        relation=relation,
        signed=signed,
        adjoint=True,
        classical_bits=bits,
    )
    check_half_amplitudes(n, signed, relation, compute, erase)


def test_half_relation_validation() -> None:
    with pytest.raises(ValueError, match="relation"):
        half_compare_ir((0,), (1,), (3, 4), 2, (5, 6), relation="bad")


def test_user_append_half_comparison_example() -> None:
    from examples.algorithms.register_half_comparison import main

    main()


def test_output_binding_validation() -> None:
    with pytest.raises(ValueError, match="relation"):
        half_compare_outputs((0,), (1,), (3, 4), 2, relation="bad")
    with pytest.raises(ValueError, match="boolean"):
        half_compare_outputs((0,), (1,), (3, 4), 2, adjoint=1)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("relation", ["ge", "lt"])
@pytest.mark.parametrize("signed", [False, True])
def test_pinned_reversed_adjoint_port_alias_disagrees_with_classical_model(
    relation: str, signed: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QInt, QUInt
    from qualtran.bloqs.arithmetic.comparison import (
        LinearDepthHalfGreaterThanEqual,
        LinearDepthHalfLessThan,
    )

    cls = (
        LinearDepthHalfGreaterThanEqual if relation == "ge" else LinearDepthHalfLessThan
    )
    source = cls((QInt if signed else QUInt)(2))
    values = source.on_classical_vals(a=1, b=0)
    reverse = source.adjoint()
    assert reverse.on_classical_vals(**values) == {"a": 1, "b": 0}
    assert reverse.decompose_bloq().call_classically(**values) == (0, 1)
    # Native logical-port binding preserves the executed source's aliases,
    # rather than silently replacing them with the classical action's names.
    a, b, target, difference, signs, bits = layout(2)
    assert half_compare_outputs(
        a, b, difference, target[0], relation=relation, adjoint=True
    ) == {"a": b, "b": a}
