"""Retain literal generic THC controls across supported logical declarations."""

from importlib import import_module
from typing import Any

import pytest
import sympy

from tests._reference_dependencies import unavailable_reference

try:
    import_module("numpy")
    import_module("qualtran")
    import_module("kickmix")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

from flagquantum.compiler.formats.legacy_thc_selection_factories import _selection
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.ecosystem._default_linear_combination_oracles import _signature
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_thc_selection_source_constructors import (
    source_module as source_module,
)
from tests.unit.test_thc_walk_source_context import assert_calls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("inverse", (False, True))
@pytest.mark.parametrize("control", (None, 0, 1))
@pytest.mark.parametrize(
    "spin,angle,keep",
    (
        (4, 8, 8),
        (3, 8, 8),
        (4, 0, 0),
        (4, 1, 0),
        (4, sympy.Symbol("b", integer=True, positive=True), 8),
        (sympy.Symbol("n", integer=True, positive=True), 8, 8),
    ),
)
def test_generic_source_calls_and_declaration_dependencies(
    source_module: Any,
    inverse: bool,
    control: int | None,
    spin: int | sympy.Expr,
    angle: int | sympy.Expr,
    keep: int,
) -> None:
    from qualtran import Controlled, CtrlSpec

    args = dict(
        num_mu=2,
        num_spin_orb=spin,
        num_bits_theta=angle,
        keep_bitsize=keep,
        control_val=control,
    )
    original = source_module.SelectTHC(**args)
    original = Controlled(original.adjoint() if inverse else original, CtrlSpec())
    record = _selection("main", None, args, adjoint=inverse, generic_control=True)
    assert isinstance(record, ProgramRecord)
    assert not verify_program_record(record)
    assert_calls(record, original)
    compare_layers(record, original, atomic_leaves(record))
    opaque = [f for f in record.functions if "positive" in f.name and not f.blocks]
    assert len(opaque) >= 4


@pytest.mark.parametrize("mu,spin", ((0, 4), (2, 0), (2, 1)))
def test_original_failed_zero_word_body_retains_controlled_header(
    source_module: Any, mu: int, spin: int
) -> None:
    from qualtran import BloqError, Controlled, CtrlSpec

    args = dict(num_mu=mu, num_spin_orb=spin, num_bits_theta=8, keep_bitsize=8)
    original = Controlled(source_module.SelectTHC(**args), CtrlSpec())
    with pytest.raises(BloqError, match="does not accept Soquets"):
        original.decompose_bloq()
    declaration = _selection("main", None, args, generic_control=True)
    assert isinstance(declaration, FunctionRecord)
    assert not declaration.blocks and declaration.registers == _signature(
        original.signature
    )
    assert not verify_program_record(ProgramRecord((declaration,)))
