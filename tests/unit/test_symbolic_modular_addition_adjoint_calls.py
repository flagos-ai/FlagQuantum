"""Symbolic modular adjoints reverse the available source root literally."""

import sys
from importlib import import_module
from typing import Any
from unittest.mock import patch

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    import_module("numpy")
    import sympy

    import_module("cirq")
    import_module("kickmix")
    from qualtran import (
        DecomposeNotImplementedError,
        DecomposeTypeError,
        QMontgomeryUInt,
        QUInt,
    )
    from qualtran.bloqs.mod_arithmetic.mod_addition import CModAdd, ModAdd
except ModuleNotFoundError as error:
    unavailable_reference(error)

from flagquantum.compiler.formats.legacy_modular_addition_factories import (
    _symbolic_addition,
)
from flagquantum.compiler.registers.arithmetic.addition_component import (
    addition_component_program,
)
from flagquantum.compiler.registers.arithmetic.controlled_addition_component import (
    controlled_addition_program,
)
from flagquantum.compiler.registers.comparison.controlled_component import (
    controlled_greater_program,
)
from flagquantum.compiler.registers.comparison.linear_component import (
    linear_greater_program,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.core.ir.program.register_types import (
    montgomery_register_type,
    unsigned_register_type,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_ecc_addition_model import frozen_source as frozen_source
from tests.unit.test_symbolic_modular_addition_calls import atomic_views
from tests.unit.test_symbolic_modular_doubling_calls import assert_unbound

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (3, 5))
@pytest.mark.parametrize("controlled,cv", ((False, 1), (True, 0), (True, 1)))
def test_exact_source_inverse_parent_and_complete_shared_children(
    width: int, controlled: bool, cv: int
) -> None:
    p = sympy.Symbol("p")
    dtype = (
        unsigned_register_type(width)
        if controlled
        else montgomery_register_type(width, p)
    )
    word = (
        unsigned_register_type(width) if controlled else montgomery_register_type(width)
    )
    wide = (
        unsigned_register_type(width + 1)
        if controlled
        else montgomery_register_type(width + 1)
    )
    source: Any = CModAdd(QUInt(width), p, cv) if controlled else ModAdd(width, p)
    source = source.adjoint()
    record = _symbolic_addition(
        dtype, p, controlled=controlled, control_value=cv, adjoint=True
    )
    assert len(record.functions[0].blocks[0].operations) == (14 if controlled else 17)
    terminals = atomic_views(record)
    templates = (
        (
            controlled_addition_program(word, wide, control_value=cv, adjoint=True)
            if controlled
            else addition_component_program(
                width + 1, left_dtype=wide, right_dtype=wide, adjoint=True
            )
        ),
        (
            controlled_greater_program(word, control_value=cv, adjoint=True)
            if controlled
            else linear_greater_program(width, adjoint=True)
        ),
    )
    reused: dict[str, Any] = {}
    from qualtran.bloqs.arithmetic.addition import Add
    from qualtran.bloqs.arithmetic.comparison import (
        CLinearDepthGreaterThan,
        LinearDepthGreaterThan,
    )
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd

    expected = (
        (
            CAdd(QUInt(width), QUInt(width + 1), cv).adjoint()
            if controlled
            else Add(QMontgomeryUInt(width + 1), QMontgomeryUInt(width + 1)).adjoint()
        ),
        (
            CLinearDepthGreaterThan(QUInt(width), cv).adjoint()
            if controlled
            else LinearDepthGreaterThan(width).adjoint()
        ),
    )
    for template, owner in zip(templates, expected, strict=True):
        for function in record.functions:
            if function.registers == template.functions[
                0
            ].registers and matches_component_definition(
                record, function.name, template
            ):
                terminals.add(function.name)
                reused[function.name] = owner
    pairs = compare_layers(record, source, terminals)
    functions = {f.name: f for f in record.functions}
    refused = 0
    for name, child in pairs.items():
        if name not in terminals:
            continue
        if name in reused:
            assert child == reused[name]
            continue
        with pytest.raises((DecomposeTypeError, DecomposeNotImplementedError)):
            child.decompose_bloq()
        if not functions[name].blocks:
            refused += 1
            while hasattr(child, "subbloq"):
                child = child.subbloq
            assert type(child).__name__ == "AddK" and child.k in (-p, p)
    assert refused == 2
    assert_unbound(record)
    denied = {
        name: None
        for name in sys.modules
        if name.split(".", 1)[0] in {"qualtran", "cirq", "kickmix"}
    }
    with patch.dict(sys.modules, denied):
        assert (
            _symbolic_addition(
                dtype, p, controlled=controlled, control_value=cv, adjoint=True
            )
            == record
        )
