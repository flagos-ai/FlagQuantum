"""Reuse complete predicate witnesses for canonical integer storage controls."""

from itertools import product

import pytest

from tests.unit.test_component_control_extension import (
    test_original_extension_graph_gates_all_branches_and_resources as extension_case,
)
from tests.unit.test_component_structured_control import (
    SINGLE,
)
from tests.unit.test_component_structured_control import (
    test_directional_predicate_ports_match_original as predicate_case,
)
from tests.unit.test_component_structured_control import (
    test_exact_adjoint_body_restores_all_forward_inverse_histories as adjoint_case,
)
from tests.unit.test_component_structured_control import (
    test_source_graph_gates_all_histories_phases_and_private_work as controlled_case,
)
from tests.unit.test_register_control_predicate import CASES, ControlCase

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

STORAGE_CASES: tuple[ControlCase, ...] = (
    (("QIntOnesComp", 2, (), (-1,), "ones_complement"),),
    (("QIntOnesComp", 3, (), (2,), "ones_complement"),),
    (("BQUInt", 2, (), (1,), "unsigned"),),
    (("QMontgomeryUInt", 3, (), (3,), "unsigned"),),
    *CASES[4:],
)


@pytest.mark.parametrize("case", STORAGE_CASES)
@pytest.mark.parametrize("expanded", [False, True])
def test_typed_integer_controls_match_full_source_execution(
    case: ControlCase, expanded: bool
) -> None:
    controlled_case(case, "spectator", expanded)


@pytest.mark.parametrize("case", STORAGE_CASES)
def test_typed_integer_predicate_directional_ports(case: ControlCase) -> None:
    predicate_case(case)


@pytest.mark.parametrize("case", STORAGE_CASES[:4])
@pytest.mark.parametrize("expanded", [False, True])
def test_integer_control_exact_adjoint_restores_all_histories(
    case: ControlCase, expanded: bool
) -> None:
    adjoint_case(case, expanded)


@pytest.mark.parametrize("case,new", tuple(product(STORAGE_CASES[:4], SINGLE[:2])))
@pytest.mark.parametrize("expanded", [False, True])
def test_typed_integer_control_extension_preserves_all_branches(
    case: ControlCase, new: ControlCase, expanded: bool
) -> None:
    extension_case(case, new, "spectator", expanded)
