"""Native producers behind the unchanged Compiler/Runtime conformance consumer."""

import math
from pathlib import Path
from typing import Literal

import pytest

from flagquantum.compiler._construction.selection import (
    ConstructionBudget,
    select_register_program,
)
from flagquantum.compiler._construction.target import Target
from flagquantum.compiler.coupling import CouplingMap
from flagquantum.ecosystem._qmod.adapter import (
    SUPPORTED_SOURCE_VERSION,
    QmodImport,
    parse_qmod,
)
from tests.integration.test_construction_register_execution import (
    BASIS,
)
from tests.integration.test_construction_register_execution import (
    _check as check_model,
)
from tests.register_program_cases import tutorial_program

pytestmark = pytest.mark.integration
CORPUS = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"


def _parse(source: str, filename: str = "synthetic.qmod") -> QmodImport:
    return parse_qmod(
        source, filename=filename, source_version=SUPPORTED_SOURCE_VERSION
    )


def test_original_x_source_executes_with_its_declared_dimension() -> None:
    original = CORPUS / "X.qmod"
    imported = _parse(original.read_text(), str(original))
    selection = select_register_program(imported.program, imported.functions)
    assert len(selection.registers["q"]) == 1
    assert selection.selected.resources.n_qubits == 1
    check_model(selection, {(1,): 1})


@pytest.mark.parametrize("producer", ["native", "recorded"])
@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_native_and_recorded_frontends_share_the_same_numerical_consumer(
    producer: str,
    routed: bool,
    objective: Literal["width", "depth"],
) -> None:
    if producer == "native":
        imported = _parse(
            "qfunc main(output x:qnum<2>, output y:qnum) { allocate(x); hadamard_transform(x); y=x**2+1; }"
        )
        program, functions = imported.program, imported.functions
    else:
        program, functions = tutorial_program(2)
    target = Target(
        basis=BASIS if routed else None,
        couplings=tuple(CouplingMap.line(n) for n in range(1, 21)) if routed else (),
        optimize=not routed,
        restore_layout=not routed,
    )
    selection = select_register_program(
        program,
        functions,
        target=target,
        objective=objective,
        budget=ConstructionBudget(qubits=20),
    )
    if routed:
        assert any(
            op.name == "swap"
            for op in selection.selected.target.executable.instructions
        )
    check_model(selection, {(x, x * x + 1): 0.5 for x in range(4)})


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_ordered_native_phase_and_post_compute_mutation_are_coherent(
    objective: Literal["width", "depth"],
) -> None:
    source = """qfunc main(output flag:qbit, output n:qnum, output res:qnum) {
      allocate(flag);
      H(flag);
      S(flag);
      allocate(2,n);
      hadamard_transform(n);
      res=flag*n;
      X(flag);
    }"""
    imported = _parse(source)
    selection = select_register_program(
        imported.program, imported.functions, objective=objective
    )
    check_model(
        selection,
        {
            (1 - q, n, q * n): (1j if q else 1) / math.sqrt(8)
            for q in range(2)
            for n in range(4)
        },
    )


def test_asymmetric_mixed_width_outputs_preserve_numeric_significance() -> None:
    imported = _parse(
        "qfunc main(output small:qnum<3>, output bit:qbit, output sum:qnum) { small=3; bit=1; sum=small+bit; }"
    )
    selection = select_register_program(imported.program, imported.functions)
    assert {name: len(ports) for name, ports in selection.registers.items()} == {
        "small": 3,
        "bit": 1,
        "sum": 4,
    }
    check_model(selection, {(3, 1, 4): 1})


def test_fresh_assignment_preserves_a_wider_declared_destination() -> None:
    imported = _parse(
        "qfunc main(output x:qnum<2>, output y:qnum<6>) { allocate(x); hadamard_transform(x); y=x*x+1; }"
    )
    selection = select_register_program(imported.program, imported.functions)
    assert len(selection.registers["y"]) == 6
    check_model(selection, {(x, x * x + 1): 0.5 for x in range(4)})


def test_reduced_product_source_preserves_all_initialized_outputs() -> None:
    imported = _parse(
        "qfunc main(output a:qnum, output b:qnum, output res:qnum) { a=2; b=3; res=a*b; }"
    )
    selection = select_register_program(imported.program, imported.functions)
    check_model(selection, {(2, 3, 6): 1})


def test_folded_literal_expression_still_has_quantum_output_storage() -> None:
    imported = _parse("qfunc main(output q:qnum) { q=1+2*3; }")
    selection = select_register_program(imported.program)
    assert selection.selected.arithmetic.peak_workspace == 0
    check_model(selection, {(7,): 1})
