"""Full logical graph order retains symbolic metadata with concrete widths."""

from typing import Any

import pytest

from flagquantum.compiler._component_trace.width_schedule import (
    WidthStep,
    schedule_width_steps,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_schedule import local_operation_order
from flagquantum.core.ir.program import FunctionRecord
from flagquantum.core.ir.program.register_types import fixed_point_register_type
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("qualtran")
sympy = pytest.importorskip("sympy")


@pytest.mark.parametrize("symbolic", [False, True])
def test_declared_symbolic_fraction_changes_complete_graph_priority(
    symbolic: bool,
) -> None:
    from attrs import frozen
    from qualtran import Bloq, BloqBuilder, QAny, QFxp, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran._infra.binst_graph_iterators import greedy_topological_sort

    fraction = sympy.Symbol("f", integer=True, positive=True) if symbolic else 4

    @frozen
    class Expand(Bloq):  # type: ignore[misc]  # Untyped frozen reference SDK.
        @property
        def signature(self) -> Any:
            return SourceSignature(
                (
                    SourceRegister("a", QFxp(8, fraction)),
                    SourceRegister("b", QFxp(8, fraction), side=Side.RIGHT),
                )
            )

    @frozen
    class Neutral(Bloq):  # type: ignore[misc]  # Untyped frozen reference SDK.
        @property
        def signature(self) -> Any:
            return SourceSignature((SourceRegister("c", QAny(8)),))

    source_signature = SourceSignature(
        (
            SourceRegister("a", QFxp(8, fraction)),
            SourceRegister("b", QFxp(8, fraction), side=Side.RIGHT),
            SourceRegister("c", QAny(8)),
        )
    )
    source, inputs = BloqBuilder.from_signature(source_signature)
    expanded = source.add_d(Expand(), a=inputs["a"])
    neutral = source.add_d(Neutral(), c=inputs["c"])
    composite = source.finalize(**expanded, **neutral)
    expected = tuple(
        type(node.bloq).__name__.lower()
        for node in greedy_topological_sort(composite._binst_graph)
        if hasattr(node, "bloq")
    )

    dtype = fixed_point_register_type(8, fraction)
    expand_registers = (
        Register("a", dtype),
        Register("b", dtype, side=RegisterSide.RIGHT),
    )
    neutral_registers = (Register("c", RegisterType("opaque", 8)),)
    signature = Signature((*expand_registers, *neutral_registers))
    functions = (
        FunctionRecord("expand", (), registers=expand_registers),
        FunctionRecord("neutral", (), registers=neutral_registers),
    )
    builder = ComponentProgramBuilder(signature, functions)
    outputs = builder.call_dict("expand", {"a": builder.inputs["a"]})
    outputs.update(builder.call_dict("neutral", {"c": builder.inputs["c"]}))
    program = builder.finalize(outputs)
    main = program.functions[0]
    order = local_operation_order(
        main,
        {function.name: function for function in program.functions},
        logical_graph=True,
    )
    assert tuple(main.blocks[0].operations[i].callee for i in order) == expected
    assert expected == (("expand", "neutral") if symbolic else ("neutral", "expand"))


def test_symbolic_signature_flag_rejects_non_boolean_values() -> None:
    from dataclasses import replace

    invalid: Any = 1
    step = replace(WidthStep("component"), symbolic_signature=invalid)
    with pytest.raises(ValueError, match="built-in boolean"):
        schedule_width_steps((step,), ())


def test_source_and_native_refuse_symbolic_element_counts() -> None:
    from qualtran import BloqBuilder, DecomposeTypeError, QAny
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature

    n = sympy.Symbol("n", integer=True, positive=True)
    signature = SourceSignature((SourceRegister("q", QAny(n), shape=(n,)),))
    with pytest.raises(DecomposeTypeError, match="symbolic shape"):
        BloqBuilder.from_signature(signature)
    with pytest.raises(ValueError, match="array dimensions"):
        ComponentProgramBuilder(
            Signature((Register("q", RegisterType("opaque", n), (n,)),))
        )
