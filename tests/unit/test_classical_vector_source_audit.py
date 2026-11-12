"""Whole classical-vector source audit, including failure side effects."""

from typing import Any

import numpy as np
import pytest

from flagquantum.compiler.registers.component_basis_vectors import (
    integer_basis_function,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_classical_vectors import (
    _add_classical_bras,
    _add_classical_kets,
    flank_component_with_classical_vectors,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("shape", [(2,), (1, 2), (0,), (2, 0)])
@pytest.mark.parametrize("missing", [False, True])
def test_array_effect_refusal_precedes_definition_consumption_even_when_empty(
    shape: tuple[int, ...], missing: bool
) -> None:
    from qualtran import BloqBuilder, QBit
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.basic_gates import IntState
    from qualtran.simulation.xcheck_classical_quimb import (
        _add_classical_bras as source_bras,
    )

    source = BloqBuilder()
    word = source.add(IntState(0, 1))
    native = ComponentProgramBuilder(
        Signature([Register("val", RegisterType("bit", 1), side=RegisterSide.RIGHT)]),
        (integer_basis_function("zero", RegisterType("bit", 1), 0),),
    )
    binding = native.call_dict("zero", {})["val"]
    error = KeyError if missing else ValueError
    with pytest.raises(error):
        source_bras(
            source,
            [SourceRegister("x", QBit(), shape)],
            {"x": np.zeros(shape, dtype=int)},
            {} if missing else {"x": word},
        )
    with pytest.raises(error):
        _add_classical_bras(
            native,
            [Register("x", RegisterType("bit", 1), shape)],
            {"x": np.zeros(shape, dtype=int)},
            {} if missing else {"x": binding},
            "test_",
        )
    # The scalar is still live and no effect was added/defined. Source and
    # native both allow returning that very same word after the error.
    src = source.finalize(val=word)
    record = native.finalize({"val": binding})
    assert len(src.bloq_instances) == 1
    assert [f.name for f in record.functions] == ["main", "zero"]
    assert len(record.functions[0].blocks[0].operations) == 1


@pytest.mark.parametrize("direction", ["ket", "bra"])
@pytest.mark.parametrize("shape", [(2,), (1, 2)])
def test_short_value_array_preserves_successful_prefix(
    direction: str, shape: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran import BloqBuilder, QBit
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.basic_gates import IntEffect, IntState
    from qualtran.simulation.xcheck_classical_quimb import (
        _add_classical_bras as source_bras,
    )
    from qualtran.simulation.xcheck_classical_quimb import (
        _add_classical_kets as source_kets,
    )

    import flagquantum.compiler.registers.component_classical_vectors as module

    events: dict[str, list[tuple[int, bool]]] = {"source": [], "native": []}
    source_add = BloqBuilder.add

    def observe_source(self: Any, bloq: Any, **bindings: Any) -> Any:
        result = source_add(self, bloq, **bindings)
        if isinstance(bloq, (IntState, IntEffect)):
            events["source"].append((int(bloq.val), isinstance(bloq, IntState)))
        return result

    factory = integer_basis_function

    def observe_native(
        name: str, dtype: RegisterType, value: int, *, state: bool = True
    ) -> Any:
        result = factory(name, dtype, value, state=state)
        events["native"].append((value, state))
        return result

    source = BloqBuilder()
    native = ComponentProgramBuilder(Signature(()))
    source_words = np.empty(shape, dtype=object)
    native_words = np.empty(shape, dtype=object)
    if direction == "bra":
        for i, index in enumerate(np.ndindex(shape)):
            source_words[index] = source.add(IntState(0, 1))
            native.define(integer_basis_function(f"zero{i}", RegisterType("bit", 1), 0))
            native_words[index] = native.call_dict(f"zero{i}", {})["val"]
    short = np.zeros((1,) if shape == (2,) else (1, 1), dtype=int)
    sr = SourceRegister("x", QBit(), shape)
    nr = Register("x", RegisterType("bit", 1), shape)
    with monkeypatch.context() as patch:
        patch.setattr(BloqBuilder, "add", observe_source)
        patch.setattr(module, "integer_basis_function", observe_native)
        with pytest.raises(IndexError):
            if direction == "ket":
                source_kets(source, [sr], {"x": short})
            else:
                source_bras(source, [sr], {"x": short}, {"x": source_words})
        with pytest.raises(IndexError):
            if direction == "ket":
                _add_classical_kets(native, [nr], {"x": short}, "test_")
            else:
                _add_classical_bras(
                    native, [nr], {"x": short}, {"x": native_words}, "test_"
                )
    assert events["source"] == events["native"] == [(0, direction == "ket")]
    # Failure on the second value must not undo the first successful call.


def identity(shape: tuple[int, ...]) -> tuple[ProgramRecord, Any]:
    from qualtran import Bloq, QBit
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature

    native = ComponentProgramBuilder(
        Signature([Register("x", RegisterType("bit", 1), shape)])
    )
    record = native.finalize(native.inputs)
    source: Any = type(
        "VectorAuditIdentity",
        (Bloq,),
        {
            "signature": property(
                lambda self: SourceSignature([SourceRegister("x", QBit(), shape)])
            ),
            "on_classical_vals": lambda self, **values: values,
        },
    )()
    return record, source


@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,)])
@pytest.mark.parametrize("explicit", [False, True])
def test_complete_wrapper_closed_call_order_and_evaluator_identity(
    shape: tuple[int, ...], explicit: bool
) -> None:
    from qualtran.bloqs.basic_gates import IntEffect, IntState
    from qualtran.simulation.xcheck_classical_quimb import flank_with_classical_vectors

    from flagquantum.compiler.registers.component_basis_vectors import (
        integer_basis_function_data,
    )

    record, source = identity(shape)
    inputs: dict[str, object] = {
        "x": np.zeros(shape, dtype=int) if shape else np.int64(0)
    }
    seen: list[object] = []

    def evaluate(values: Any) -> Any:
        seen.append(values)
        return values

    actual = flank_component_with_classical_vectors(
        record, inputs, inputs if explicit else None, evaluate=evaluate
    )
    if shape == (0,) and not explicit:
        with pytest.raises(KeyError, match="RightDangle"):
            flank_with_classical_vectors(source, inputs)
        assert len(actual.functions[0].blocks[0].operations) == 1
        assert actual.functions[0].registers == ()
        return
    original = flank_with_classical_vectors(
        source, inputs, inputs if explicit else None
    )
    assert (not seen) if explicit else (len(seen) == 1 and seen[0] is inputs)
    definitions = {f.name: f for f in actual.functions}
    kinds = []
    for call in actual.functions[0].blocks[0].operations:
        function = definitions[call.callee or ""]
        value = integer_basis_function_data(function)
        kinds.append("child" if value is None else "state" if value[1] else "effect")
    source_kinds = [
        (
            "state"
            if isinstance(i.bloq, IntState)
            else "effect" if isinstance(i.bloq, IntEffect) else "child"
        )
        for i, _, _ in original.iter_bloqsoqs()
    ]
    assert kinds == source_kinds
    assert actual.functions[0].registers == ()


@pytest.mark.parametrize(
    "case", ["float", "symbolic_value", "numpy_bool", "zero_width", "symbolic_width"]
)
def test_source_constructor_acceptance_is_not_native_executable_admission(
    case: str,
) -> None:
    import sympy
    from qualtran import BloqBuilder, QUInt
    from qualtran import Register as SourceRegister
    from qualtran.simulation.xcheck_classical_quimb import (
        _add_classical_kets as source_kets,
    )

    width: Any = 1
    value: Any = 0
    if case == "float":
        value = 0.5
    elif case == "symbolic_value":
        value = sympy.Symbol("v", integer=True)
    elif case == "numpy_bool":
        value = np.bool_(True)
    elif case == "zero_width":
        width = 0
    else:
        width = sympy.Symbol("n", integer=True, positive=True)
    source = BloqBuilder()
    result = source_kets(source, [SourceRegister("x", QUInt(width))], {"x": value})
    assert set(result) == {"x"}
    # This checks constructor acceptance only: the upstream graph is neither
    # contracted nor claimed executable for these values or widths.
    native = ComponentProgramBuilder(Signature(()))
    with pytest.raises(ValueError):
        _add_classical_kets(
            native, [Register("x", RegisterType("uint", width))], {"x": value}, "test_"
        )
