"""Measurement phase lookup is scoped to this call's declared output ports."""

import numpy as np
import pytest

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.classical_measurement import MeasurementOutcomePhase
from flagquantum.simulation.component_classical_state import ComponentClassicalState
from tests.unit.test_classical_action_tensor import signature_from_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("shape", [(), (2,), (1, 2)])
@pytest.mark.parametrize("lookup", ["valid", "missing", "bad_index", "wrong_rank"])
def test_phase_output_lookup_after_assignment_matches_source(
    shape: tuple[int, ...], lookup: str
) -> None:
    from qualtran import Bloq, QBit, Register
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.classical_sim import (
        MeasurementPhase,
        PhasedClassicalSimState,
    )

    index = () if not shape else tuple(size - 1 for size in shape)
    name = "q"
    if lookup == "missing":
        name = "absent"
    elif lookup == "bad_index":
        index = (2,) if shape else (0,)
    elif lookup == "wrong_rank":
        index = (0, 0, 0)
    signature = SourceSignature([Register("q", QBit(), shape=shape)])
    cls = type(
        "PhaseLookup",
        (Bloq,),
        {
            "signature": property(lambda self: signature),
            "on_classical_vals": lambda self, **values: values,
            "basis_state_phase": lambda self, **values: MeasurementPhase(name, index),
        },
    )
    source = cls().as_composite_bloq()
    native_signature = signature_from_source(signature)
    leaf_builder = ComponentProgramBuilder(native_signature, name="phase_lookup")
    leaf = leaf_builder.finalize(leaf_builder.inputs).functions[0]
    builder = ComponentProgramBuilder(Signature(native_signature), (leaf,))
    record = builder.finalize(builder.call_dict(leaf.name, builder.inputs))
    value: object = np.ones(shape, dtype=np.uint8) if shape else 1
    original = PhasedClassicalSimState.from_cbloq(source, {"q": value})
    state = ComponentClassicalState(
        record,
        (0,),
        {"q": value},
        lambda function, values: values,
        lambda function, values: MeasurementOutcomePhase(name, index),
        phased=True,
    )
    if lookup == "valid":
        expected = original.simulate()
        actual = state.simulate()
        np.testing.assert_array_equal(actual["q"], expected["q"])
        assert state.phase == original.phase == -1
    else:
        with pytest.raises(ValueError, match="Could not find the requested soquet"):
            original.simulate()
        with pytest.raises(
            ValueError, match="measurement phase requires a current output"
        ):
            state.step()
        # Source writes outputs before attempting lookup. Native keeps that
        # visible prefix and leaves the accumulator unchanged on lookup failure.
        assert state.last_position == 0 and state.phase == original.phase == 1
        for result in record.functions[0].blocks[0].operations[0].results:
            assert state.assignments[result] == 1
        np.testing.assert_array_equal(state.finalize()["q"], value)
        assert any(soq.binst in source.bloq_instances for soq in original.soq_assign)


def test_phase_cannot_resolve_a_previous_calls_same_named_output() -> None:
    from qualtran import Bloq, BloqBuilder, QBit, Register, Side
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.classical_sim import (
        MeasurementPhase,
        PhasedClassicalSimState,
    )

    thru = SourceSignature([Register("q", QBit())])
    right_only_other = SourceSignature(
        [Register("q", QBit(), side=Side.LEFT), Register("r", QBit(), side=Side.RIGHT)]
    )
    first = type(
        "First",
        (Bloq,),
        {
            "signature": property(lambda self: thru),
            "on_classical_vals": lambda self, **values: values,
        },
    )()
    second = type(
        "Second",
        (Bloq,),
        {
            "signature": property(lambda self: right_only_other),
            "on_classical_vals": lambda self, **values: {"r": values["q"]},
            "basis_state_phase": lambda self, **values: MeasurementPhase("q"),
        },
    )()
    bb, inputs = BloqBuilder.from_signature(thru)
    q = bb.add(first, q=inputs["q"])
    r = bb.add(second, q=q)
    original = PhasedClassicalSimState.from_cbloq(bb.finalize(q=r), {"q": 1})
    # A typed rename is represented by ordinary return metadata, not a new op.
    from dataclasses import replace

    from flagquantum.core.ir.program.registers import Register as NativeRegister
    from flagquantum.core.ir.program.registers import RegisterSide

    sig = signature_from_source(thru)
    leaf_builder = ComponentProgramBuilder(sig, name="first")
    f = leaf_builder.finalize(leaf_builder.inputs).functions[0]
    g = replace(
        f,
        name="second",
        registers=(
            NativeRegister("q", sig.get_left("q").dtype, side=RegisterSide.LEFT),
            NativeRegister("r", sig.get_left("q").dtype, side=RegisterSide.RIGHT),
        ),
    )
    builder = ComponentProgramBuilder(sig, (f, g))
    qn = builder.call_dict("first", builder.inputs)["q"]
    rn = builder.call_dict("second", {"q": qn})["r"]
    record = builder.finalize({"q": rn})
    state = ComponentClassicalState(
        record,
        (0, 1),
        {"q": 1},
        lambda function, values: (
            values if function.name == "first" else {"r": values["q"]}
        ),
        lambda function, values: (
            None if function.name == "first" else MeasurementOutcomePhase("q")
        ),
        phased=True,
    )
    assert state.step() is state
    with pytest.raises(ValueError, match="measurement phase requires a current output"):
        state.step()
    with pytest.raises(ValueError, match="Could not find the requested soquet"):
        original.simulate()
    assert state.finalize() == {"q": 1} and state.phase == 1
