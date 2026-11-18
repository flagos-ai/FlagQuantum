"""Bind complete locked S/T atoms through the existing named emitter boundary."""

from collections import Counter
from dataclasses import replace
from hashlib import sha1
from pathlib import Path
from typing import Literal

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import PhaseCall
from flagquantum.compiler.registers.phase.phase_family import phase_family_ir
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.register_types import opaque_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.drawer.fixed_phase_symbols import fixed_phase_wire_symbol
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import register_score_data
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from flagquantum.simulation.unitary import get_unitary
from tests.unit.test_drawer_component_score import assert_source_geometry

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("kind", ("s", "t"))
def test_source_atoms_capture_replace_and_rebind(kind: Literal["s", "t"]) -> None:
    from qualtran.bloqs.basic_gates import SGate, TGate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    original = (SGate if kind == "s" else TGate)()
    captured = import_register_emitter(original)
    assert captured.layout == (("q", 1),)
    assert captured.requirements() == (0, 0) and captured.binding_swaps == (0, 0)
    restored = replace(
        captured,
        snapshots=(
            CircuitIR.from_json(captured.snapshots[0].to_json()),
            CircuitIR.from_json(captured.snapshots[1].to_json()),
        ),
    )
    state = torch.tensor([1 + 2j, 3 - 1j, -2j, 4 + 1j], dtype=torch.complex128)
    state /= state.norm()
    for inverse in (False, True):
        source = original.adjoint() if inverse else original
        code = kind + "dg" if inverse else kind
        native = phase_family_ir(kind, 1, adjoint=inverse)
        assert restored.snapshots[int(inverse)].instructions == (
            Instruction(code, (0,)),
        )
        bound = restored({"q": (1,)}, (), (), inverse)
        assert bound == native
        expected = torch.as_tensor(source.tensor_contract())
        for instructions in (bound, native):
            actual = Circuit(2, inputs=state, dtype=torch.complex128).append(
                FieldOperation(2, instructions)
            )
            torch.testing.assert_close(
                actual.state().flatten(),
                (expected @ state.reshape(2, 2).T).T.flatten(),
                rtol=0,
                atol=2e-12,
            )
            torch.testing.assert_close(
                get_unitary(actual),
                torch.kron(torch.eye(2, dtype=torch.complex128), expected.contiguous()),
                rtol=0,
                atol=2e-12,
            )
            actual.append(FieldOperation(2, restored({"q": (1,)}, (), (), not inverse)))
            torch.testing.assert_close(
                actual.state().flatten(), state, rtol=0, atol=2e-12
            )
        assert Counter(g.name for g in bound) == {code: 1}
        assert query_qec_gate_cost(PhaseCall(kind, adjoint=inverse)).asdict() == (
            get_cost_value(source, QECGatesCost()).asdict()
        )
        assert get_cost_value(source, QubitCount()) == 1


@pytest.mark.parametrize("kind", ("s", "t"))
def test_source_files_factories_tensor_axes_and_complete_display(kind: str) -> None:
    from qualtran import BloqBuilder, QBit
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.basic_gates import s_gate, t_gate
    from qualtran.drawing import get_musical_score_data

    module = s_gate if kind == "s" else t_gate
    blobs = (
        (
            "6111677a1e55ccab022e988598366a0f1bdcfc94",
            "98ed87f00f6788bd20a54eeed8b3770e738f13a8",
        )
        if kind == "s"
        else (
            "77cc66bbed4f0fcf35d2565e338075890b0889f6",
            "3bfc107c313fc5757b4b715a493713852681e4c1",
        )
    )
    for suffix, blob in zip((".py", "_test.py"), blobs, strict=True):
        raw = (Path(module.__file__).parent / (kind + "_gate" + suffix)).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
    factory = getattr(module, "_" + kind + "_gate")
    doc = getattr(module, "_" + kind.upper() + "_GATE_DOC")
    original = factory.make()
    assert doc.bloq_cls is type(original) and doc.examples == (factory,)
    if kind == "s":
        assert doc.call_graph_example is None
    register = Register("q", opaque_register_type(1))
    for source in (original, original.adjoint()):
        assert source.adjoint().adjoint() == source
        assert source.wire_symbol(None).text == ""
        for source_reg in (source.signature[0], SourceRegister("other", QBit())):
            for index in ((), (0,), (2, 3)):
                assert source.wire_symbol(source_reg, index).text == str(source)
        (tensor,) = source.my_tensors({"q": "input"}, {"q": "output"})
        assert tensor.inds == (("output", 0), ("input", 0))
        assert tuple(tensor.tags) == (str(source),)
        torch.testing.assert_close(
            torch.as_tensor(tensor.data), torch.as_tensor(source.tensor_contract())
        )
        step = RegisterStep(
            str(source),
            ("q",),
            ("q",),
            input_registers=(register,),
            output_registers=(register,),
        )
        score = layout_register_score({"q": 1}, ("q",), (step,))
        builder, inputs = BloqBuilder.from_signature(source.signature)
        reference = builder.finalize(**builder.add_d(source, **inputs))
        assert_source_geometry(
            register_score_data(score, register_wire_symbol=fixed_phase_wire_symbol),
            get_musical_score_data(reference),
            ordered_ports=True,
        )
