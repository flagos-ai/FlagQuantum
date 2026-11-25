"""Check source tensor ordering, shaped ports and codec-only companion metadata."""

from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.register_verifier import executable_register_width
from tests.unit.multi_register_fixture import (
    companion_signatures,
    multi_register_fixture,
)
from tests.unit.register_source_graph import SourcePorts, SourceResult
from tests.unit.register_source_workspace import source_workspace_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_shaped_fixture_preserves_source_wires_and_full_complex_vector() -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.for_testing.atom import TestAtom, TestTwoBitOp
    from qualtran.bloqs.for_testing.many_registers import TestMultiRegister
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = TestMultiRegister()
    native = multi_register_fixture()
    funcs = {fn.name: fn for fn in native.functions}
    signature = funcs["main"].registers
    assert signature is not None
    assert [(r.name, r.bitsize, r.shape, r.side.name) for r in signature] == [
        (r.name, r.bitsize, r.shape, r.side.name) for r in source.signature
    ]
    call_names = {"atom": "TestAtom", "two_bit": "TestTwoBitOp"}
    kinds = []
    for op in funcs["main"].blocks[0].operations:
        assert op.callee is not None
        if op.callee in call_names:
            kinds.append(call_names[op.callee])
        else:
            register_ports = funcs[op.callee].registers
            assert register_ports is not None
            kinds.append("Join" if register_ports[0].shape else "Split")
    original = source.decompose_bloq()
    ordered = sorted(original.bloq_instances, key=lambda node: node.i)
    assert kinds == [type(node.bloq).__name__ for node in ordered]

    pairs = ((10, 0), (9, 1), (8, 2), (7, 3))
    x, z = (11,), (6, 4, 5)
    ports = {"xx": x, "yy": tuple(w for pair in pairs for w in pair), "zz": z}
    observed: list[tuple[torch.Tensor, tuple[int, ...]]] = []

    def atom(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, TestAtom):
            name, wires = "x", inputs["q"]
        elif isinstance(bloq, TestTwoBitOp):
            name, wires = "cx", (*inputs["ctrl"], *inputs["target"])
        else:
            return None
        matrix = torch.as_tensor(bloq.tensor_contract().copy(), dtype=torch.complex128)
        observed.append((matrix, wires))
        assert all(inputs[port] == ws for port, ws in desired.items())
        return (Instruction(name, wires),), inputs

    expected_gates, lifetimes = source_workspace_gates(
        source, ports, (), (), custom_leaf=atom
    )
    bound = bind_register_program(native, (x, *pairs, z))
    assert not lifetimes and not bound.events
    assert bound.outputs == (x, *pairs, z)
    assert_source(bound.circuit.instructions, expected_gates)
    assert [gate.name for gate in bound.circuit.instructions] == [
        "x",
        "cx",
        "cx",
        "cx",
        "cx",
    ]
    assert get_cost_value(source, QubitCount()) == 12
    with pytest.raises(DecomposeTypeError):
        get_cost_value(source, QECGatesCost())

    values = torch.arange(1, 4097, dtype=torch.float64)
    ket = torch.complex(values, 2 - values)
    ket /= ket.norm()
    expected = ket.reshape((2,) * 12)
    for matrix, operands in observed:
        leading = tuple(range(len(operands)))
        local = expected.movedim(operands, leading)
        expected = (matrix @ local.reshape(1 << len(operands), -1)).reshape(local.shape)
        expected = expected.movedim(leading, operands)
    actual = Circuit(12, inputs=ket, dtype=ket.dtype).append(
        FieldOperation(12, bound.circuit.instructions)
    )
    torch.testing.assert_close(
        actual.state().flatten(), expected.flatten(), atol=1e-14, rtol=0
    )


def test_typed_companions_metadata_and_original_invalid_binding() -> None:
    pytest.importorskip("qualtran")
    from qualtran import BQUInt, DecomposeNotImplementedError, QFxp
    from qualtran._infra.composite_bloq import BloqError
    from qualtran.bloqs.for_testing.many_registers import (
        TestBoundedQUInt,
        TestMultiTypedRegister,
        TestQFxp,
    )

    sources = (TestBoundedQUInt(), TestQFxp(), TestMultiTypedRegister())
    for native, source in zip(companion_signatures(), sources, strict=True):
        for ns, ss in zip(native, source.signature, strict=True):
            assert (ns.name, ns.bitsize, ns.shape, ns.side.name) == (
                ss.name,
                ss.bitsize,
                ss.shape,
                ss.side.name,
            )
            params = dict(ns.dtype.parameters)
            if isinstance(ss.dtype, BQUInt):
                assert params == {"bound": ss.dtype.iteration_length}
            elif isinstance(ss.dtype, QFxp):
                assert params == {
                    "fractional_width": ss.dtype.num_frac,
                    "signed": ss.dtype.signed,
                }
            else:
                assert params == {} and executable_register_width(ns.dtype) == 8
                continue
            assert executable_register_width(ns.dtype) == ss.dtype.num_qubits
    for atom in sources[:2]:
        with pytest.raises(DecomposeNotImplementedError):
            atom.decompose_bloq()
    with pytest.raises(BloqError, match="register dtypes are not consistent"):
        sources[-1].decompose_bloq()
