"""Import complete live constructor dependencies without leaf-local ID aliases."""

import builtins
from collections import Counter
from hashlib import sha1
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_graph import import_register_graph
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def live_matrix_graph(polarity: int, nested: bool) -> Any:
    from qualtran import BloqBuilder, CtrlSpec, Signature
    from qualtran._infra.controlled import Controlled
    from qualtran.bloqs.for_testing.matrix_gate import MatrixGate

    builder, words = BloqBuilder.from_signature(
        Signature.build(ctrl=1, q=1, **({"ctrl2": 1} if nested else {}))
    )
    for matrix in (((0, 1), (1, 0)), ((1, 0), (0, 1j))):
        child = Controlled(MatrixGate(1, matrix), CtrlSpec(cvs=polarity))
        if nested:
            child = Controlled(child, CtrlSpec())
        names = tuple(reg.name for reg in child.signature)
        ordered_words: tuple[Any, ...] = (words["ctrl"], words["q"])
        if nested:
            ordered_words = (words["ctrl2"], *ordered_words)
        result = builder.add_d(child, **dict(zip(names, ordered_words, strict=True)))
        words["ctrl"], words["q"] = result[names[-2]], result[names[-1]]
        if nested:
            words["ctrl2"] = result[names[0]]
    return builder.finalize(**words)


@pytest.mark.parametrize("polarity", (0, 1))
@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize("adjoint", (False, True))
def test_child_identity_and_phase_survive_owned_graph_conversion(
    polarity: int, nested: bool, adjoint: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = live_matrix_graph(polarity, nested)
    if nested and polarity == 0 and adjoint:
        from qualtran import BloqError

        with pytest.raises(BloqError, match="Soquet named `ctrl1`"):
            graph.adjoint()
        return
    source = graph.adjoint() if adjoint else graph
    program = import_register_graph(source)
    inputs = ((2,), (0,), (3,)) if nested else ((2,), (0,))
    from qualtran.serialization import bloq as serializer

    raw = Path(serializer.__file__).read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "132930b2804536155f5b242dbede88518fcca459"
    )
    source_matrix = torch.tensor(source.tensor_contract(), dtype=torch.complex128)
    physical = tuple(wire for word in inputs for wire in word)
    mask = sum(1 << (3 - wire) for wire in physical)
    oracle = torch.zeros((16, 16), dtype=torch.complex128)
    for column in range(16):
        logical_in = sum(
            ((column >> (3 - wire)) & 1) << (len(physical) - 1 - index)
            for index, wire in enumerate(physical)
        )
        for logical_out in range(1 << len(physical)):
            row = (column & ~mask) | sum(
                ((logical_out >> (len(physical) - 1 - index)) & 1) << (3 - wire)
                for index, wire in enumerate(physical)
            )
            oracle[row, column] = source_matrix[logical_out, logical_in]
    source_import = builtins.__import__

    def owned_import(name: str, *args: Any, **kwargs: Any) -> Any:
        assert name.split(".")[0] not in {"qualtran", "cirq", "kickmix"}
        return source_import(name, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", owned_import)
        bound = bind_register_program(program, inputs)
        assert bound.outputs == inputs and not bound.events
        circuit = Circuit(4)
        circuit.append(FieldOperation(4, bound.circuit.instructions))
        ir = circuit.to_ir()
        assert CircuitIR.from_json(ir.to_json()).to_json() == ir.to_json()
        counts = Counter(g.name for g in ir.instructions)
        assert counts == {"any": 2, **({"x": 4} if adjoint and polarity == 0 else {})}
        coherent = torch.tensor(
            [complex(i + 1, 3 - i) for i in range(16)], dtype=torch.complex128
        )
        coherent /= coherent.norm()
        states = [*torch.eye(16, dtype=torch.complex128), coherent]
        for state in states:
            torch.testing.assert_close(
                forced_branch(state, ir.instructions, ()),
                oracle @ state,
                atol=2e-12,
                rtol=0,
            )
