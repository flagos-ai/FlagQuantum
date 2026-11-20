"""Shared stored PLU calls match original construction and invert exactly."""

from typing import Any

import pytest

from flagquantum.compiler.registers.finite_field.linear_component import (
    binary_linear_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from tests.unit.test_field_multiplication import reference_gates

np = pytest.importorskip("numpy")

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


class SourceCalls:
    """Read original build calls directly before DAG export schedules them."""

    def __init__(self) -> None:
        self.gates: list[tuple[str, tuple[int, ...]]] = []

    def split(self, word: Any) -> Any:
        return word

    def join(self, word: Any, *, dtype: Any) -> Any:
        return word

    def add(self, bloq: Any, **ports: Any) -> Any:
        from qualtran.bloqs.basic_gates import CNOT
        from qualtran.bloqs.gf_arithmetic.gf2_multiplication import SynthesizeLRCircuit

        if isinstance(bloq, CNOT):
            self.gates.append(("cx", (ports["ctrl"], ports["target"])))
            return ports["ctrl"], ports["target"]
        assert isinstance(bloq, SynthesizeLRCircuit)
        return bloq.build_composite_bloq(self, **ports)["q"]


@pytest.mark.parametrize(
    "matrix", [((0, 1), (1, 1)), ((1, 1, 0), (0, 1, 1), (1, 1, 1))]
)
def test_shared_linear_source_and_exact_inverse(
    matrix: tuple[tuple[int, ...], ...],
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import SynthesizeLRCircuit

    width = len(matrix)
    inputs = tuple((i,) for i in range(width))
    forward = bind_register_program(binary_linear_program(matrix), inputs)
    gates, outputs = reference_gates(
        SynthesizeLRCircuit(np.array(matrix)), {"q": tuple(range(width))}, ()
    )
    reader = SourceCalls()
    SynthesizeLRCircuit(np.array(matrix)).build_composite_bloq(
        reader, q=np.array(range(width))
    )
    assert [(g.name, g.wires) for g in forward.circuit.instructions] == reader.gates
    assert forward.outputs == tuple((i,) for i in outputs["q"])
    inverse = bind_register_program(
        binary_linear_program(matrix, adjoint=True), forward.outputs
    )
    assert inverse.outputs == inputs
    assert inverse.circuit.instructions == tuple(reversed(forward.circuit.instructions))


@pytest.mark.parametrize("matrix", [(), ((1, 0),), ((0,),), ((2,),), ((True,),)])
def test_invalid_linear_maps(matrix: tuple[tuple[int, ...], ...]) -> None:
    with pytest.raises(ValueError):
        binary_linear_program(matrix)
