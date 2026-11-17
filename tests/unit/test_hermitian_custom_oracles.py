"""Replace both sparse oracles without changing the Hermitian consumer."""

from hashlib import sha1
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.custom_sparse_oracles import (
    ConjunctionSqrtEntry,
    custom_hermitian_block,
)
from flagquantum import Circuit
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.core.ir import CircuitIR, Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    assert_source_wire_streams,
    source_partition_leaf,
    walk_source_graph,
)
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_uniform_preparation import reference_uniform_preparation_gates

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_oracles(width: int, base: float = 0.37, delta: float = 0.62) -> Any:
    """User extensions of the original abstract oracles; upstream stays immutable."""
    from attrs import frozen
    from qualtran import QAny, QUInt
    from qualtran.bloqs.basic_gates import CNOT, Ry, XGate
    from qualtran.bloqs.block_encoding.sparse_matrix import RowColumnOracle
    from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import (
        SparseMatrixHermitian,
        SqrtEntryOracle,
    )
    from qualtran.bloqs.mcmt import And

    @frozen
    class Column(RowColumnOracle):
        n: int

        @property
        def system_bitsize(self) -> int:
            return self.n

        @property
        def num_nonzero(self) -> int:
            return 2**self.n

        def build_composite_bloq(self, bb: Any, **soqs: Any) -> dict[str, Any]:
            left, index = bb.split(soqs["l"]), bb.split(soqs["i"])
            for k in range(self.n):
                index[k], left[k] = bb.add(CNOT(), ctrl=index[k], target=left[k])
                left[k] = bb.add(XGate(), q=left[k])
            return {
                "l": bb.join(left, dtype=QUInt(self.n)),
                "i": bb.join(index, dtype=QUInt(self.n)),
            }

    @frozen
    class Entry(SqrtEntryOracle):
        n: int
        base: float
        delta: float

        @property
        def system_bitsize(self) -> int:
            return self.n

        @property
        def epsilon(self) -> float:
            return 0.0

        def build_composite_bloq(
            self, bb: Any, q: Any, i: Any, j: Any
        ) -> dict[str, Any]:
            rows, columns = bb.split(i), bb.split(j)
            q = bb.add(Ry(self.base), q=q)
            (rows[-1], columns[-1]), predicate = bb.add(
                And(), ctrl=np.asarray([rows[-1], columns[-1]])
            )
            q = bb.add(Ry(self.delta / 2), q=q)
            predicate, q = bb.add(CNOT(), ctrl=predicate, target=q)
            q = bb.add(Ry(-self.delta / 2), q=q)
            predicate, q = bb.add(CNOT(), ctrl=predicate, target=q)
            rows[-1], columns[-1] = bb.add(
                And().adjoint(),
                ctrl=np.asarray([rows[-1], columns[-1]]),
                target=predicate,
            )
            return {
                "q": q,
                "i": bb.join(rows, dtype=QAny(self.n)),
                "j": bb.join(columns, dtype=QAny(self.n)),
            }

    return SparseMatrixHermitian(Column(width), Entry(width, base, delta), 0)


@pytest.mark.parametrize(
    "base,delta", ((0.0, 0.0), (0.0, np.pi / 2), (np.pi / 2, 0.62))
)
def test_user_entry_cost_classification_matches_original_leaves(
    base: float, delta: float
) -> None:
    source = source_oracles(1, base, delta).entry_oracle
    native = ConjunctionSqrtEntry(1, base, delta)
    for adjoint in (False, True):
        assert_source_model(
            native.resource_model(adjoint=adjoint),
            source.adjoint() if adjoint else source,
        )


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.block_encoding import sparse_matrix_hermitian

    raw = Path(sparse_matrix_hermitian.__file__).read_bytes()
    assert (
        sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
        == "95b077d8eb6ac7deb814c910276b4a1d9d603bee"
    )


def locked_stream(
    source: Any, ports: SourcePorts, scratch: int, slot: int
) -> tuple[Instruction, ...]:
    from qualtran import Adjoint
    from qualtran.bloqs.basic_gates import CNOT, Ry, Swap, XGate
    from qualtran.bloqs.basic_gates.swap import CSwap
    from qualtran.bloqs.mcmt import And
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        partition = source_partition_leaf(bloq, inputs, desired)
        if partition is not None:
            return partition
        inverse = isinstance(bloq, Adjoint)
        base = bloq.subbloq if inverse else bloq
        if isinstance(base, PrepareUniformSuperposition):
            return reference_uniform_preparation_gates(
                base.n, (), inputs["target"], (), (), (), adjoint=inverse
            ), inputs
        if isinstance(bloq, And):
            complete = inputs | {"target": (scratch,)}
            outputs = {"ctrl": inputs["ctrl"]} if bloq.uncompute else complete
            return reference_arithmetic_gates(
                bloq, complete, (), {scratch: slot}, cirq_style=True
            ), outputs
        if isinstance(bloq, Ry):
            return (
                Instruction("ry", inputs["q"], {"theta": float(bloq.angle)}),
            ), inputs
        if isinstance(bloq, XGate):
            return (Instruction("x", inputs["q"]),), inputs
        if isinstance(bloq, CNOT):
            return (Instruction("cx", (*inputs["ctrl"], *inputs["target"])),), inputs
        if isinstance(bloq, CSwap):
            return reference_arithmetic_gates(
                bloq, inputs, (), {}, preserve_components=True
            ), inputs
        if isinstance(bloq, Swap):
            return tuple(
                Instruction("swap", (a, b))
                for a, b in zip(inputs["x"], inputs["y"], strict=True)
            ), inputs
        return None

    gates, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    return gates


@pytest.mark.parametrize("width", (1, 2))
@pytest.mark.parametrize("controlled", (False, True))
def test_user_oracles_source_stream_resources_and_all_branches(
    width: int, controlled: bool
) -> None:
    native = custom_hermitian_block(width)
    source = source_oracles(width)
    source = source.controlled() if controlled else source
    assert_source_model(native.resource_model(controlled=controlled), source)
    assert native.requirements() == (1, 1)
    shift = int(controlled)
    through = 2 * width + 2 + shift
    system = tuple(range(shift, shift + width))
    ancilla = tuple(range(shift + width, through))
    control = 0 if controlled else None
    ports = {"system": system, "ancilla": ancilla}
    if controlled:
        ports["ctrl"] = (0,)
    op = native.make_on(
        system=system,
        ancilla=ancilla,
        control=control,
        workspace=(through,),
        classical_bits=(1,),
    )
    circuit = Circuit(through + 3, dtype=torch.complex128).append(op)
    gates = circuit.to_ir().instructions
    assert_source_wire_streams(gates, locked_stream(source, ports, through, 1))
    counted = count_emitted_instructions(circuit.to_ir())
    assert counted.t_count == 8 + (7 * (width + 1) if controlled else 0)
    assert sum(g.name == "measure" for g in gates) == 2
    assert {g.metadata["classical_bit"] for g in gates if g.name == "measure"} == {1}

    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    dimension = 1 << width
    logical = matrix[-(1 << (2 * width + 2)) :, -(1 << (2 * width + 2)) :]
    projection = logical.reshape(dimension, 4 * dimension, dimension, 4 * dimension)[
        :, 0, :, 0
    ]
    entries = torch.tensor(
        [
            [
                np.cos((0.37 + 0.62 * ((row & column) & 1)) / 2) ** 2 / dimension
                for column in range(dimension)
            ]
            for row in range(dimension)
        ],
        dtype=torch.complex128,
    )
    torch.testing.assert_close(projection, entries, atol=3e-12, rtol=0)
    torch.testing.assert_close(matrix, matrix.mH, atol=3e-12, rtol=0)
    values = torch.arange(1 << (through + 1), dtype=torch.float64)
    state = torch.complex(values + 1, values % 7 - 3).reshape(1 << through, 2)
    state /= state.norm()

    def extend(value: torch.Tensor) -> torch.Tensor:
        # Logical source ports, clean work, entangled spectator, measured |1>.
        result = torch.zeros((1 << through, 2, 2, 2), dtype=torch.complex128)
        result[:, 0, :, 1] = value
        return result.flatten()

    prefix = Instruction(
        "measure", (through + 2,), metadata={"is_dynamic": True, "classical_bit": 0}
    )
    suffix = Instruction("x", (through + 1,), metadata={"conditions": ((0, 1),)})
    initial = extend(state)
    expected = extend((matrix @ state).flip(1))
    for outcomes in product((0, 1), repeat=2):
        torch.testing.assert_close(
            forced_branch(initial, (prefix, *gates, suffix), (1, *outcomes)),
            expected / 2,
            atol=5e-12,
            rtol=0,
        )

    inverse = native.make_on(
        system=system,
        ancilla=ancilla,
        control=control,
        workspace=(through,),
        classical_bits=(1,),
        adjoint=True,
    )
    repeated = CircuitIR(through + 3, (*gates, *inverse.instructions)).validate()
    for outcomes in product((0, 1), repeat=4):
        torch.testing.assert_close(
            forced_branch(initial, repeated.instructions, outcomes),
            initial / 4,
            atol=6e-12,
            rtol=0,
        )
