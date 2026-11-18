"""Close source Unitary external binding and owned serialization as whole files."""

from hashlib import sha1
from itertools import product
from pathlib import Path

import pytest
import torch

from flagquantum import Circuit
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._block_encoding import import_block_encoding
from flagquantum.simulation.unitary import get_unitary
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_foreign_block_encoding import original_models

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_unitary_files() -> None:
    from qualtran.bloqs.block_encoding import unitary

    folder = Path(unitary.__file__).parent
    for name, expected in (
        ("unitary.py", "bbcff63d78cbd58979bee3670597a778fdd4bdd2"),
        ("unitary_test.py", "4bead193a26eb24303fb96a15bd2233774924e0e"),
        ("unitary.ipynb", "cca05e99e365416bada5a7785a43ac1ba3d57555"),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == expected


@pytest.mark.parametrize("overrides", (False, True))
def test_actual_original_factories_four_way_snapshot_and_owned_ir_roundtrip(
    overrides: bool,
) -> None:
    from qualtran.bloqs.block_encoding.unitary import (
        _unitary_block_encoding,
        _unitary_block_encoding_properties,
    )

    source = (
        _unitary_block_encoding_properties if overrides else _unitary_block_encoding
    )()
    block = import_block_encoding(source, models=original_models(source))
    assert (block.alpha, block.epsilon) == (source.alpha, source.epsilon)
    assert [(r.name, r.bitsize) for r in block.signature] == [
        (r.name, r.bitsize) for r in source.signature
    ]
    assert source.build_call_graph(None) == {source.U: 1}
    assert block.binding_swaps == (0, 0, 0, 0)
    for controlled, adjoint in product((False, True), repeat=2):
        root = source.controlled() if controlled else source
        if adjoint:
            root = root.adjoint()
        offset = int(controlled)
        ancillary = tuple(range(offset + 1, offset + 1 + block.ancilla_bitsize))
        resource = tuple(
            range(
                offset + 1 + len(ancillary),
                offset + 1 + len(ancillary) + block.resource_bitsize,
            )
        )
        n = offset + 1 + len(ancillary) + len(resource)
        operation = block.make_on(
            system=(offset,),
            ancilla=ancillary,
            resource=resource,
            control=(0,) if controlled else (),
            adjoint=adjoint,
        )
        ir = Circuit(n).append(operation).to_ir()
        assert len(ir.instructions) == 1
        gate = ir.instructions[0]
        assert gate.name == ("cphase" if controlled else "tdg" if adjoint else "t")
        assert gate.wires == ((0, 1) if controlled else (0,))
        assert_source_model(
            block.resource_model(controlled=controlled, adjoint=adjoint), root
        )
        restored = CircuitIR.from_json(ir.to_json())
        expected = torch.as_tensor(root.tensor_contract(), dtype=torch.complex128)
        for actual in (ir, restored):
            torch.testing.assert_close(
                get_unitary(actual, dtype=torch.complex128),
                expected,
                atol=2e-12,
                rtol=0,
            )


@pytest.mark.parametrize("adjoint", (False, True))
def test_external_measured_arithmetic_child_preserves_all_branches_and_work(
    adjoint: bool,
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.block_encoding import Unitary

    source = Unitary(Add(QUInt(2), QUInt(2)), ancilla_bitsize=1, resource_bitsize=1)
    controlled = False
    block = import_block_encoding(source, include_controlled=False)
    root = source.controlled() if controlled else source
    if adjoint:
        root = root.adjoint()
    offset = int(controlled)
    system = tuple(range(offset, offset + 4))
    ancillary, resource = (offset + 4,), (offset + 5,)
    logical = offset + 6
    work, slots = block.requirements(controlled=controlled)
    workspace = tuple(range(logical, logical + work))
    operation = block.make_on(
        system=system,
        ancilla=ancillary,
        resource=resource,
        control=(0,) if controlled else (),
        workspace=workspace,
        classical_bits=tuple(range(slots)),
        adjoint=adjoint,
    )
    ir = Circuit(logical + work).append(operation).to_ir()
    restored = CircuitIR.from_json(ir.to_json())
    expected_matrix = torch.as_tensor(root.tensor_contract(), dtype=torch.complex128)
    values = torch.arange(1 << logical, dtype=torch.float64)
    state = torch.complex(values + 1, values % 7 - 3)
    state /= state.norm()
    clean = torch.zeros(1 << work, dtype=torch.complex128)
    clean[0] = 1
    expected = torch.kron(expected_matrix @ state, clean)
    from collections import Counter

    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.compiler.registers.arithmetic.arithmetic_resource_model import (
        source_add_cost,
    )
    from tests.unit.test_unitary_block_encoding import _fixture

    declared, *_ = _fixture("add2", 1, 1)
    from dataclasses import replace

    price = declared.resource_model(source_add_cost(2, 2))
    assert_source_model(price, root)
    assert block.binding_swaps[:2] == (0, 0)
    priced = replace(block, models=(price, price, None, None))
    assert_source_model(priced.resource_model(adjoint=adjoint), root)
    source_cost = get_cost_value(root, QECGatesCost())
    counts = Counter(g.name for g in ir.instructions)
    assert counts["t"] + counts["tdg"] == 4 * source_cost.and_bloq
    assert counts["measure"] == source_cost.measurement
    events = counts["measure"]
    assert events <= 8
    for history in product((0, 1), repeat=events):
        for actual in (ir, restored):
            torch.testing.assert_close(
                forced_branch(torch.kron(state, clean), actual.instructions, history),
                expected / (2 ** (events / 2)),
                atol=4e-12,
                rtol=0,
            )


@pytest.mark.parametrize("uncompute", (False, True))
def test_source_controlled_and_boundary_is_not_replaced_by_another_circuit(
    uncompute: bool,
) -> None:
    import cirq
    from qualtran import DecomposeTypeError, QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.block_encoding import Unitary
    from qualtran.bloqs.mcmt import And

    partial = And(uncompute=uncompute).controlled()
    with pytest.raises(DecomposeTypeError, match="non-thru registers"):
        partial.decompose_bloq()
    assert (
        cirq.decompose_once(partial.on(*cirq.LineQubit.range(4)), default=None) is None
    )
    source = Unitary(Add(QUInt(2), QUInt(2)))
    with pytest.raises(ValueError, match="cannot be decomposed"):
        import_block_encoding(source)
    plain = import_block_encoding(source, include_controlled=False)
    assert plain.snapshots[2:] == (None, None)
    with pytest.raises(ValueError, match="not captured"):
        plain.requirements(controlled=True)


@pytest.mark.parametrize("overrides", (False, True))
def test_original_object_serialization_failure_is_separate_from_native_ir(
    overrides: bool,
) -> None:
    from qualtran.bloqs.block_encoding.unitary import (
        _unitary_block_encoding,
        _unitary_block_encoding_properties,
    )
    from qualtran.testing import BloqCheckResult, check_bloq_example_serializes

    example = (
        _unitary_block_encoding_properties if overrides else _unitary_block_encoding
    )
    result, diagnostic = check_bloq_example_serializes(example)
    assert result == BloqCheckResult.FAIL
    assert "Serialization Failed" in diagnostic and "inhomogeneous" in diagnostic


def test_shaped_external_child_uses_original_partition_admission() -> None:
    from dataclasses import dataclass

    import numpy as np
    from numpy.typing import NDArray
    from qualtran import Bloq, BloqBuilder, QBit, Register, Signature
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.block_encoding import Unitary
    from sympy import Symbol

    @dataclass(frozen=True)
    class ShapedX(Bloq):
        shape: tuple[int | Symbol, ...]

        @property
        def signature(self) -> Signature:
            return Signature([Register("q", QBit(), shape=self.shape)])

        def build_composite_bloq(
            self, bb: BloqBuilder, q: NDArray[np.object_]
        ) -> dict[str, NDArray[np.object_]]:
            return {
                "q": np.asarray(
                    [bb.add(XGate(), q=wire) for wire in q.flat], dtype=object
                ).reshape(q.shape)
            }

    for shape in ((1,), (1, 1)):
        source = Unitary(ShapedX(shape))
        block = import_block_encoding(source)
        for controlled, adjoint in product((False, True), repeat=2):
            original = source.controlled() if controlled else source
            if adjoint:
                original = original.adjoint()
            n = 1 + int(controlled)
            operation = block.make_on(
                system=(int(controlled),),
                control=(0,) if controlled else (),
                adjoint=adjoint,
            )
            torch.testing.assert_close(
                get_unitary(
                    Circuit(n).append(operation).to_ir(), dtype=torch.complex128
                ),
                torch.as_tensor(original.tensor_contract(), dtype=torch.complex128),
                atol=2e-12,
                rtol=0,
            )
    for shape in ((2,), (Symbol("k", positive=True, integer=True),)):
        source = Unitary(ShapedX(shape))
        assert source.system_bitsize == 1
        with pytest.raises(ValueError, match="Total bitsize"):
            source.tensor_contract()
        with pytest.raises(ValueError):
            import_block_encoding(source, include_controlled=False)
