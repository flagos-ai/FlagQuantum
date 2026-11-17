"""Audit the locked packed PrepareHubbard factory and original oracle edges."""

from collections.abc import Mapping
from dataclasses import fields
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from sympy import Symbol

from examples.algorithms.hubbard_lcu_catalog import (
    HubbardLCUExample,
    hubbard_packed_prepare_circuit,
)
from flagquantum.algorithms.primitives.black_box_prepare import (
    BlackBoxPrepare,
    PackedPrepareDeclaration,
)
from flagquantum.algorithms.primitives.prepare_oracle import (
    PrepareOracle as NativeOracle,
)
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler._component_trace.resource_model import (
    GateCost,
    estimate_component_resources,
)
from flagquantum.compiler.registers.layout.partition import partition_register
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.register_types import opaque_register_type
from flagquantum.core.ir.program.registers import Register as NativeRegister
from flagquantum.core.value_math import SymbolicReal
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source_module() -> Any:
    import kickmix
    from qualtran.bloqs.state_preparation import black_box_prepare

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(black_box_prepare.__file__).parent
    for name, digest in (
        ("__init__.py", "024a57c221ae73f95645dbaa4247f423b037a932"),
        ("black_box_prepare.py", "42aacba78eb58628d93eee0db860754827d2f42e"),
        ("black_box_prepare_test.py", "8dc0acab59408715563bfd651249bc516162fea5"),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return black_box_prepare


def compare_signature(
    native: BlackBoxPrepare | PackedPrepareDeclaration, source: Any
) -> None:
    for local, pinned in (
        (native.selection_registers, source.selection_registers),
        (native.junk_registers, source.junk_registers),
    ):
        assert [(r.name, r.dtype.num_qubits, r.dtype.encoding) for r in local] == [
            (r.name, r.bitsize, "opaque") for r in pinned
        ]
    assert [
        (r.name, r.dtype.num_qubits, r.side.name, r.dtype.encoding)
        for r in native.signature
    ] == [
        (r.name, r.bitsize, r.side.name, "bit" if r.bitsize == 1 else "opaque")
        for r in source.signature
    ]


def pinned_packed_vector(
    source: Any, logical: int
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    import cirq
    from qualtran.bloqs.mod_arithmetic import ModAddK

    wires = cirq.LineQubit.range(logical)
    context = cirq.DecompositionContext(cirq.GreedyQubitManager(prefix="packed_source"))
    operations = cirq.decompose(
        source.on(*wires),
        context=context,
        keep=lambda op: isinstance(op.gate, ModAddK),
        on_stuck_raise=None,
    )

    def executable(operation: cirq.Operation) -> cirq.Operation:
        # The reference does not decompose this leaf. Use its classical permutation.
        gate = operation.gate
        if not isinstance(gate, ModAddK):
            return operation
        words = 1 << gate.bitsize
        matrix = np.zeros((4 * words, 4 * words))
        for control, word in product(range(4), range(words)):
            row = gate.on_classical_vals(ctrl=control, x=word)
            matrix[control * words + row["x"], control * words + word] = 1
        return cirq.MatrixGate(matrix).on(*operation.qubits)

    circuit = cirq.Circuit(executable(op) for op in operations)
    ancillas = sorted(circuit.all_qubits() - set(wires))
    raw = (
        cirq.Simulator(dtype=np.complex128, seed=17)
        .simulate(circuit, qubit_order=(*wires, *ancillas))
        .final_state_vector.reshape(1 << logical, -1)
    )
    np.testing.assert_allclose(raw[:, 1:], 0, atol=3e-12, rtol=0)
    return np.asarray(raw[:, 0], dtype=np.complex128)


def test_original_packed_hubbard_factory(source_module: Any) -> None:
    from qualtran.bloqs.bookkeeping.auto_partition import AutoPartition
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = source_module._black_box_prepare.make()
    example = HubbardLCUExample(2, packed=True)
    native = example.prepare_oracle
    assert native.normalization == source.l1_norm_of_coeffs
    compare_signature(native, source)
    ((stage, _, _),) = source.decompose_bloq().iter_bloqsoqs()
    assert isinstance(stage.bloq, AutoPartition) and stage.bloq.bloq == source.prepare
    child_layout = example.preparation.selection_layout
    assert [(reg.name, tuple(names)) for reg, names in stage.bloq.partitions] == [
        ("selection", tuple(name for name, _ in child_layout)),
        ("junk", ("temp",)),
    ]
    through = native.selection_width + native.junk_width
    assert (native.selection_width, native.junk_width) == (8, 2)
    model = estimate_component_resources(example.prepare_model())
    expected_cost = get_cost_value(source, QECGatesCost())
    assert model.gates == GateCost(
        **{f.name: getattr(expected_cost, f.name) for f in fields(GateCost)}
    )
    assert model.peak_logical_qubits == get_cost_value(source, QubitCount()) == 12

    state = torch.zeros(4096, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    state[0] = 1
    expected[::4] = torch.as_tensor(pinned_packed_vector(source, through).copy())
    child = example.preparation
    wires = partition_register(tuple(range(through)), child.layout)
    for adjoint in (False, True):
        body = hubbard_packed_prepare_circuit(example, adjoint=adjoint).to_ir()
        named = (
            DynamicCircuit(12)
            .append(child.make_on(wires, workspace=(10, 11), adjoint=adjoint))
            .to_ir()
        )
        assert body.instructions == named.instructions
        literal = count_emitted_instructions(body)
        assert literal.t_count == 47 and literal.declared_qubits == 12
        assert (
            literal.unconditional["measure"] == 3 and literal.unconditional["cx"] == 66
        )
        start, end = (expected, state) if adjoint else (state, expected)
        for history in product((0, 1), repeat=3):
            torch.testing.assert_close(
                forced_branch(start, body.instructions, history),
                end / sqrt(8),
                atol=3e-12,
                rtol=0,
            )


def test_original_dummy_oracle_signature_and_binding(source_module: Any) -> None:
    from qualtran.bloqs.state_preparation.black_box_prepare_test import (
        TestPrepareOracle,
    )

    source = source_module.BlackBoxPrepare(TestPrepareOracle())
    ((stage, _, _),) = source.decompose_bloq().iter_bloqsoqs()
    observed = []

    def bind(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        bits: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        observed.append((ports, work, bits, adjoint))
        return ()

    native = BlackBoxPrepare((("z", 1),), (("a", 5),), bind)
    compare_signature(native, source)
    assert [tuple(names) for _, names in stage.bloq.partitions] == [("z",), ("a",)]
    assert native.normalization is None
    circuit = DynamicCircuit(6)
    circuit.append(native.make_on(selection=(4,), junk=(0, 1, 5, 2, 3)))
    assert observed == [({"z": (4,), "a": (0, 1, 5, 2, 3)}, (), (), False)]


@pytest.mark.parametrize("selection_width,junk_width", [(0, 5), (1, 0), (0, 0)])
def test_zero_width_groups_follow_source(
    selection_width: int,
    junk_width: int,
    source_module: Any,
) -> None:
    from qualtran import QAny, Register, Signature
    from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle

    class Oracle(PrepareOracle):  # type: ignore[misc]
        @property
        def selection_registers(self) -> tuple[Any, ...]:
            return (Register("z", QAny(selection_width)),)

        @property
        def junk_registers(self) -> tuple[Any, ...]:
            return (Register("a", QAny(junk_width)),)

        @property
        def signature(self) -> Any:
            return Signature.build(z=selection_width, a=junk_width)

    source = source_module.BlackBoxPrepare(Oracle())
    stages = tuple(source.decompose_bloq().iter_bloqsoqs())
    assert len(stages) == int(bool(selection_width))
    if stages:
        assert [tuple(names) for _, names in stages[0][0].bloq.partitions] == [("z",)]
    calls = []

    def emit(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        bits: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        calls.append((ports, adjoint))
        return ()

    class DeclarationOracle(NativeOracle):
        @property
        def selection_registers(self) -> tuple[NativeRegister, ...]:
            return (NativeRegister("z", opaque_register_type(selection_width)),)

        @property
        def junk_registers(self) -> tuple[NativeRegister, ...]:
            return (NativeRegister("a", opaque_register_type(junk_width)),)

    declaration = PackedPrepareDeclaration(DeclarationOracle())
    compare_signature(declaration, source)
    with pytest.raises(NotImplementedError):
        _ = declaration.l1_norm_of_coeffs
    native = declaration.bind(emit)
    assert native.normalization is None
    compare_signature(native, source)
    circuit = DynamicCircuit(max(1, selection_width + junk_width))
    for adjoint in (False, True):
        circuit.append(
            native.make_on(
                selection=tuple(range(selection_width)),
                junk=tuple(range(selection_width, selection_width + junk_width)),
                adjoint=adjoint,
            )
        )
    expected = (
        [({"z": (0,)}, adjoint) for adjoint in (False, True)] if selection_width else []
    )
    assert calls == expected
    assert circuit.to_ir().instructions == ()


@pytest.mark.parametrize("shape_width", [2, Symbol("n", integer=True, positive=True)])
def test_packed_shaped_ports_keep_widths_and_norm(
    source_module: Any, shape_width: Any
) -> None:
    from qualtran import QAny, Register
    from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle

    norm = 7.0 if isinstance(shape_width, int) else Symbol("lambda", positive=True)

    class SourceOracle(PrepareOracle):  # type: ignore[misc]
        @property
        def selection_registers(self) -> tuple[Any, ...]:
            return (
                Register("i", QAny(shape_width), shape=(2,)),
                Register("j", QAny(1)),
            )

        @property
        def junk_registers(self) -> tuple[Any, ...]:
            return (Register("work", QAny(shape_width), shape=(3,)),)

        @property
        def l1_norm_of_coeffs(self) -> SymbolicReal:
            return norm

    class DeclarationOracle(NativeOracle):
        @property
        def selection_registers(self) -> tuple[NativeRegister, ...]:
            return (
                NativeRegister("i", opaque_register_type(shape_width), shape=(2,)),
                NativeRegister("j", opaque_register_type(1)),
            )

        @property
        def junk_registers(self) -> tuple[NativeRegister, ...]:
            return (
                NativeRegister("work", opaque_register_type(shape_width), shape=(3,)),
            )

        @property
        def l1_norm_of_coeffs(self) -> SymbolicReal:
            return norm

    native = PackedPrepareDeclaration(DeclarationOracle())
    source = source_module.BlackBoxPrepare(SourceOracle())
    compare_signature(native, source)
    assert native.selection_width == source.selection_bitsize == 2 * shape_width + 1
    assert native.junk_width == source.junk_bitsize == 3 * shape_width
    assert native.l1_norm_of_coeffs == source.l1_norm_of_coeffs == norm
    assert str(native) == str(source) == "Prep"

    def emit(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        bits: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        assert ports == {"i": (4, 3, 2, 1), "j": (0,), "work": (10, 9, 8, 7, 6, 5)}
        assert not work and not bits and not adjoint
        return (Instruction("cx", (ports["i"][0], ports["work"][-1])),)

    if not isinstance(shape_width, int):
        with pytest.raises(ValueError, match="concrete width for i"):
            native.bind(emit)
        return
    box = native.bind(emit)
    assert box.normalization == norm
    circuit = DynamicCircuit(11)
    circuit.append(box.make_on(selection=(4, 3, 2, 1, 0), junk=(10, 9, 8, 7, 6, 5)))
    assert circuit.to_ir().instructions == (Instruction("cx", (4, 5)),)


def test_append_width_validation() -> None:
    native = BlackBoxPrepare((("x", 1),), (), lambda *_: ())
    circuit = DynamicCircuit(1)
    with pytest.raises(ValueError, match="exceeds circuit width"):
        circuit.append(native.make_on(selection=(1,), junk=()))
    assert circuit.to_ir().instructions == ()
